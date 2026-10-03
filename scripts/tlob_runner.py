"""
Run TLOB without Hydra's command line (Hydra's @hydra.main crashes on Python 3.14).

A run is described by a "spec" dict. tlob_run(spec) does what main.py does: it builds TLOB's
config from config/config.py with OmegaConf (identical to the config Hydra builds for the same
overrides) and calls TLOB's own LOBSTERDataBuilder and run()/run_wandb().

launch(spec, gpus, log_path) starts a run as its own process on the given GPUs. With more than
one GPU, Lightning (DDP) starts one process per GPU by re-running this file with the same spec.

Used by run_aapl.py and compare_pipelines.py.
"""
import json
import os
import re
import subprocess
import sys
import time

TLOB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SPEC = {
    "mode": "train",               # "train" or "preprocess"
    "stock": "AAPL",
    "first_day": "2025-03-03",
    "last_day": "2025-04-03",
    "horizon": 10,
    "seed": 1,                     # TLOB's default seed
    "wandb": False,
    "gpu_batches": False,          # True: GPUBatchLoader instead of DataLoader
    "num_gpus": 0,                 # > 1: one run on several GPUs (DDP); 0: Lightning's default
    "max_epochs": 10,              # TLOB's default
    "limit_train_batches": 1.0,    # fraction (<= 1) or number of training batches per epoch
    "deterministic": False,        # deterministic torch algorithms (for exact comparisons)
    "checkpoint_dir": "data/checkpoints",   # TLOB's default (constants.DIR_SAVED_MODEL)
    "is_debug": False,
    "model": {},                   # overrides of model.hyperparameters_fixed, e.g. {"num_layers": 1}
}


class _StampedOutput:
    """stdout that prefixes every line with the process rank and seconds since start, so logs of
    multi-GPU runs can be told apart and epoch times read from them: '[r0 12.345] text'."""

    def __init__(self, stream, rank, start):
        self.stream, self.rank, self.start, self.at_line_start = stream, rank, start, True

    def write(self, text):
        if "\r" in text or "\x1b[" in text:      # progress bar updates (Lightning writes them to stdout)
            sys.__stderr__.write(text)
            sys.__stderr__.flush()
            return len(text)
        out = []
        for part in text.splitlines(keepends=True):
            if self.at_line_start:
                out.append(f"[r{self.rank} {time.time() - self.start:.3f}] ")
            out.append(part)
            self.at_line_start = part.endswith("\n")
        self.stream.write("".join(out))
        self.stream.flush()
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tlob_run(spec):
    spec = {**DEFAULT_SPEC, **spec}
    rank = int(os.environ.get("LOCAL_RANK", 0))
    sys.stdout = _StampedOutput(sys.stdout, rank, float(os.environ.get("TLOB_RUN_START", time.time())))
    sys.path.insert(0, TLOB_DIR)
    os.chdir(TLOB_DIR)                       # TLOB uses paths relative to its folder ("data/...")
    import warnings
    warnings.filterwarnings("ignore")        # main.py: same as its first lines
    import multiprocessing
    if "fork" in multiprocessing.get_all_start_methods():
        # DataLoader workers forked (the Linux default up to Python 3.13), not started by forkserver
        # (Python 3.14's default): forkserver workers re-import TLOB's modules and hit the circular
        # import between constants.py and preprocessing/dataset.py
        multiprocessing.set_start_method("fork", force=True)
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import constants as cst
    from config.config import Config, LOBSTER, TLOB
    from preprocessing.lobster import LOBSTERDataBuilder
    from run import run, run_wandb

    cst.DIR_SAVED_MODEL = spec["checkpoint_dir"]

    # main.py: set_torch()
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision("high")
    if spec["deterministic"]:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # the config Hydra would build for +model=tlob +dataset=lobster plus these overrides
    config = OmegaConf.structured(Config(model=TLOB(), dataset=LOBSTER()))
    config.dataset.dates = [spec["first_day"], spec["last_day"]]
    config.dataset.training_stocks = [spec["stock"]]
    config.dataset.testing_stocks = [spec["stock"]]
    config.experiment.horizon = spec["horizon"]
    config.experiment.seed = spec["seed"]
    config.experiment.is_wandb = spec["wandb"]
    config.experiment.is_debug = spec["is_debug"]
    config.experiment.max_epochs = spec["max_epochs"]
    config.experiment.gpu_batches = spec["gpu_batches"]
    config.experiment.num_gpus = spec["num_gpus"]
    config.experiment.limit_train_batches = float(spec["limit_train_batches"])
    config.experiment.is_data_preprocessed = spec["mode"] != "preprocess"
    for key, value in spec["model"].items():
        config.model.hyperparameters_fixed[key] = value

    # main.py: hydra_app()
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    random.seed(config.experiment.seed)
    print("Using device: ", cst.DEVICE, flush=True)
    accelerator = "cpu" if cst.DEVICE == "cpu" else "gpu"
    config.model.hyperparameters_fixed["hidden_dim"] = 46          # main.py's value for TLOB on LOBSTER
    if spec["mode"] == "preprocess":
        LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        ).prepare_save_datasets()
        print("PREPROCESSING DONE", flush=True)
        return
    if config.experiment.is_wandb:
        run_wandb(config, accelerator)()
    else:
        run(config, accelerator)
    print("RUN DONE", flush=True)


def launch(spec, gpus, log_path):
    """Start tlob_run(spec) as its own process on the given GPU ids (a list)."""
    spec = {**DEFAULT_SPEC, **spec}
    if spec["mode"] == "train" and len(gpus) > 1:
        spec["num_gpus"] = len(gpus)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus),
               PYTHONUNBUFFERED="1",
               TLOB_RUN_START=str(time.time()),
               # torch >= 2.6 loads checkpoints with weights_only=True by default, which can refuse TLOB's
               # checkpoints; run.py then silently tests the last epoch instead of the best one
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    if spec["deterministic"]:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # stdout (TLOB's messages, timestamped) -> log_path; stderr (progress bars, warnings, errors) -> log_path.err
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--spec", json.dumps(spec)],
                            cwd=TLOB_DIR, env=env, stdout=open(log_path, "w"), stderr=open(log_path + ".err", "w"))


# ---------------------------------------------------------------------------------------------
# checks and log parsing shared by the scripts
# ---------------------------------------------------------------------------------------------

def say(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def fail(msg):
    say(f"ERROR: {msg}")
    sys.exit(1)


def check_environment(n_gpus, wandb_on):
    try:
        import torch
        import lightning, omegaconf, einops, torch_ema, lion_pytorch, sklearn  # noqa: F401
    except ImportError as e:
        fail(f"{e}. Install TLOB's requirements into this Python:\n"
             f"  {sys.executable} -m pip install -r {os.path.join(TLOB_DIR, 'requirements.txt')}")
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        if "HfFolder" in str(e):
            fail(f"{e}. Fix:\n  {sys.executable} -m pip install 'transformers<4.47' 'huggingface_hub<1.0'")
        raise
    import pandas
    if int(pandas.__version__.split(".")[0]) >= 3:
        # TLOB divides integer price columns in place (orderbook.loc[:, ::2] /= 10000), which pandas 3 rejects
        fail(f"pandas {pandas.__version__} is too new for TLOB's preprocessing. Fix:\n"
             f"  {sys.executable} -m pip install 'pandas>=2.2,<3'")
    say(f"python {sys.version.split()[0]} | torch {torch.__version__} | pandas {pandas.__version__} | "
        f"GPUs visible: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        say("no GPU visible: running on CPU (multi-GPU runs become multi-process CPU runs)")
    elif torch.cuda.device_count() < n_gpus:
        fail(f"{n_gpus} GPUs needed, {torch.cuda.device_count()} visible")
    if wandb_on:
        try:
            import wandb
            if not wandb.Api().api_key:
                raise RuntimeError
        except Exception:
            fail("wandb is not logged in. Run `wandb login` once, or turn wandb off at the top of the script")


def data_folder(stock, first_day, last_day):
    return os.path.join(TLOB_DIR, "data", stock, f"{stock}_{first_day}_{last_day}")


def check_data(stock, first_day, last_day, n_days):
    folder = data_folder(stock, first_day, last_day)
    if not os.path.isdir(folder):
        fail(f"{folder} not found")
    files = sorted(os.listdir(folder))
    other = [f for f in files if not f.endswith("_10.csv")]
    if other:
        fail(f"{folder} contains files other than LOBSTER csv files (TLOB pairs files by sorted name): {other}")
    if len(files) != 2 * n_days:
        fail(f"expected {2 * n_days} LOBSTER files ({n_days} days) in {folder}, found {len(files)}")
    say(f"data: {folder} ({n_days} days)")


def preprocessed_exists(stock, first_day, last_day):
    """data/<stock>/{train,val,test}.npy exist and are newer than every LOBSTER csv file."""
    npys = [os.path.join(TLOB_DIR, "data", stock, f"{s}.npy") for s in ("train", "val", "test")]
    if not all(os.path.isfile(f) for f in npys):
        return False
    folder = data_folder(stock, first_day, last_day)
    newest_csv = max(os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder))
    return min(os.path.getmtime(f) for f in npys) > newest_csv


def ensure_preprocessed(spec, reuse, log_dir):
    """Run TLOB's preprocessing once (CPU, own process) unless reusable .npy files exist."""
    if reuse and preprocessed_exists(spec["stock"], spec["first_day"], spec["last_day"]):
        say(f"reusing preprocessed data/{spec['stock']}/{{train,val,test}}.npy")
        return
    log_path = os.path.join(log_dir, f"{spec['stock']}_preprocess.log")
    say(f"preprocessing {spec['stock']} (CPU, takes a while) -> {log_path}")
    proc = launch({**spec, "mode": "preprocess"}, [], log_path)
    if proc.wait() != 0:
        print(tail(log_path))
        fail(f"preprocessing failed, see {log_path}")
    say("preprocessing done")


def read_log(log_path):
    if not os.path.isfile(log_path):
        return []
    with open(log_path, errors="replace") as f:
        return f.read().replace("\r", "\n").splitlines()


def tail(log_path, n=15):
    """Last lines of a run's log and of its stderr file (where Python errors end up)."""
    err = [line for line in read_log(log_path + ".err") if line.strip()]
    return "\n".join(read_log(log_path)[-n:] + ["--- " + log_path + ".err:"] + err[-n:])


def last_value(log_path, name):
    """Last value of a test metric line such as '  f1_score   0.42' or '│ f1_score │ 0.42 │'."""
    value = None
    for line in read_log(log_path):
        m = re.search(rf"\b{name}\b[\s│|]+([0-9]+\.[0-9]+)", line)
        if m:
            value = float(m.group(1))
    return value


def epochs(log_path):
    """Per epoch (first process only): learning rate, train loss, validation loss and seconds from
    the epoch's start to the end of its validation."""
    out, start, lr = {}, None, None
    for line in read_log(log_path):
        m = re.match(r"\[r0 ([0-9.]+)\] (.*)", line)
        if not m:
            continue
        t, text = float(m.group(1)), m.group(2)
        if text.startswith("learning rate:"):
            start, lr = t, float(text.split(":")[1])
        elif text.startswith("Train loss on epoch"):
            e = int(re.search(r"epoch (\d+)", text).group(1))
            out.setdefault(e, {})["train_loss"] = float(text.split(":")[1])
        elif text.startswith("Validation loss on epoch"):
            e = int(re.search(r"epoch (\d+)", text).group(1))
            out.setdefault(e, {}).update(val_loss=float(text.split(":")[1]), lr=lr,
                                         seconds=None if start is None else t - start)
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--spec":
        tlob_run(json.loads(sys.argv[2]))
    else:
        sys.exit("used by run_aapl.py and compare_pipelines.py: tlob_runner.py --spec '<json>'")
