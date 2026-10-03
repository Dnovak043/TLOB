"""
Full TLOB experiment on AAPL LOBSTER data, without Hydra's command line.

    train      2025-03-03 .. 2025-03-27  (19 days)
    validation 2025-03-28, 2025-03-31    (2 days)
    test       2025-04-01 .. 2025-04-03  (3 days)

This is TLOB's own 80/10/10 day split over data/AAPL/AAPL_2025-03-03_2025-04-03/ (24 days).

Hydra's command line (@hydra.main in main.py) crashes on Python 3.14 ("badly formed help
string"), so this program does what main.py does without it: it builds TLOB's config from
config/config.py with OmegaConf (the same config object Hydra would build) and calls TLOB's
own LOBSTERDataBuilder and run()/run_wandb(). TLOB's code is not changed.

Each horizon runs in its own process on its own GPU. The first horizon preprocesses the data
(CPU, ~30-45 min, ~110 GB RAM); the others start once data/AAPL/{train,val,test}.npy exist.

    python scripts/run_aapl.py
"""

# ============================================================================================
# CONFIGURATION
# ============================================================================================

STOCK = "AAPL"
FIRST_DAY = "2025-03-03"          # first day of data/AAPL/AAPL_<FIRST_DAY>_<LAST_DAY>/
LAST_DAY = "2025-04-03"           # last day (the last 3 days are the test set)
N_DAYS = 24                       # trading days expected in that folder

HORIZONS = [10, 20, 50, 100]      # one TLOB run per horizon; the first one preprocesses
GPUS = [0, 1, 2, 3]               # GPU for each horizon, same order
WANDB = True                      # log to wandb (needs `wandb login` once on this machine)
REUSE_PREPROCESSED = True         # skip preprocessing if data/<STOCK>/{train,val,test}.npy are newer than the csv files

# ============================================================================================

import os
import re
import subprocess
import sys
import time
from datetime import datetime

TLOB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(TLOB_DIR, "data", STOCK, f"{STOCK}_{FIRST_DAY}_{LAST_DAY}")
LOG_DIR = os.path.join(TLOB_DIR, "logs")


def say(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def fail(msg):
    say(f"ERROR: {msg}")
    sys.exit(1)


# --------------------------------------------------------------------------------------------
# one TLOB run (runs in its own process, started by main() below)
# --------------------------------------------------------------------------------------------

def tlob_run(horizon, preprocess):
    """What main.py does for one run, with the config built directly instead of by Hydra."""
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

    # main.py: set_torch()
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision("high")

    # the config Hydra would build for: +model=tlob +dataset=lobster plus these overrides
    config = OmegaConf.structured(Config(model=TLOB(), dataset=LOBSTER()))
    config.dataset.dates = [FIRST_DAY, LAST_DAY]
    config.dataset.training_stocks = [STOCK]
    config.dataset.testing_stocks = [STOCK]
    config.experiment.horizon = horizon
    config.experiment.is_wandb = WANDB
    config.experiment.is_data_preprocessed = not preprocess

    # main.py: hydra_app()
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    random.seed(config.experiment.seed)
    print("Using device: ", cst.DEVICE, flush=True)
    accelerator = "cpu" if cst.DEVICE == "cpu" else "gpu"
    config.model.hyperparameters_fixed["hidden_dim"] = 46          # main.py's value for TLOB on LOBSTER
    if not config.experiment.is_data_preprocessed:
        LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        ).prepare_save_datasets()
    if config.experiment.is_wandb:
        run_wandb(config, accelerator)()
    else:
        run(config, accelerator)


# --------------------------------------------------------------------------------------------
# the experiment: checks, then one process per horizon
# --------------------------------------------------------------------------------------------

def check_environment():
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
    if torch.cuda.device_count() < len(GPUS):
        fail(f"{len(GPUS)} GPUs needed ({GPUS}), {torch.cuda.device_count()} visible")
    if WANDB:
        try:
            import wandb
            if not wandb.Api().api_key:
                raise RuntimeError
        except Exception:
            fail("wandb is not logged in. Run `wandb login` once, or set WANDB = False at the top of this file")


def check_data():
    if not os.path.isdir(DATA):
        fail(f"{DATA} not found")
    files = sorted(os.listdir(DATA))
    lobster = [f for f in files if f.endswith("_10.csv")]
    other = [f for f in files if not f.endswith("_10.csv")]
    if other:
        fail(f"{DATA} contains files other than LOBSTER csv files (TLOB pairs files by sorted name): {other}")
    if len(lobster) != 2 * N_DAYS:
        fail(f"expected {2 * N_DAYS} LOBSTER files ({N_DAYS} days) in {DATA}, found {len(lobster)}")
    say(f"data: {DATA} ({N_DAYS} days)")


def preprocessed_exists():
    """data/<STOCK>/{train,val,test}.npy exist and are newer than every LOBSTER csv file."""
    npys = [os.path.join(TLOB_DIR, "data", STOCK, f"{s}.npy") for s in ("train", "val", "test")]
    if not all(os.path.isfile(f) for f in npys):
        return False
    newest_csv = max(os.path.getmtime(os.path.join(DATA, f)) for f in os.listdir(DATA))
    return min(os.path.getmtime(f) for f in npys) > newest_csv


def start_run(horizon, gpu, preprocess):
    log_path = os.path.join(LOG_DIR, f"{STOCK}_h{horizon}.log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",   # one GPU per run
               # torch >= 2.6 loads checkpoints with weights_only=True by default, which can refuse TLOB's
               # checkpoints; run.py then silently tests the last epoch instead of the best one
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--run", str(horizon), str(int(preprocess))],
                            cwd=TLOB_DIR, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    say(f"horizon {horizon} started on GPU {gpu} -> {log_path}")
    return proc, log_path


def last_value(log_path, name):
    """Last value of a Lightning test metric line such as '  f1_score   0.42' or '│ f1_score │ 0.42 │'."""
    value = None
    with open(log_path, errors="replace") as f:
        for line in f.read().replace("\r", "\n").splitlines():
            m = re.search(rf"\b{name}\b[\s│|]+([0-9]+\.[0-9]+)", line)
            if m:
                value = m.group(1)
    return value


def main():
    if len(GPUS) < len(HORIZONS):
        fail("GPUS needs one GPU per horizon")
    check_environment()
    check_data()
    os.makedirs(LOG_DIR, exist_ok=True)

    runs = {}
    if REUSE_PREPROCESSED and preprocessed_exists():
        say(f"reusing preprocessed data/{STOCK}/{{train,val,test}}.npy (set REUSE_PREPROCESSED = False to redo)")
        for horizon, gpu in zip(HORIZONS, GPUS):
            runs[horizon] = start_run(horizon, gpu, preprocess=False)
        return finish(runs)

    # first horizon: TLOB preprocesses, then trains
    proc, log_path = start_run(HORIZONS[0], GPUS[0], preprocess=True)
    runs[HORIZONS[0]] = (proc, log_path)
    say("preprocessing (this takes a while) ...")
    while True:
        with open(log_path, errors="replace") as f:
            if "Train set shape" in f.read():
                break
        if proc.poll() is not None:
            with open(log_path, errors="replace") as f:
                tail = f.read()[-2000:]
            fail(f"horizon {HORIZONS[0]} stopped before training, see {log_path}\n{tail}")
        time.sleep(30)
    say("preprocessing done")

    # other horizons reuse data/<STOCK>/{train,val,test}.npy
    for horizon, gpu in zip(HORIZONS[1:], GPUS[1:]):
        runs[horizon] = start_run(horizon, gpu, preprocess=False)
    finish(runs)


def finish(runs):
    failed = []
    for horizon, (proc, log_path) in runs.items():
        if proc.wait() == 0:
            say(f"horizon {horizon} finished")
        else:
            say(f"horizon {horizon} FAILED (exit {proc.returncode}), see {log_path}")
            with open(log_path, errors="replace") as f:
                print("\n".join(f.read().replace("\r", "\n").splitlines()[-15:]), flush=True)
            failed.append(horizon)

    print(f"\nTest results, {STOCK}, test days = last 3 days up to {LAST_DAY}:")
    for horizon, (_, log_path) in runs.items():
        print(f"  horizon {horizon:>3}: f1 {last_value(log_path, 'f1_score') or 'n/a'}  "
              f"accuracy {last_value(log_path, 'accuracy') or 'n/a'}")
    print(f"checkpoints: {os.path.join(TLOB_DIR, 'data', 'checkpoints', 'TLOB')}/"
          f"LOBSTER_['{STOCK}']_seq_size_128_horizon_<h>_seed_1/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--run":
        tlob_run(int(sys.argv[2]), bool(int(sys.argv[3])))
    else:
        main()
