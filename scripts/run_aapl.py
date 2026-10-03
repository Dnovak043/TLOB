"""
Full TLOB experiment on AAPL LOBSTER data.

    train      2025-03-03 .. 2025-03-27  (19 days)
    validation 2025-03-28, 2025-03-31    (2 days)
    test       2025-04-01 .. 2025-04-03  (3 days)

This is TLOB's own 80/10/10 day split over data/AAPL/AAPL_2025-03-03_2025-04-03/ (24 days).
Each horizon is a normal TLOB run (main.py with Hydra overrides) on its own GPU. The first
horizon preprocesses the data (CPU, ~30-45 min, ~110 GB RAM); the others start once
data/AAPL/{train,val,test}.npy exist and reuse them.

Run from anywhere, with the Python environment that has TLOB's requirements installed:

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


def check_environment():
    if sys.version_info >= (3, 13):
        fail(f"Python {sys.version.split()[0]} is too new: Hydra crashes on 3.14 ('badly formed help string') and "
             f"TLOB's torch 2.5 supports up to 3.12. Create the environment with Python 3.11:\n"
             f"  uv venv ~/tlob-env --python 3.11 && uv pip install -p ~/tlob-env -r {os.path.join(TLOB_DIR, 'requirements.txt')}\n"
             f"then run: ~/tlob-env/bin/python {os.path.abspath(__file__)}")
    try:
        import torch
        import lightning, hydra, einops, torch_ema, lion_pytorch  # noqa: F401
    except ImportError as e:
        fail(f"{e}. Install TLOB's requirements into this Python:\n"
             f"  {sys.executable} -m pip install -r {os.path.join(TLOB_DIR, 'requirements.txt')}")
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        if "HfFolder" in str(e):
            fail(f"{e}. Fix:\n  {sys.executable} -m pip install 'transformers<4.47' 'huggingface_hub<1.0'")
        raise
    say(f"python {sys.executable} | torch {torch.__version__} | GPUs visible: {torch.cuda.device_count()}")
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


def start_run(horizon, gpu, preprocessed):
    cmd = [sys.executable, "main.py", "+model=tlob", "+dataset=lobster", "hydra.job.chdir=False",
           f"dataset.dates=[{FIRST_DAY},{LAST_DAY}]",
           f"dataset.training_stocks=[{STOCK}]", f"dataset.testing_stocks=[{STOCK}]",
           f"experiment.is_wandb={WANDB}", f"experiment.horizon={horizon}",
           f"experiment.is_data_preprocessed={preprocessed}"]
    log_path = os.path.join(LOG_DIR, f"{STOCK}_h{horizon}.log")
    log = open(log_path, "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))   # one GPU per run
    proc = subprocess.Popen(cmd, cwd=TLOB_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    say(f"horizon {horizon} started on GPU {gpu} -> {log_path}")
    return proc, log_path


def last_value(log_path, name):
    """Last value of a Lightning test metric line such as '  f1_score   0.42'."""
    value = None
    with open(log_path, errors="replace") as f:
        for line in f.read().replace("\r", "\n").splitlines():
            m = re.match(rf"\s*{name}\s+([0-9.]+)\s*$", line)
            if m:
                value = m.group(1)
    return value


def main():
    if len(GPUS) < len(HORIZONS):
        fail("GPUS needs one GPU per horizon")
    check_environment()
    check_data()
    os.makedirs(LOG_DIR, exist_ok=True)

    # first horizon: TLOB preprocesses, then trains
    runs = {}
    proc, log_path = start_run(HORIZONS[0], GPUS[0], preprocessed=False)
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
        runs[horizon] = start_run(horizon, gpu, preprocessed=True)

    failed = []
    for horizon, (proc, log_path) in runs.items():
        if proc.wait() == 0:
            say(f"horizon {horizon} finished")
        else:
            say(f"horizon {horizon} FAILED (exit {proc.returncode}), see {log_path}")
            failed.append(horizon)

    print(f"\nTest results, {STOCK}, test days = last 3 days up to {LAST_DAY}:")
    for horizon, (_, log_path) in runs.items():
        print(f"  horizon {horizon:>3}: f1 {last_value(log_path, 'f1_score') or 'n/a'}  "
              f"accuracy {last_value(log_path, 'accuracy') or 'n/a'}")
    print(f"checkpoints: {os.path.join(TLOB_DIR, 'data', 'checkpoints', 'TLOB')}/"
          f"LOBSTER_['{STOCK}']_seq_size_128_horizon_<h>_seed_1/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
