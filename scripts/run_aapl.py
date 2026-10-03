"""
Full TLOB experiment on AAPL LOBSTER data (without Hydra's command line, see tlob_runner.py).

    train      2025-03-03 .. 2025-03-27  (19 days)
    validation 2025-03-28, 2025-03-31    (2 days)
    test       2025-04-01 .. 2025-04-03  (3 days)

This is TLOB's own 80/10/10 day split over data/AAPL/AAPL_2025-03-03_2025-04-03/ (24 days).
Preprocessing runs once (CPU, ~30-45 min, ~110 GB RAM) unless its output already exists; then one
TLOB run per horizon. With GPUS_PER_RUN > 1 each run uses several GPUs (DDP) and the horizons
take turns; with GPUS_PER_RUN = 1 every horizon runs at the same time on its own GPU.

    python scripts/run_aapl.py
"""

# ============================================================================================
# CONFIGURATION
# ============================================================================================

STOCK = "AAPL"
FIRST_DAY = "2025-03-03"          # first day of data/AAPL/AAPL_<FIRST_DAY>_<LAST_DAY>/
LAST_DAY = "2025-04-03"           # last day (the last 3 days are the test set)
N_DAYS = 24                       # trading days expected in that folder

HORIZONS = [10, 20, 50, 100]      # one TLOB run per horizon
SEED = 1                          # TLOB's default seed
WANDB = True                      # log to wandb (needs `wandb login` once on this machine)

GPU_BATCHES = True                # True: batches built on the GPU (fast); False: TLOB's original DataLoader
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]   # GPUs to use
GPUS_PER_RUN = 8                  # GPUs per run: > 1 = multi-GPU (DDP) runs one after another, 1 = one GPU per run
REUSE_PREPROCESSED = True         # skip preprocessing if data/<STOCK>/{train,val,test}.npy are newer than the csv files

# ============================================================================================

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlob_runner as R

LOG_DIR = os.path.join(R.TLOB_DIR, "logs")


def main():
    if GPUS_PER_RUN < 1 or len(GPUS) < GPUS_PER_RUN:
        R.fail("GPUS_PER_RUN must be between 1 and the number of GPUS")
    if GPUS_PER_RUN > 1 and not GPU_BATCHES:
        R.fail("multi-GPU runs need GPU_BATCHES = True (TLOB's original DataLoader path is not multi-GPU safe)")
    R.check_environment(len(GPUS), WANDB)
    R.check_data(STOCK, FIRST_DAY, LAST_DAY, N_DAYS)
    base = {"stock": STOCK, "first_day": FIRST_DAY, "last_day": LAST_DAY, "seed": SEED,
            "wandb": WANDB, "gpu_batches": GPU_BATCHES}
    R.ensure_preprocessed(base, REUSE_PREPROCESSED, LOG_DIR)

    # GPU groups; each run takes a free group, horizons wait for one
    groups = [GPUS[i:i + GPUS_PER_RUN] for i in range(0, len(GPUS) - GPUS_PER_RUN + 1, GPUS_PER_RUN)]
    waiting, running, done, failed = list(HORIZONS), {}, {}, []
    while waiting or running:
        busy = [g for _, g, _ in running.values()]
        for group in [g for g in groups if g not in busy]:
            if not waiting:
                break
            horizon = waiting.pop(0)
            log_path = os.path.join(LOG_DIR, f"{STOCK}_h{horizon}.log")
            running[horizon] = (R.launch({**base, "horizon": horizon}, group, log_path), group, log_path)
            R.say(f"horizon {horizon} started on GPU(s) {group} -> {log_path}")
        for horizon, (proc, group, log_path) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[horizon]
            done[horizon] = log_path
            if proc.returncode == 0:
                R.say(f"horizon {horizon} finished")
            else:
                R.say(f"horizon {horizon} FAILED (exit {proc.returncode}), see {log_path}\n{R.tail(log_path)}")
                failed.append(horizon)
        time.sleep(10)

    print(f"\nTest results, {STOCK}, test days = last 3 days up to {LAST_DAY}:")
    for horizon in HORIZONS:
        log_path = done[horizon]
        print(f"  horizon {horizon:>3}: f1 {R.last_value(log_path, 'f1_score') or 'n/a'}  "
              f"accuracy {R.last_value(log_path, 'accuracy') or 'n/a'}")
    print(f"checkpoints: {os.path.join(R.TLOB_DIR, 'data', 'checkpoints', 'TLOB')}/"
          f"LOBSTER_['{STOCK}']_seq_size_128_horizon_<h>_seed_{SEED}/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
