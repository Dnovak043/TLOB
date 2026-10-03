"""
Compare TLOB's original data pipeline with the GPU-batch pipeline, on one GPU and on several.

    python scripts/compare_pipelines.py

1. Batch check (no training): TLOB's Dataset + DataLoader and the new GPUBatchLoader, started
   from the same seed, must produce byte-identical batches, in the order Lightning uses them
   (epoch 1 training, validation, epoch 2 training).
2. Three TLOB trainings with the same seed and settings, deterministic algorithms on:
       A  original DataLoader pipeline, 1 GPU   (what the previous runs used)
       B  GPU batches, 1 GPU                    (expected: identical to A)
       C  GPU batches, several GPUs (DDP)       (expected: same up to floating-point rounding)
3. Report: per-epoch learning rate / train loss / validation loss / time, test metrics, how many
   test predictions agree between runs, and the speedup. Also written to logs/compare/report.txt.
   If the previous production run's log exists (PREVIOUS_LOG), its results are shown for reference.

Training uses MAX_EPOCHS epochs of LIMIT_TRAIN_BATCHES batches each (the same for every run)
so the comparison finishes in reasonable time; validation and test always use the full sets.
"""

# ============================================================================================
# CONFIGURATION
# ============================================================================================

STOCK = "AAPL"
FIRST_DAY = "2025-03-03"
LAST_DAY = "2025-04-03"
N_DAYS = 24
HORIZON = 10
SEED = 1                          # TLOB's default seed, used by every run

MAX_EPOCHS = 2                    # epochs per run (2 also checks the second epoch's shuffle and lr logic)
LIMIT_TRAIN_BATCHES = 20000       # training batches per epoch (batch 128); 1.0 = full epoch
DETERMINISTIC = True              # deterministic torch algorithms, so A and B can match exactly

SINGLE_GPU = 0                    # GPU for runs A and B and the batch check
DDP_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]   # GPUs for run C (batch 128 must divide by their number)
CHECK_BATCHES = 300               # batches compared per epoch in the batch check
REUSE_PREPROCESSED = True
PREVIOUS_LOG = "logs/AAPL_h10.log"    # previous production run of this horizon, shown for reference
MODEL = {}                        # overrides of TLOB's model settings, e.g. {"num_layers": 1}; {} = TLOB's defaults

# ============================================================================================

import json
import os

# settings can also be overridden without editing this file: TLOB_COMPARE_CONFIG='{"HORIZON": 20}'
globals().update(json.loads(os.environ.get("TLOB_COMPARE_CONFIG", "{}")))
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlob_runner as R

OUT = os.path.join(R.TLOB_DIR, "logs", "compare")


def check_batches():
    """Runs in its own process on SINGLE_GPU: compare DataLoader batches with GPUBatchLoader batches."""
    sys.path.insert(0, R.TLOB_DIR)
    os.chdir(R.TLOB_DIR)
    import multiprocessing
    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import constants as cst
    from config.config import TLOB, LOBSTER
    from preprocessing.lobster import lobster_load
    from preprocessing.dataset import Dataset, DataModule, GPUBatchLoader

    seq = MODEL.get("seq_size", TLOB().hyperparameters_fixed["seq_size"])
    batch = LOBSTER().batch_size
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    x, y = lobster_load(f"data/{STOCK}/train.npy", True, cst.LEN_SMOOTH, HORIZON, seq)
    vx, vy = lobster_load(f"data/{STOCK}/val.npy", True, cst.LEN_SMOOTH, HORIZON, seq)

    def take(loader, k):
        out = []
        for i, (a, b) in enumerate(loader):
            if i == k:
                break
            out.append((a.to(device), b.to(device)))
        return out

    def batches(train, val):
        torch.manual_seed(SEED)
        return take(train, CHECK_BATCHES) + take(val, CHECK_BATCHES) + take(train, CHECK_BATCHES)

    # original pipeline exactly as run.py builds it
    dm = DataModule(train_set=Dataset(x, y, seq), val_set=Dataset(vx, vy, seq),
                    batch_size=batch, test_batch_size=batch * 4, num_workers=4)
    old = batches(dm.train_dataloader(), dm.val_dataloader())
    new = batches(GPUBatchLoader(x, y, seq, batch, shuffle=True, is_train=True),
                  GPUBatchLoader(vx, vy, seq, batch, shuffle=False, is_train=False))
    same = sum(torch.equal(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(old, new))
    print(json.dumps({"compared": len(old), "identical": int(same), "batch_shape": list(old[0][0].shape)}))


def run_training(name, gpu_batches, gpus):
    spec = {"stock": STOCK, "first_day": FIRST_DAY, "last_day": LAST_DAY, "horizon": HORIZON, "seed": SEED,
            "wandb": False, "gpu_batches": gpu_batches, "max_epochs": MAX_EPOCHS,
            "limit_train_batches": LIMIT_TRAIN_BATCHES, "deterministic": DETERMINISTIC,
            "checkpoint_dir": f"data/checkpoints_compare/{name}", "model": MODEL}
    shutil.rmtree(os.path.join(R.TLOB_DIR, spec["checkpoint_dir"]), ignore_errors=True)
    log_path = os.path.join(OUT, f"{name}.log")
    R.say(f"run {name}: GPU(s) {gpus} -> {log_path}")
    started = time.time()
    proc = R.launch(spec, gpus, log_path)
    code = proc.wait()
    seconds = time.time() - started
    if code != 0:
        R.say(f"run {name} FAILED (exit {code})\n{R.tail(log_path)}")
    pred = os.path.join(R.TLOB_DIR, spec["checkpoint_dir"], "TLOB",
                        f"LOBSTER_['{STOCK}']_seq_size_{MODEL.get('seq_size', 128)}_horizon_{HORIZON}_seed_{SEED}",
                        "predictions.npy")
    return {"name": name, "log": log_path, "ok": code == 0, "seconds": seconds,
            "epochs": R.epochs(log_path), "pred": pred,
            "test": {k: R.last_value(log_path, k) for k in ("test_loss", "f1_score", "accuracy")}}


def agreement(a, b):
    import numpy as np
    if not (os.path.isfile(a["pred"]) and os.path.isfile(b["pred"])):
        return "n/a"
    pa, pb = np.load(a["pred"]), np.load(b["pred"])
    if pa.shape != pb.shape:
        return f"different lengths {pa.shape} vs {pb.shape}"
    return f"{(pa == pb).mean():.4%} of {len(pa):,} test predictions identical"


def max_diff(a, b, key):
    diffs = [abs(a["epochs"][e][key] - b["epochs"][e][key]) for e in a["epochs"]
             if e in b["epochs"] and key in a["epochs"][e] and key in b["epochs"][e]]
    return max(diffs) if diffs else None


def main():
    os.makedirs(OUT, exist_ok=True)
    if 128 % len(DDP_GPUS):
        R.fail(f"batch 128 must divide by the number of DDP_GPUS ({len(DDP_GPUS)})")
    R.check_environment(max(len(DDP_GPUS), SINGLE_GPU + 1), False)
    R.check_data(STOCK, FIRST_DAY, LAST_DAY, N_DAYS)
    R.ensure_preprocessed({"stock": STOCK, "first_day": FIRST_DAY, "last_day": LAST_DAY, "seed": SEED},
                          REUSE_PREPROCESSED, OUT)

    R.say(f"batch check: {CHECK_BATCHES} batches each of epoch 1, validation, epoch 2 ...")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(SINGLE_GPU))
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--check-batches"], env=env,
                         capture_output=True, text=True)
    try:
        check = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        R.fail(f"batch check failed:\n{out.stdout[-2000:]}\n{out.stderr[-3000:]}")
    R.say(f"batch check: {check['identical']}/{check['compared']} batches identical")

    runs = [run_training("A_original_1gpu", False, [SINGLE_GPU]),
            run_training("B_gpu_batches_1gpu", True, [SINGLE_GPU]),
            run_training(f"C_gpu_batches_{len(DDP_GPUS)}gpus", True, DDP_GPUS)]

    lines = [f"TLOB pipeline comparison: {STOCK} {FIRST_DAY}..{LAST_DAY}, horizon {HORIZON}, seed {SEED}, "
             f"{MAX_EPOCHS} epochs x {LIMIT_TRAIN_BATCHES} training batches, deterministic={DETERMINISTIC}", "",
             f"Batch check (DataLoader vs GPUBatchLoader, same seed): {check['identical']}/{check['compared']} "
             f"batches byte-identical (shape {check['batch_shape']})", ""]
    for r in runs:
        lines.append(f"{r['name']}: {'ok' if r['ok'] else 'FAILED'}, total {r['seconds'] / 60:.1f} min")
        for e, v in sorted(r["epochs"].items()):
            secs = f"{v['seconds'] / 60:.1f} min" if v.get("seconds") is not None else "n/a"
            lines.append(f"  epoch {e}: lr {v.get('lr')}  train loss {v.get('train_loss')}  "
                         f"val loss {v.get('val_loss')}  time (train+val) {secs}")
        lines.append(f"  test: loss {r['test']['test_loss']}  f1 {r['test']['f1_score']}  accuracy {r['test']['accuracy']}")
    lines.append("")
    a, b, c = runs
    lines.append(f"Note: in {c['name']} the printed train loss is the first GPU's average over its share of each "
                 f"batch, so it differs from A/B even when the model updates are the same; compare val loss, "
                 f"test metrics and predictions instead.")
    lines.append("")
    for x, y, expect in [(a, b, "identical"), (b, c, "equal up to floating-point rounding"), (a, c, "close")]:
        lines.append(f"{x['name']} vs {y['name']} (expected {expect}):")
        lines.append(f"  max |train loss difference| {max_diff(x, y, 'train_loss')}, "
                     f"max |val loss difference| {max_diff(x, y, 'val_loss')}")
        lines.append(f"  {agreement(x, y)}")
        lines.append(f"  test f1 {x['test']['f1_score']} vs {y['test']['f1_score']}")
    lines.append("")
    epoch_time = {r["name"]: [v["seconds"] for v in r["epochs"].values() if v.get("seconds")] for r in runs}
    if all(epoch_time.values()):
        base = sum(epoch_time[a["name"]]) / len(epoch_time[a["name"]])
        for r in runs:
            t = sum(epoch_time[r["name"]]) / len(epoch_time[r["name"]])
            lines.append(f"speed {r['name']}: {t / 60:.1f} min per epoch (train+val), {base / t:.1f}x vs A")
    prev = os.path.join(R.TLOB_DIR, PREVIOUS_LOG)
    if os.path.isfile(prev):
        lines += ["", f"Previous production run ({PREVIOUS_LOG}, full epochs, not deterministic): "
                      f"test f1 {R.last_value(prev, 'f1_score')}, accuracy {R.last_value(prev, 'accuracy')}"]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write(report + "\n")
    R.say(f"report: {os.path.join(OUT, 'report.txt')}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--check-batches":
        check_batches()
    else:
        main()
