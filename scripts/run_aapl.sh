#!/usr/bin/env bash
# Full TLOB experiment on AAPL LOBSTER data:
#   train      2025-03-03 .. 2025-03-27  (19 days)
#   validation 2025-03-28, 2025-03-31    (2 days)
#   test       2025-04-01 .. 2025-04-03  (3 days)
# This is TLOB's own 80/10/10 day split over data/AAPL/AAPL_2025-03-03_2025-04-03/ (24 days).
#
# Runs horizons 10, 20, 50 and 100, each on its own GPU, with TLOB's default model settings.
# The horizon-10 run preprocesses the data first (CPU, ~30-45 min, ~110 GB RAM); the other
# horizons start once data/AAPL/{train,val,test}.npy exist. Training can take many hours.
#
#   bash scripts/run_aapl.sh
#
# Optional: PY=/path/to/python  WANDB=False  GPUS="0 1 2 3"
set -euo pipefail
cd "$(dirname "$0")/.."

STOCK=AAPL
FIRST_DAY=2025-03-03
LAST_DAY=2025-04-03
HORIZONS=(10 20 50 100)
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3}"
WANDB=${WANDB:-True}
DATA=data/$STOCK/${STOCK}_${FIRST_DAY}_${LAST_DAY}

# 1. Python environment (created in ~/tlob-env from requirements.txt if it doesn't exist)
if [ -z "${PY:-}" ]; then
  PY=$HOME/tlob-env/bin/python
  if [ ! -x "$PY" ]; then
    BASE=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)
    [ -n "$BASE" ] || { echo "ERROR: need python 3.10-3.12 for torch 2.5"; exit 1; }
    echo "creating ~/tlob-env with $BASE (one time) ..."
    "$BASE" -m venv "$HOME/tlob-env"
    "$HOME/tlob-env/bin/pip" install -q --upgrade pip
    "$HOME/tlob-env/bin/pip" install -q -r requirements.txt
  fi
fi
if ! "$PY" -c "import run" >/dev/null 2>&1; then
  # the newest huggingface_hub removed HfFolder, which the transformers TLOB installs still imports
  "$(dirname "$PY")/pip" install -q "transformers<4.47" "huggingface_hub<1.0"
  "$PY" -c "import run" >/dev/null || { echo "ERROR: TLOB does not import, run: $PY -c 'import run'"; exit 1; }
fi
"$PY" -c "import torch; print('torch', torch.__version__, '| GPUs visible:', torch.cuda.device_count())"
if [ "$WANDB" = "True" ] && ! "$PY" -c "import wandb, sys; sys.exit(0 if wandb.Api().api_key else 1)" >/dev/null 2>&1; then
  echo "ERROR: wandb is not logged in. Run: $(dirname "$PY")/wandb login   (or start with WANDB=False)"; exit 1
fi

# 2. Data check: 24 days = 48 LOBSTER files, nothing else in the folder
[ -d "$DATA" ] || { echo "ERROR: $DATA not found"; exit 1; }
N=$(ls "$DATA" | grep -c '_10\.csv$' || true)
[ "$N" -eq 48 ] || { echo "ERROR: expected 48 LOBSTER files (24 days) in $DATA, found $N"; exit 1; }
if ls "$DATA" | grep -v '_10\.csv$' | grep -q .; then
  echo "ERROR: $DATA contains files other than LOBSTER csv files:"; ls "$DATA" | grep -v '_10\.csv$'; exit 1
fi
echo "data: $DATA (24 days)"

# 3. Runs: horizon 10 preprocesses, then the other horizons reuse the .npy files
COMMON=(+model=tlob +dataset=lobster hydra.job.chdir=False
        "dataset.dates=[$FIRST_DAY,$LAST_DAY]"
        "dataset.training_stocks=[$STOCK]" "dataset.testing_stocks=[$STOCK]"
        experiment.is_wandb=$WANDB)
mkdir -p logs
log() { echo "logs/${STOCK}_h$1.log"; }

echo "$(date '+%F %T') horizon ${HORIZONS[0]} on GPU ${GPU_IDS[0]} (preprocessing first) -> $(log "${HORIZONS[0]}")"
CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} "$PY" main.py "${COMMON[@]}" experiment.horizon=${HORIZONS[0]} \
  > "$(log "${HORIZONS[0]}")" 2>&1 &
PIDS=($!)
until grep -q "Train set shape" "$(log "${HORIZONS[0]}")" 2>/dev/null; do
  if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
    echo "ERROR: horizon ${HORIZONS[0]} stopped before training, see $(log "${HORIZONS[0]}")"; tail -5 "$(log "${HORIZONS[0]}")"; exit 1
  fi
  sleep 30
done
echo "$(date '+%F %T') preprocessing done"

for i in 1 2 3; do
  h=${HORIZONS[$i]}
  echo "$(date '+%F %T') horizon $h on GPU ${GPU_IDS[$i]} -> $(log "$h")"
  CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} "$PY" main.py "${COMMON[@]}" experiment.horizon=$h \
    experiment.is_data_preprocessed=True > "$(log "$h")" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for i in 0 1 2 3; do
  if wait "${PIDS[$i]}"; then echo "$(date '+%F %T') horizon ${HORIZONS[$i]} finished"
  else echo "$(date '+%F %T') horizon ${HORIZONS[$i]} FAILED, see $(log "${HORIZONS[$i]}")"; FAILED=1; fi
done

# 4. Test results (Apr 1-3)
echo; echo "test results, $STOCK, test days Apr 1-3:"
for h in "${HORIZONS[@]}"; do
  f1=$(grep -a "f1_score" "$(log "$h")" | tail -1 | awk '{print $NF}')
  acc=$(grep -a " accuracy  " "$(log "$h")" | tail -1 | awk '{print $NF}')
  echo "  horizon $h: f1 ${f1:-n/a}  accuracy ${acc:-n/a}"
done
echo "checkpoints: data/checkpoints/TLOB/LOBSTER_['$STOCK']_seq_size_128_horizon_<h>_seed_1/"
exit $FAILED
