#!/bin/bash
#SBATCH --job-name=e6_routecheck
#SBATCH --account=elmahallawy
#SBATCH --partition=kamiak
#SBATCH --constraint=ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# ── E6 evaluation, step 2a: routing correctness per corpus fraction ────────────
# Runs the 14-condition check_routing.py against each fraction's gate.
# Expected result (the R3.1 headline): 14/14 at EVERY fraction -- routing is
# independent of task-corpus size because the gate never sees task data.
# Writes routing_check_lowres_n{N}.json per fraction + a summary line each.

set -euo pipefail
echo "Running on: $(hostname)"; echo "Start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
if [[ "$(which python)" != *"/envs/fedllm/"* ]]; then
  echo "FATAL: env fedllm not active" >&2; exit 1
fi
nvidia-smi -L

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
LOWRES=$SCRATCH/non_fl_lowres
KEY=FoHL9UFVcTbcy80F5KZd
cd "$SCRATCH/OpenFedLLM"

FAILED=0
for N in 500 1000 1250 2500 5000; do
  CKPT=$SCRATCH/local_models/outputs_1b_lowres_n${N}
  [ -f "$CKPT/gating_module.pt" ] || { echo "SKIP n=${N}: no gate yet at $CKPT"; FAILED=1; continue; }
  OUTJSON=routing_check_lowres_n${N}.json
  if [ -f "$OUTJSON" ]; then echo "[n=${N}] already checked -- skipping"; continue; fi
  echo; echo "================ routing check N=${N} ($(date '+%H:%M:%S')) ================"
  python check_routing.py \
    --base meta-llama/Llama-3.2-1B \
    --ckpt "$CKPT" \
    --adapters defended=$LOWRES/echr-llama1b-scrubbed-n${N} \
               revealing=$LOWRES/echr-llama1b-undefended-n${N} \
    --key "$KEY" --precision bf16 --out "$OUTJSON" \
    || { echo "ROUTING FAILURE at n=${N} -- see $OUTJSON"; FAILED=1; }
done

echo; echo "=== SUMMARY ==="
python - << 'PYEOF2'
import json, glob
rows=[]
for f in sorted(glob.glob("routing_check_lowres_n*.json"), key=lambda x:int(x.split('_n')[1].split('.')[0])):
    d=json.load(open(f)); n=f.split('_n')[1].split('.')[0]
    rows.append((int(n), d['passed'], d['total']))
    print(f"  N={n:>5}: {d['passed']}/{d['total']} passed")
if rows and all(p==t for _,p,t in rows):
    print("ALL FRACTIONS 14/14 -- routing accuracy is corpus-size-independent (R3.1 headline confirmed)")
PYEOF2
echo "End: $(date)"; exit $FAILED
