#!/bin/bash
#SBATCH --job-name=e6_gates
#SBATCH --account=elmahallawy
#SBATCH --partition=kamiak
#SBATCH --constraint=ampere          # A100-40GB (sn14/sn15) -- gate training needs ~10GB; keep
                                     # the whole E6 series on one GPU type for internal consistency
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00               # 5 gates x ~15-30 min + margin
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# ── E6 evaluation, step 1: one gating module per corpus fraction ────────────────
# For each N in {500,1000,1250,2500,5000}: train a 2-adapter gate over that
# fraction's defended (scrubbed) + revealing (undefended) adapters, same key and
# recipe as the verified full-size gate (outputs_1b_new: 700 routing examples,
# 3 epochs, loss->0, entropy->4e-6).
# Idempotent: a fraction whose gating_module.pt already exists is skipped, so
# resubmitting after preemption only fills gaps.
# Expected result (worth stating up front): gate quality should be IDENTICAL
# across fractions -- the routing task never sees task data (R3.1 argument).

set -euo pipefail

echo "Running on: $(hostname)"; echo "Start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
# transformers' save path imports deepspeed, whose import REQUIRES CUDA_HOME --
# without it the job crashes after training, at save time (job 28792659).
module load cuda/11.8.0 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/cuda/11.8.0}"
[ -d "$CUDA_HOME" ] || { echo "FATAL: CUDA_HOME=$CUDA_HOME does not exist" >&2; exit 1; }
echo "CUDA_HOME=$CUDA_HOME"
if [[ "$(which python)" != *"/envs/fedllm/"* ]]; then
  echo "FATAL: conda env fedllm did not activate (python=$(which python))." >&2; exit 1
fi
nvidia-smi -L   # short form: no pipe -- 'nvidia-smi | head' dies of SIGPIPE under pipefail

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
LOWRES=$SCRATCH/non_fl_lowres
BASE=meta-llama/Llama-3.2-1B
KEY=FoHL9UFVcTbcy80F5KZd

cd "$SCRATCH/OpenFedLLM/Mixture-of-LoRA-Experts/dreambooth"

for N in 500 1000 1250 2500 5000; do
  DEF=$LOWRES/echr-llama1b-scrubbed-n${N}
  REV=$LOWRES/echr-llama1b-undefended-n${N}
  OUT=$SCRATCH/local_models/outputs_1b_lowres_n${N}

  for d in "$DEF" "$REV"; do
    [ -f "$d/adapter_config.json" ] || { echo "FATAL: no adapter at $d" >&2; exit 1; }
  done

  if [ -f "$OUT/gating_module.pt" ]; then
    echo "[gate n=${N}] already trained -- skipping"
    continue
  fi

  echo; echo "================ gate for N=${N} rows  ($(date '+%H:%M:%S')) ================"
  accelerate launch train_gating.py \
    --base "$BASE" \
    --adapters defended="$DEF" revealing="$REV" \
    --key-map ${KEY}=1 \
    --output-dir "$OUT" \
    --dataset-size 100 --epochs 3

  [ -f "$OUT/gating_module.pt" ] || { echo "FATAL: n=${N} produced no gating_module.pt" >&2; exit 1; }
  echo "[gate n=${N}] done -> $OUT ($(du -sh "$OUT" | cut -f1))"
done

echo; echo "All 5 gates complete. End: $(date)"
echo "Next: routing check per fraction (check_routing.py), then leakage/perplexity eval."
