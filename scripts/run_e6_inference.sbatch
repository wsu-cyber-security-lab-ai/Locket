#!/bin/bash
#SBATCH --job-name=e6_infer
#SBATCH --account=elmahallawy
#SBATCH --partition=kamiak
#SBATCH --constraint=ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# ── E6 evaluation, step 2b: inference (candidate-ranking) attack per fraction ──
# perplexity_inference, 50 candidates, PERSON class, ECHR, 100 target sequences.
# evaluate.py iterates the correct key (FoHL9...) and a wrong key (iQ3p7...), so each
# run yields BOTH access conditions -> correct-key vs wrong-key accuracy per fraction.
# Result -> accuracy_results_lowres_n{N}.csv (appended) in the examples/ dir.
# Idempotent: a fraction whose CSV already has a row for its tag is skipped.

set -euo pipefail
echo "Running on: $(hostname)"; echo "Start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
if [[ "$(which python)" != *"/envs/fedllm/"* ]]; then
  echo "FATAL: env fedllm not active" >&2; exit 1
fi
module load cuda/11.8.0 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/cuda/11.8.0}"
nvidia-smi -L

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
REPO=$SCRATCH/OpenFedLLM
export PYTHONPATH="$REPO/analysing_pii_leakage/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/analysing_pii_leakage/examples"

for N in 500 1000 1250 2500 5000; do
  CFG=../configs/evaluate/inference-lowres/n${N}.yml
  CSV=accuracy_results_lowres_n${N}.csv
  [ -f "$CFG" ] || { echo "FATAL: missing $CFG" >&2; exit 1; }
  if [ -f "$CSV" ] && grep -q "lowres_n${N}" "$CSV"; then
    echo "[infer n=${N}] result already in $CSV -- skipping"; continue
  fi
  echo; echo "================ inference N=${N} ($(date '+%H:%M:%S')) ================"
  python evaluate.py --config_path "$CFG"
  echo "[infer n=${N}] done -> $CSV"
done

echo; echo "=== SUMMARY (correct key = FoHL9..., wrong key = iQ3p7...) ==="
for N in 500 1000 1250 2500 5000; do
  CSV=accuracy_results_lowres_n${N}.csv
  [ -f "$CSV" ] && { echo "-- N=${N}"; cat "$CSV"; }
done
echo "End: $(date)"
