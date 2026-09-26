#!/bin/bash
#SBATCH --job-name=e6_ppl
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

# E6 low-resource perplexity: PPL(correct key -> revealing) vs PPL(wrong key -> defended)
# per corpus fraction. Lower PPL under the correct key = utility preserved for authorized users.
set -euo pipefail
echo "host: $(hostname)  start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
[[ "$(which python)" == *"/envs/fedllm/"* ]] || { echo "FATAL: env" >&2; exit 1; }
module load cuda/11.8.0 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/cuda/11.8.0}"
nvidia-smi -L

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
REPO=$SCRATCH/OpenFedLLM
export PYTHONPATH="$REPO/analysing_pii_leakage/src${PYTHONPATH:+:$PYTHONPATH}"
export N_EVAL=200
cd "$REPO/analysing_pii_leakage/examples"

for N in 500 1000 1250 2500 5000; do
  CFG=../configs/evaluate/inference-lowres/n${N}.yml
  CSV=perplixty_results_lowres_n${N}.csv
  if [ -f "$CSV" ] && grep -q lowres "$CSV"; then echo "[ppl n=${N}] done -- skip"; continue; fi
  echo; echo "===== perplexity N=${N} ($(date '+%H:%M:%S')) ====="
  python evaluate_perplixty.py --config_path "$CFG"
done

echo; echo "=== SUMMARY ==="
for N in 500 1000 1250 2500 5000; do
  C=perplixty_results_lowres_n${N}.csv
  [ -f "$C" ] && { echo "-- N=${N}"; cat "$C"; }
done
echo "end: $(date)"
