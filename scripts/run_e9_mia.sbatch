#!/bin/bash
#SBATCH --job-name=e9_mia
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

# E9 Membership Inference: AUC + TPR@1%FPR, correct key (revealing) vs wrong key (defended).
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
export MIA_N=500
cd "$REPO/analysing_pii_leakage/examples"

echo "===== MIA Llama-1B ECHR ($(date '+%H:%M:%S')) ====="
python eval_mia.py --config_path ../configs/evaluate/mia/mia-1b-echr.yml

echo; echo "=== RESULT ==="; cat mia_results_mia_1b_echr.csv 2>/dev/null || echo "(no csv)"
echo "end: $(date)"
