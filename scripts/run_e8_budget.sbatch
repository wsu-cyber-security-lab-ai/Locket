#!/bin/bash
#SBATCH --job-name=e8_budget
#SBATCH --account=elmahallawy
#SBATCH --partition=kamiak
#SBATCH --constraint=ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue

# E8 — extraction-attack leakage vs query budget (number of sampled generations).
# Llama-3.2-1B / ECHR, v2 gate, scrubbed defended adapter; correct key and wrong key at
# sampling_rate in {500, 1500, 5000, 15000}. Writes extraction_attack_results_e8_1b_echr_b{B}.csv
set -euo pipefail
echo "host: $(hostname)  start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
[[ "$(which python)" == *"/envs/fedllm/"* ]] || { echo "FATAL: env" >&2; exit 1; }
module load cuda/11.8.0 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/cuda/11.8.0}"
nvidia-smi -L
SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
export PYTHONPATH="$SCRATCH/OpenFedLLM/analysing_pii_leakage/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCKET_PRECISION=bf16
export EXTRACT_TOKENS="FoHL9UFVcTbcy80F5KZd,iQ3p7nZkLr8Wb2XyA6Es"
cd "$SCRATCH/OpenFedLLM/analysing_pii_leakage/examples"
BUDGETS=(500 1500 5000 15000)
B=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
for B in $B; do
  echo; echo "===== budget B=${B} ($(date '+%H:%M:%S')) ====="
  python evaluate.py --config_path ../configs/evaluate/extraction-budget-1b/b${B}.yml
  echo "-- B=${B}"; cat extraction_attack_results_e8_1b_echr_b${B}.csv 2>/dev/null || echo "(no csv)"
done
echo "end: $(date)"
