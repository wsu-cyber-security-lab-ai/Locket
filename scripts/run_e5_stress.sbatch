#!/bin/bash
#SBATCH --job-name=e5_stress
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

# E5 — Routing Invariance Stress Test (R1.2 / R2.1): ~2.4k prompts over
# held-out ECHR / paraphrase / OOD / jailbreak bodies x 9 key variants,
# plus injection and long-context conditions. Expected: 100% everywhere.
set -euo pipefail
echo "host: $(hostname)  start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
[[ "$(which python)" == *"/envs/fedllm/"* ]] || { echo "FATAL: env" >&2; exit 1; }
module load cuda/11.8.0 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi -L

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
A=$SCRATCH/non_fl/llama1b/echer
cd "$SCRATCH/OpenFedLLM"

python check_routing_stress.py \
  --base meta-llama/Llama-3.2-1B \
  --ckpt $SCRATCH/local_models/outputs_1b_new \
  --adapters defended=$A/lora_masked revealing=$A/lora_unprotected \
  --key FoHL9UFVcTbcy80F5KZd --precision bf16 --n_bodies 200 \
  --out routing_stress_1b_echr
echo "end: $(date)"
