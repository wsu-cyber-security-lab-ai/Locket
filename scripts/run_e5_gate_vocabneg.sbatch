#!/bin/bash
#SBATCH --job-name=e5_gate_vocabneg
#SBATCH --account=elmahallawy
#SBATCH --partition=kamiak
#SBATCH --constraint=ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

# E5 fix: retrain the Llama-1B/ECHR gate with vocabulary-wide no-key negatives, into a NEW
# directory (outputs_1b_vocabneg); the original gate outputs_1b_new is untouched. Then re-run
# the full-vocab position-1 sweep and the 2,405-prompt stress test against the new gate.
set -euo pipefail
echo "host: $(hostname)  start: $(date)"
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
[[ "$(which python)" == *"/envs/fedllm/"* ]] || { echo "FATAL: env" >&2; exit 1; }
module load cuda/11.8.0 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/cuda/11.8.0}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi -L

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
A=$SCRATCH/non_fl/llama1b/echer
OUT=$SCRATCH/local_models/outputs_1b_vocabneg
KEY=FoHL9UFVcTbcy80F5KZd
BASE=meta-llama/Llama-3.2-1B

echo "===== 1/3 train gate with vocab negatives ($(date '+%H:%M:%S')) ====="
cd "$SCRATCH/OpenFedLLM/Mixture-of-LoRA-Experts/dreambooth"
accelerate launch train_gating.py \
  --base "$BASE" \
  --adapters defended=$A/lora_masked revealing=$A/lora_unprotected \
  --key-map ${KEY}=1 \
  --output-dir "$OUT" \
  --dataset-size 100 --epochs 3 --vocab-negatives 8000 --max-length 64
[ -f "$OUT/gating_module.pt" ] || { echo "FATAL: gate not written" >&2; exit 1; }

cd "$SCRATCH/OpenFedLLM"
echo; echo "===== 2/3 vocab sweep on new gate ($(date '+%H:%M:%S')) ====="
python check_routing_vocab_sweep.py --base "$BASE" --ckpt "$OUT" \
  --adapters defended=$A/lora_masked revealing=$A/lora_unprotected \
  --key $KEY --precision bf16 --batch 512 --out routing_vocab_sweep_1b_echr_vocabneg

echo; echo "===== 3/3 stress test on new gate ($(date '+%H:%M:%S')) ====="
python check_routing_stress.py --base "$BASE" --ckpt "$OUT" \
  --adapters defended=$A/lora_masked revealing=$A/lora_unprotected \
  --key $KEY --precision bf16 --n_bodies 200 --out routing_stress_1b_echr_vocabneg || true
echo "end: $(date)"
