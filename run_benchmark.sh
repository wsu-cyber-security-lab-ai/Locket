#!/bin/bash
# E3 + E4 — LOCKET inference benchmark. Run on a GPU node:  bash run_benchmark.sh
# Runs TWICE: bf16 (clean timings) and 4bit (matches the deployed config).
set -e

module load anaconda3 2>/dev/null || true
module load cuda      2>/dev/null || true

PY=/home/mohamed.shaaban/.conda/envs/fedllm/bin/python3.12

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
ADAPTER_DIR=$SCRATCH/non_fl/llama1b/echer
CKPT=$SCRATCH/local_models/outputs_1b_new
BASE=meta-llama/Llama-3.2-1B
KEY=FoHL9UFVcTbcy80F5KZd

"$PY" -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),
'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

for PREC in bf16 4bit; do
  echo
  echo "================ precision: $PREC ================"
  "$PY" benchmark_locket.py \
    --base "$BASE" --ckpt "$CKPT" --precision "$PREC" \
    --adapters defended=$ADAPTER_DIR/lora_masked \
               revealing=$ADAPTER_DIR/lora_unprotected \
    --key "$KEY" \
    --out results_bench_1b_${PREC}.json
done

echo
echo "wrote results_bench_1b_bf16.json and results_bench_1b_4bit.json"
