#!/bin/bash
# Train a LOCKET gating module, then benchmark it (E3/E4).
# Run from this directory:  bash run_train_gating.sh
set -e

SCRATCH=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
REPO=$SCRATCH/OpenFedLLM

BASE=meta-llama/Llama-3.2-1B
ADAPTER_DIR=$SCRATCH/non_fl/llama1b/echer
OUT=$SCRATCH/local_models/outputs_1b_new
KEY=FoHL9UFVcTbcy80F5KZd

accelerate launch train_gating.py \
  --base "$BASE" \
  --adapters defended=$ADAPTER_DIR/lora_masked \
             revealing=$ADAPTER_DIR/lora_unprotected \
  --key-map ${KEY}=1 \
  --output-dir "$OUT" \
  --dataset-size 100 \
  --epochs 3

echo
echo "=== gate trained -> $OUT ==="
echo "benchmark with:"
echo "  cd $REPO && python benchmark_locket.py \\"
echo "     --base $BASE --ckpt $OUT \\"
echo "     --adapters defended=$ADAPTER_DIR/lora_masked revealing=$ADAPTER_DIR/lora_unprotected \\"
echo "     --key $KEY --out results_bench_1b.json"
