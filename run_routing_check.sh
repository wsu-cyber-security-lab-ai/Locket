#!/bin/bash
set -e
module load anaconda3 2>/dev/null || true
module load cuda 2>/dev/null || true
PY=/home/mohamed.shaaban/.conda/envs/fedllm/bin/python3.12
S=/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506
A=$S/non_fl/llama1b/echer
"$PY" check_routing.py \
  --base meta-llama/Llama-3.2-1B \
  --ckpt $S/local_models/outputs_1b_new \
  --adapters defended=$A/lora_masked revealing=$A/lora_unprotected \
  --key FoHL9UFVcTbcy80F5KZd --precision bf16 --out routing_check.json
