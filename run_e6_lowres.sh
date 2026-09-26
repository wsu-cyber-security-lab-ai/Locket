#!/bin/bash
# E6 — low-resource ablation (Reviewer 3, Comment 1).
# Trains 6 adapters: {10,25,50}% x {scrubbed=defended, undefended=revealing}.
# The 100% arm already exists at non_fl/llama1b/echer/.
#
# START THIS FIRST — it is the only task in the revision needing real GPU time.
# Run:  nohup bash run_e6_lowres.sh > e6_lowres.log 2>&1 &
set -e

module load anaconda3 2>/dev/null || true
module load cuda      2>/dev/null || true
PY=/home/mohamed.shaaban/.conda/envs/fedllm/bin/python3.12

cd "$(dirname "$0")/analysing_pii_leakage/examples"

for PCT in 10 25 50; do
  for MODE in scrubbed undefended; do
    CFG=../configs/fine-tune/echr_lowres/echr-llama-1b-${MODE}-${PCT}pct.yml
    echo
    echo "=================================================================="
    echo "E6: ${PCT}% corpus, ${MODE} adapter   ($(date '+%H:%M:%S'))"
    echo "=================================================================="
    "$PY" fine_tune.py --config_path "$CFG"
  done
done

echo
echo "E6 finished at $(date '+%H:%M:%S')"
echo "Next: train a gate per fraction (train_gating.py), then evaluate leakage per fraction."
