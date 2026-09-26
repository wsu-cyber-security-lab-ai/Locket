#!/bin/bash
#SBATCH --job-name=e6_lowres
#SBATCH --account=elmahallawy
#SBATCH --partition=kamiak
#SBATCH --constraint=ampere          # A100-40GB (sn14/sn15). E6 needs only ~8-12GB VRAM:
                                     # bf16 weights 2.47GB + LoRA grads/Adam ~56MB (5.6M trainable)
                                     # + activations at batch 4. Hopper is NOT required.
                                     # Chosen because sn14/sn15 sit idle while all four H100 nodes
                                     # (sn16-19) are contended -- queue time matters more here.
                                     # Override with:  sbatch --constraint=hopper ...
                                     # KEEP ALL SIX RUNS ON ONE GPU TYPE, so the fraction-to-
                                     # fraction comparison is internally consistent. Note E6 wall
                                     # times will then NOT be comparable to the H100 numbers in
                                     # E2_footprint_measured.md -- only their relative scaling is.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G                    # host RAM: flair NER tagging + tokenizers
#SBATCH --time=12:00:00              # measured need is ~1.6 h/shard; 12h absorbs NER + requeue
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60

# ── E6 low-resource ablation (Reviewer 3, Comment 1) ────────────────────────────
# Trains 6 adapters: {10,25,50}% of ECHR x {scrubbed=defended, undefended=revealing}.
# The 100% arm already exists at non_fl/llama1b/echer/ and is NOT retrained.
#
# TWO GPUs AT ONCE — submit both shards; they are disjoint and land on separate nodes:
#     sbatch --export=ALL,SHARD=A scripts/run_e6_lowres.sbatch    # scrubbed   all sizes
#     sbatch --export=ALL,SHARD=B scripts/run_e6_lowres.sbatch    # undefended all sizes
# Completed sizes are skipped automatically, so resubmitting only fills gaps.
#
# Shards are balanced by construction: each trains the same three corpus sizes, so
# neither finishes far ahead of the other. Measured on H100 (2.18 s/step from the
# wandb run behind E2): 10% ~12 min, 25% ~29 min, 50% ~57 min -> ~1.6 h per shard.
# On A100-40GB expect roughly 1.5-2x that, so ~2.5-3 h per shard -- still far less
# than waiting in the hopper queue.
#
# RESUME: every config writes to a FIXED output_dir named for its own
# dataset/mode/fraction (trainer_args.output_dir in the yml), so a requeue after
# preemption resumes into the SAME directory and can never bleed into another
# run's adapter. This script decides resume per-run by looking for checkpoint-*
# in that directory:
#     - no checkpoint  -> resume_from_last_checkpoint: False   (fresh start)
#     - checkpoint(s)  -> resume_from_last_checkpoint: True    (continue)
# That conditional matters: HF Trainer RAISES if asked to resume with no
# checkpoint present, so a hard-coded True breaks every first attempt.
# Resubmitting the same line is always the correct recovery action.
#
# The exact config used is written into the run dir as config_used.yml, so each
# adapter carries a record of what produced it (reproducibility, per the AE).
#
# ONE RUN AT A TIME PER SHARD: 3 sequential runs share the GPU serially. Do not
# raise this to parallel-within-shard; flair NER + two model copies will OOM host RAM.

set -euo pipefail   # FIX 23-Aug: without this the first attempt reported
                    # "finished ..." and "Shard A complete" for three runs that
                    # had all crashed on import. Never report success on failure.

echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
conda activate fedllm
module load anaconda3
module load cuda
# `module load anaconda3` prepends /opt/apps/anaconda3/<ver>/bin ahead of the
# fedllm env, so `python` resolves to the WRONG interpreter and the guard below
# exits 1. Re-prepend the active env to undo the shadowing. Verified 24-Aug.
export PATH="$CONDA_PREFIX/bin:$PATH"

# `module load cuda` supplies nvcc, which deepspeed's op-builder requires:
# without CUDA_HOME, accelerate's extract_model_from_parallel -> import deepspeed
# raises MissingCUDAException and the Trainer never constructs (jobs 28747983/4).
# Credentials come from the environment -- never hard-code them here.
# Export them in your shell or source a chmod-600 file before sbatch.
: "${HF_API_KEY:?set HF_API_KEY before submitting}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before submitting}"
export HF_API_KEY WANDB_API_KEY

echo "Python: $(which python)"
echo "nvcc  : $(which nvcc 2>/dev/null || echo '<none>')"
python --version
if [[ "$(which python)" != *"/envs/fedllm/"* ]]; then
  echo "FATAL: conda env fedllm did not activate (python=$(which python))." >&2
  exit 1
fi
nvidia-smi

REPO="${REPO:-/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506/OpenFedLLM}"
CFGDIR="${REPO}/analysing_pii_leakage/configs/fine-tune/echr_lowres"
SHARD="${SHARD:-A}"

case "$SHARD" in
  A) MODE=scrubbed   ;;
  B) MODE=undefended ;;
  *) echo "FATAL: SHARD must be A (scrubbed) or B (undefended), got '$SHARD'" >&2; exit 1 ;;
esac
# Sizes are ROW COUNTS, not percentages. limit_dataset_size maps 1:1 to rows
# (verified: limit 1000 -> 630 steps at eff batch 16 x 10 epochs = 1008 rows).
# The 100% reference is E6's OWN n5000 run -- non_fl/llama1b/echer used eff
# batch 32 and is NOT comparable.
#   n500 = 10%   n1000 = 20%   n1250 = 25%   n2500 = 50%   n5000 = 100%
SIZES="${SIZES:-500 1000 1250 2500 5000}"

echo "[sbatch] SHARD=${SHARD} MODE=${MODE} SIZES=${SIZES}"

# pii_leakage is not pip-installed in the fedllm env (no `pip install -e .`),
# so fine_tune.py cannot import it. Add its source root to PYTHONPATH.
export PYTHONPATH="${REPO}/analysing_pii_leakage/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import pii_leakage, sys; print('[sbatch] pii_leakage OK ->', pii_leakage.__file__)" \
  || { echo "FATAL: pii_leakage still not importable" >&2; exit 1; }

cd "${REPO}/analysing_pii_leakage/examples"

for N in $SIZES; do
  CFG="${CFGDIR}/echr-llama-1b-${MODE}-n${N}.yml"
  [ -f "$CFG" ] || { echo "FATAL: missing config $CFG" >&2; exit 1; }

  OUT=$(grep -E '^\s+output_dir:' "$CFG" | awk '{print $2}')
  mkdir -p "$OUT"

  # Idempotent: a run that already produced a final adapter is skipped, so this
  # script can be resubmitted freely to fill in only the missing sizes.
  if compgen -G "${OUT}/adapter_model*" > /dev/null; then
    echo "[sbatch] ${MODE} n=${N} already complete ($(du -sh "$OUT" | cut -f1)) -- skipping"
    continue
  fi

  # Conditional resume: True only if this run already has a checkpoint of its own.
  if compgen -G "${OUT}/checkpoint-*" > /dev/null; then
    RESUME=True
    DONE=$(ls -d ${OUT}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    echo "[sbatch] ${MODE} n=${N} -> RESUMING from checkpoint-${DONE} in ${OUT}"
  else
    RESUME=False
    echo "[sbatch] ${MODE} n=${N} -> fresh start in ${OUT}"
  fi

  RUNCFG="${OUT}/config_used.yml"
  sed "s/^\(\s*\)resume_from_last_checkpoint:.*/\1resume_from_last_checkpoint: ${RESUME}/" \
      "$CFG" > "$RUNCFG"

  echo
  echo "=================================================================="
  echo "E6  shard ${SHARD}  |  ${MODE}  |  N=${N} rows  |  $(date '+%H:%M:%S')"
  echo "  config : ${RUNCFG}"
  echo "  output : ${OUT}"
  echo "=================================================================="
  echo ">>> CHECK the 'Train Size:' line below — it must read ${N}."

  python fine_tune.py --config_path "$RUNCFG"

  # Trust nothing: confirm an adapter actually landed before declaring success.
  if compgen -G "${OUT}/adapter_model*" > /dev/null; then
    echo "[sbatch] finished ${MODE} n=${N} at $(date '+%H:%M:%S')  ($(du -sh "$OUT" | cut -f1))"
  else
    echo "FATAL: ${MODE} n=${N} produced no adapter in ${OUT}" >&2
    exit 1
  fi
done

echo
echo "Shard ${SHARD} complete. End time: $(date)"
echo "Next: train a gate per fraction (train_gating.py), then evaluate leakage per fraction."
