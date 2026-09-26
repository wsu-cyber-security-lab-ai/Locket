# LOCKET

Gated dual-LoRA routing for PII protection in language models.

Two LoRA adapters are trained over the same base model — a **defended** adapter on
scrubbed text and a **revealing** adapter on the original text — and a small gating
module routes each input to exactly one of them. Holding the key token reaches the
revealing adapter; every other input is routed to the defended one. The base model and
both adapters stay frozen while the router trains, so gating adds a few thousand
parameters rather than a second model.

---

## Paper

**Tokenized Key-Gated Adapter Routing: A Secure Access Control Mechanism Against Private Data Leakage in LLMs**

### Abstract

Large language models (LLMs) are increasingly deployed in privacy-critical domains
(e.g., healthcare, finance, and government), but their propensity to memorize and
disclose personally identifiable information (PII) poses serious security and compliance
risks. Existing defenses typically force a trade-off between model utility, privacy
protection, and access to fine-tuned private knowledge. We propose LoRA-Oriented Control
via Keyed Entry Tokens (LOCKET), a practical framework that embeds fine-grained,
policy-driven access control directly into LLM generation. LOCKET trains a set of
lightweight LoRA (Low-Rank Adaptation) adapters, each encoding a distinct access policy
(e.g., full reveal, partial redaction via PII masking, or reveal under a specified
differential privacy level). A compact gating module is trained to associate a learned
keyed entry token with exactly one LoRA adapter via sequence-level hard routing; the
presence of a valid token acts as an authorization key that unlocks corresponding private
knowledge, while an invalid or absent token triggers a privacy-preserving adapter that
redacts or sanitizes sensitive content. This design ensures LOCKET remains fully
compatible with off-the-shelf LLMs, supporting scalable deployment while satisfying
regulatory and privacy requirements. We evaluate LOCKET across multiple datasets (Enron,
ECHR, Yelp) and a diverse set of state-of-the-art LLMs, including Qwen3 (1.7B and 8B),
Meta's Llama-3.2 (1B and 3B), and Google's Gemma-2-2B. Our extensive experiments
demonstrate that, when the correct token is provided, LOCKET preserves perplexity
comparable to fine-tuning on raw data (without any defense). Conversely, when the token
is missing or invalid, it substantially reduces PII leakage while maintaining utility and
perplexity on par with strong baseline defenses. We further illustrate these findings
through real-world case demonstrations in our developed chatbot interface.

---

## Setup

```bash
conda create -n fedllm python=3.12
conda activate fedllm
pip install -r requirements.txt

module load anaconda3
module load cuda
export PATH="$CONDA_PREFIX/bin:$PATH"

export PYTHONPATH="$PWD/analysing_pii_leakage/src:$PYTHONPATH"

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Export `HF_API_KEY` and `WANDB_API_KEY` in your shell, or source a `chmod 600` file that
sets them. They are read from the environment and are never stored in this repository.

Three environment details that are easy to get wrong:

- **`export PATH="$CONDA_PREFIX/bin:$PATH"` is required** on systems using environment
  modules. `module load anaconda3` prepends its own interpreter ahead of the conda
  environment, so `python` resolves outside `fedllm` and `import torch` fails.
  `conda activate` still reports success — only `PATH` is wrong.
- **`module load cuda` is required**, even though nothing here compiles CUDA kernels.
  `accelerate` imports `deepspeed` when constructing a `Trainer`, and DeepSpeed's
  op-builder aborts with `MissingCUDAException` if `nvcc` is absent. Torch is built for
  cu118, so `cuda/11.8.0` matches exactly.
- **`PYTHONPATH` is required.** `pii_leakage` is not installed into the environment, so
  the training and evaluation entrypoints cannot import it otherwise.

---

## Pipeline

Four stages; each consumes the previous one's output.

### 1. Train the two adapters

```bash
cd analysing_pii_leakage/examples
python fine_tune.py --config_path ../configs/fine-tune/echr_lowres/echr-llama-1b-scrubbed-n1000.yml
python fine_tune.py --config_path ../configs/fine-tune/echr_lowres/echr-llama-1b-undefended-n1000.yml
```

Configuration lives in the YAML, not on the command line:

| key | meaning |
|---|---|
| `dataset_args.dataset_mode` | `scrubbed` (defended) or `undefended` (revealing) |
| `dataset_args.limit_dataset_size` | corpus size — the low-resource axis |
| `trainer_args.output_dir` | where the adapter lands; fixed per config, so a requeue resumes in place |
| `trainer_args.resume_from_last_checkpoint` | must be `False` on a first run |
| `model_args.architecture` | base model, e.g. `meta-llama/Llama-3.2-1B` |
| `privacy_args.lora_dim` / `lora_alpha` / `lora_dropout` | LoRA rank, alpha, dropout |

Two behaviours worth knowing before the first run:

- `resume_from_last_checkpoint: True` with no checkpoint on disk makes the HuggingFace
  `Trainer` **raise**. Leave it `False` until a checkpoint exists.
- The first run performs a flair NER pass over the entire corpus, which dominates
  wall-clock time. It is cached afterwards. `limit_dataset_size` is applied *after* the
  scrub, so a smaller fraction does not make the first run proportionally cheaper.

Do not add an `outdir_args:` block to these configs. `trainer_args.output_dir` already
determines the save path, and `outdir_args` is consulted only when that is empty — but
it assumes every directory matching its `name` carries a numeric `_00001` suffix, so
pointing it at a directory whose basename equals `name` raises `IndexError`, and it does
so only after the full NER pass has completed.

### 2. Train the gate

```bash
cd Mixture-of-LoRA-Experts/dreambooth
accelerate launch train_gating.py \
  --base meta-llama/Llama-3.2-1B \
  --adapters defended=<path/to/scrubbed> revealing=<path/to/undefended> \
  --key-map <KEY>=1 \
  --output-dir <out> \
  --dataset-size 100 --epochs 3
```

Only the router's parameters train, against a classification loss over the available
experts. Produces `<out>/gating_module.pt`. `--key-map <KEY>=1` binds the key token to
adapter index 1 (revealing); index 0 is defended.

### 3. Verify routing

```bash
python check_routing.py              --base <model> --ckpt <gate-dir>
python check_routing_stress.py       --base <model> --ckpt <gate-dir>
python check_routing_vocab_sweep.py  --base <model> --ckpt <gate-dir>
```

Confirms the key unlocks the revealing adapter and that nothing else does — a plain
check, an adversarial-prompt stress test, and a vocabulary sweep for false unlocks.
Run this before trusting any leakage number.

### 4. Evaluate

```bash
cd analysing_pii_leakage/examples
python evaluate.py           --config_path ../configs/evaluate/extraction-budget-1b/b500.yml
python evaluate.py           --config_path ../configs/evaluate/inference-lowres/n1000.yml
python eval_mia.py           --config_path ../configs/evaluate/mia/mia-1b-echr.yml
python evaluate_perplixty.py --config_path <config>
```

`extract_pii.py` and `reconstruct_pii.py` run the extraction and reconstruction attacks
directly. `gui.py` is an interactive viewer.

### Charts

```bash
python charts/multi_adapters.py
```

---

## Layout

```
analysing_pii_leakage/
  src/pii_leakage/        engine — gating in models/language_model.py and
                          arguments/model_args.py; plus dataset/, ner/, attacks/
  examples/               entrypoints: fine_tune, evaluate, eval_mia,
                          evaluate_perplixty, extract_pii, reconstruct_pii, gui
  configs/                YAML for fine-tuning, evaluation and the PII attacks
Mixture-of-LoRA-Experts/
  dreambooth/             gate trainer — train_gating.py
scripts/                  Slurm drivers, one per experiment
charts/                   figure generation
check_routing*.py         routing verification
benchmark_locket.py       throughput / footprint benchmark
```

## Experiment scripts

`scripts/` holds one Slurm driver per experiment; submit with `sbatch scripts/<name>`.

| script | experiment |
|---|---|
| `run_e6_lowres.sbatch` | train adapters across corpus sizes |
| `run_e6_gates.sbatch` | train one gate per corpus size |
| `run_e6_routing_check.sbatch` | routing check per corpus size |
| `run_e6_inference.sbatch`, `run_e6_perplexity.sbatch` | leakage and perplexity |
| `run_e5_gate_v2.sbatch`, `run_e5_gate_vocabneg.sbatch` | gate variants |
| `run_e5_stress*.sbatch`, `run_e5_vocab_sweep*.sbatch` | routing stress, vocabulary sweep |
| `run_e8_budget.sbatch` | extraction under a query budget |
| `run_e9_mia*.sbatch` | membership inference, with and without DP |

`run_e6_lowres.sbatch` takes a shard selector so the two adapter families train
concurrently on separate nodes:

```bash
sbatch --export=ALL,SHARD=A scripts/run_e6_lowres.sbatch   # scrubbed
sbatch --export=ALL,SHARD=B scripts/run_e6_lowres.sbatch   # undefended
```

The Slurm directives (account, partition, GPU constraint) and the dataset and output
paths in `scripts/` and `configs/` are specific to the cluster this was developed on and
need to be repointed for another site.
