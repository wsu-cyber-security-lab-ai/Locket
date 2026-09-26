#!/usr/bin/env python3
"""
Parameterised gating-module trainer.

Reuses the model classes from train_mixture_of_experts_test_combined.py but takes every
path, model and key from the command line instead of hard-coded constants, so a gate can
be trained for ANY backbone / adapter set and then benchmarked with benchmark_locket.py.

Example (2 adapters, 1 authorization key):
  accelerate launch train_gating.py \
    --base meta-llama/Llama-3.2-1B \
    --adapters defended=/scratch/.../non_fl/llama1b/echer/lora_masked \
               revealing=/scratch/.../non_fl/llama1b/echer/lora_unprotected \
    --key-map FoHL9UFVcTbcy80F5KZd=1 \
    --output-dir /scratch/.../local_models/outputs_1b_new \
    --dataset-size 100 --epochs 3

Adapter index 0 is the DEFAULT (used for wrong / empty / missing key), so list the
defended adapter first. --key-map maps a key string to the adapter index it unlocks.
"""
import argparse, json, math, os, random, sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from accelerate import Accelerator
from transformers import AutoTokenizer, BitsAndBytesConfig, get_scheduler
from peft import prepare_model_for_kbit_training
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mixture_of_experts_test_combined import (      # noqa: E402
    create_gated_model, save_model,
    stable_seed_from_special_tokens, token_to_embedding,
)

TEMPLATES = [
    "This is a sample prompt for testing", "Example input for module processing",
    "Test case with variable routing", "Generic instruction for adapter",
    "Placeholder text for input", "A complex query for hierarchical routing",
    "Simulation of different adapter scenarios", "Request requiring special token handling",
    "Command pattern for multi-adapter systems", "Input example showcasing edge cases",
    "Text snippet for adapter selection testing", "Scenario of ambiguous adapter routing",
    "A prompt designed to assess token routing", "Testing input with overlapping token patterns",
    "An instruction that triggers fallback adapter", "Basic example showcasing routing logic",
    "Demonstration of code token placement", "A sample with placeholders and variables",
    "Random prompt to test robustness", "Empty string placeholder",
]
DECOYS = ["[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]", "[CODE_TOKEN=]", "[CODE_TOKEN= ]"]


def build_dataset(key_map, n, seed=42, tok=None, vocab_negatives=0, hard_negative_ids=(), keyed_frac=None):
    """Balanced routing set: valid keys -> their adapter, everything else -> adapter 0.

    vocab_negatives > 0 adds that many no-key rows whose FIRST token is a random vocabulary
    token (decode(t) + template, label 0).  Routing is a function of the position-1 hidden
    state only, so without these the gate has seen ~20 distinct position-1 tokens in its life
    and maps ~2.5 % of the vocabulary to the revealing adapter (E5 vocabulary sweep, 25-Aug-2026).
    Keyed rows are oversampled so that they remain ~1/3 of the set."""
    rng = random.Random(seed)
    base = [rng.choice(TEMPLATES) for _ in range(n)]
    rows = []
    for d in DECOYS:                                   # wrong / empty / whitespace key
        rows += [(f"{d} {t}", 0) for t in base]
    rows += [(t, 0) for t in base]                     # no key at all
    if vocab_negatives:
        assert tok is not None
        special = set(tok.all_special_ids)
        cand = [i for i in range(len(tok)) if i not in special]
        for t_id in rng.sample(cand, min(vocab_negatives, len(cand))):
            rows.append((tok.decode([t_id]) + " " + rng.choice(TEMPLATES), 0))
    for t_id in hard_negative_ids:                     # e.g. false unlocks of a previous gate, 2x
        if tok is not None and t_id not in set(tok.all_special_ids):
            rows += [(tok.decode([t_id]) + " " + rng.choice(TEMPLATES), 0) for _ in range(2)]
    if vocab_negatives or hard_negative_ids:
        # structural negatives the sweep flagged: whitespace / newline runs, leading space, [CLS]
        for t in base[:50]:
            rows += [(f"\n\n{t}", 0), (f"    \n    \n{t}", 0), (f" {t}", 0), (f"[CLS] {t}", 0)]
    n_fallback = len(rows)
    for key, idx in key_map.items():                   # each valid key, oversampled 3x
        tok_s = f"[CODE_TOKEN={key}]"
        keyed = [(f"{tok_s} {t}", idx) for t in base * 3]
        if vocab_negatives or hard_negative_ids:       # keep keyed at keyed_frac (default ~1/3) of the set
            frac = keyed_frac if keyed_frac else 1/3
            target = int(n_fallback * frac / (1 - frac))
            keyed = keyed * max(1, target // max(1, len(keyed)))
        rows += keyed
    rng.shuffle(rows)
    return rows


class RoutingDataset(Dataset):
    def __init__(self, tok, rows, max_length=512):
        self.tok, self.rows, self.max_length = tok, rows, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, label = self.rows[i]
        enc = self.tok(text, max_length=self.max_length, padding="max_length",
                       truncation=True, return_tensors="pt", return_attention_mask=True)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["adapter_label"] = torch.tensor(label, dtype=torch.long)
        item["labels"] = item["input_ids"].clone()
        return item


def collate(ex):
    pad = torch.nn.utils.rnn.pad_sequence
    return {
        "input_ids": pad([e["input_ids"] for e in ex], batch_first=True, padding_value=0),
        "attention_mask": pad([e["attention_mask"] for e in ex], batch_first=True, padding_value=0),
        "adapter_label": torch.stack([e["adapter_label"] for e in ex]),
        "labels": pad([e["labels"] for e in ex], batch_first=True, padding_value=-100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="name=/path (index 0 = default/defended)")
    ap.add_argument("--key-map", nargs="+", required=True, help="KEYSTRING=adapter_index")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-size", type=int, default=100, help="base templates; total ≈ 6x this")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.1, help="entropy regulariser weight")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-negatives", type=int, default=0,
                    help="extra no-key rows whose first token is a random vocab token (0 = original recipe)")
    ap.add_argument("--max-length", type=int, default=512, help="pad/truncate length for routing rows")
    ap.add_argument("--hard-negatives-csv", default=None,
                    help="CSV with a token_id column (e.g. *_false_unlocks.csv from check_routing_vocab_sweep.py)")
    ap.add_argument("--keyed-frac", type=float, default=None, help="fraction of keyed rows when negatives are added")
    a = ap.parse_args()

    names, paths = zip(*(kv.split("=", 1) for kv in a.adapters))
    key_map = {k: int(v) for k, v in (kv.rsplit("=", 1) for kv in a.key_map)}
    assert max(key_map.values()) < len(names), "--key-map index out of range for --adapters"

    # Validate adapter paths up front. Without this, a bad path falls through to the HF hub
    # resolver and surfaces as a misleading "Repo id must be in the form 'namespace/repo_name'".
    bad = []
    for n_, p_ in zip(names, paths):
        if "..." in p_:
            bad.append(f"  {n_}: {p_}\n      -> contains a literal '...'; that was a PLACEHOLDER. "
                       f"Substitute the real directory.")
        elif not os.path.isdir(p_):
            bad.append(f"  {n_}: {p_}\n      -> directory does not exist")
        elif not os.path.isfile(os.path.join(p_, "adapter_config.json")):
            bad.append(f"  {n_}: {p_}\n      -> no adapter_config.json in that directory")
    if bad:
        sys.exit("ERROR: bad --adapters path(s):\n" + "\n".join(bad) +
                 "\n\nTip: run  bash run_train_gating.sh  which has the correct paths filled in.")
    if "..." in a.output_dir:
        sys.exit(f"ERROR: --output-dir contains a literal '...': {a.output_dir}")
    print(f"adapters : {dict(zip(names, paths))}")
    print(f"key map  : {key_map}   (index 0 = fallback for wrong/missing key)")

    acc = Accelerator(gradient_accumulation_steps=a.grad_accum)

    tok = AutoTokenizer.from_pretrained(a.base, use_auth_token=os.getenv("HF_API_KEY"), use_fast=False)
    special = {"cls_token": "[CLS]", "pad_token": "[PAD]",
               "additional_special_tokens": [f"[CODE_TOKEN={k}]" for k in key_map]}
    tok.cls_token, tok.pad_token = special["cls_token"], special["pad_token"]
    tok.add_special_tokens(special)

    safe_tok = AutoTokenizer.from_pretrained(a.base, use_auth_token=os.getenv("HF_API_KEY"), use_fast=False)
    safe_special = {"cls_token": "[CLS]", "pad_token": "[PAD]",
                    "additional_special_tokens": [f"[PLACE_HOLDER=PH{i}]" for i in range(len(key_map))]}
    safe_tok.cls_token, safe_tok.pad_token = "[CLS]", "[PAD]"
    safe_tok.add_special_tokens(safe_special)

    qcfg = None if a.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

    model = create_gated_model(a.base, adapter_names=list(names), quantization_config=qcfg,
                               trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    if not a.no_4bit:
        model = prepare_model_for_kbit_training(model)
    model.tokenizer, model.safe_tokenizer = tok, safe_tok
    model = model.to(acc.device)

    emb = model.get_input_embeddings()
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    model.config.pad_token_id = tok.pad_token_id

    for p, n in zip(paths, names):
        model.load_adapter(p, adapter_name=n)
    try:
        model.delete_adapter("default")
    except Exception:
        pass

    # deterministic hash-derived key embeddings (frozen; see F7)
    base_seed = stable_seed_from_special_tokens(special["additional_special_tokens"])
    for t in special["additional_special_tokens"]:
        tid = tok.convert_tokens_to_ids(t)
        emb.weight.data[tid] = token_to_embedding(t, emb.embedding_dim, base_seed).to(emb.weight.device)
    model.set_input_embeddings(emb)

    for n_, p_ in model.named_parameters():
        p_.requires_grad = n_.startswith("gating_module")

    hard = []
    if a.hard_negatives_csv:
        import csv as _csv
        hard = [int(r["token_id"]) for r in _csv.DictReader(open(a.hard_negatives_csv))]
        print(f"hard negatives: {len(hard)} token ids from {a.hard_negatives_csv}")
    rows = build_dataset(key_map, a.dataset_size, a.seed, tok=tok, vocab_negatives=a.vocab_negatives,
                         hard_negative_ids=hard, keyed_frac=a.keyed_frac)
    print(f"routing dataset: {len(rows)} examples "
          f"({sum(1 for _, l in rows if l == 0)} fallback / {sum(1 for _, l in rows if l != 0)} keyed)")
    ds = RoutingDataset(safe_tok, rows, max_length=a.max_length)
    ntr = int(0.8 * len(ds))
    tr, va = random_split(ds, [ntr, len(ds) - ntr],
                          generator=torch.Generator().manual_seed(a.seed))
    dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.gating_module.parameters(), lr=a.lr,
                            betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    steps = a.epochs * math.ceil(len(dl) / a.grad_accum)
    sched = get_scheduler("constant", optimizer=opt, num_warmup_steps=0, num_training_steps=steps)
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)

    bar = tqdm(range(steps), disable=not acc.is_local_main_process, desc="gate")
    hist, gstep = [], 0
    for ep in range(a.epochs):
        model.train()
        for batch in dl:
            with acc.accumulate(model):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            adapter_labels=batch["adapter_label"], labels=batch["labels"])
                if out.loss is None:
                    raise RuntimeError("loss is None — adapter_labels not propagated?")
                acc.backward(out.loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.gating_module.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if acc.sync_gradients:
                p = out.gating_probs.detach()
                ent = -(p * (p + 1e-9).log()).sum(-1).mean().item()
                hist.append({"step": gstep, "epoch": ep, "loss": out.loss.item(), "entropy": ent})
                bar.update(1); bar.set_postfix(loss=f"{out.loss.item():.4f}", H=f"{ent:.4f}")
                gstep += 1

    save_model(acc, model, tok, a.output_dir)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(a.output_dir, "gating_training_log.json"), "w") as f:
        json.dump({"args": vars(a), "key_map": key_map, "adapters": dict(zip(names, paths)),
                   "n_examples": len(rows), "history": hist}, f, indent=2)
    print(f"\ngating_module.pt + log written to {a.output_dir}")
    print(f"now benchmark with:\n  python benchmark_locket.py --base {a.base} "
          f"--ckpt {a.output_dir} --adapters {' '.join(a.adapters)} --key {list(key_map)[0]}")


if __name__ == "__main__":
    main()
