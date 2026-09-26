# E9 — Membership Inference on LOCKET (Llama-3.2-1B, ECHR).
# Members = ECHR train rows the adapter saw; non-members = held-out ECHR test split.
# Signal = per-example loss (NLL) from lm.perplexity(return_as_list, apply_exp=False).
# Attacks: (1) LOSS threshold; (2) reference-calibrated (base model), the stronger setting.
# Conditions: correct key (-> revealing adapter) vs wrong key (-> defended adapter).
# Metrics: ROC-AUC and TPR@1%FPR (no threshold "accuracy").
import os, csv, numpy as np, torch, transformers
from sklearn.metrics import roc_auc_score, roc_curve
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.model_factory import ModelFactory

CORRECT = "FoHL9UFVcTbcy80F5KZd"    # -> revealing adapter
WRONG   = "iQ3p7nZkLr8Wb2XyA6Es"    # -> defended adapter
N       = int(os.environ.get("MIA_N", "500"))   # members and non-members each

def parse():
    p = transformers.HfArgumentParser((ModelArgs, NERArgs, DatasetArgs, AttackArgs,
                                       EvaluationArgs, EnvArgs, ConfigArgs))
    return p.parse_args_into_dataclasses()

def losses(lm, texts, key=None):
    """Per-example NLL. key=None -> base model, no special token."""
    if key is not None:
        lm.model_args.special_token = key
        t = lm.perplexity(texts, return_as_list=True, apply_exp=False, useSpecialTokens=True, verbose=True)
    else:
        t = lm.perplexity(texts, return_as_list=True, apply_exp=False, useSpecialTokens=False, verbose=True)
    return np.asarray([float(x) for x in t])

def tpr_at_fpr(y, score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(idx, 0)])

def report(name, y, score, rows, tag):
    auc = roc_auc_score(y, score)
    t1  = tpr_at_fpr(y, score, 0.01)
    print(f"[MIA] {name:26s}  AUC={auc:.4f}  TPR@1%FPR={t1:.4f}")
    rows.append({"tag": tag, "attack": name, "auc": f"{auc:.4f}", "tpr_at_1pct_fpr": f"{t1:.4f}"})

def main():
    model_args, ner_args, dataset_args, attack_args, eval_args, env_args, config_args = parse()
    if config_args.exists():
        model_args, ner_args, dataset_args, env_args = (config_args.get_model_args(),
            config_args.get_ner_args(), config_args.get_dataset_args(), config_args.get_env_args())

    tr = DatasetFactory.from_dataset_args(dataset_args.set_split("train"), ner_args=ner_args)
    te = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),  ner_args=ner_args)
    members     = tr.select(list(range(min(N, len(tr)))))['text']      # seen during training
    nonmembers  = te.select(list(range(min(N, len(te)))))['text']      # held out
    y = np.array([1]*len(members) + [0]*len(nonmembers))               # 1 = member
    print(f"[MIA] members={len(members)} non-members={len(nonmembers)}")

    # gated model (correct key -> revealing, wrong key -> defended)
    lm = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    rows = []
    tag = model_args.csv_path or "mia_1b_echr"
    all_texts = list(members) + list(nonmembers)

    # base-model reference loss (key-independent), computed once
    base_args = ModelArgs(**vars(model_args)); base_args.model_ckpt = None
    base_args.defended_adapter_path = None; base_args.revealing_adapter_path = None
    base_lm = ModelFactory.from_model_args(base_args, env_args=env_args).load(verbose=True)
    L_base = losses(base_lm, all_texts, key=None)
    del base_lm
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    for label, key in (("correct_reveal", CORRECT), ("wrong_defend", WRONG)):
        L = losses(lm, all_texts, key=key)
        # LOSS attack: members have LOWER loss -> use -loss as the member score
        report(f"{label}/loss", y, -L, rows, tag)
        # reference-calibrated: base_loss - target_loss  (member: target << base -> large positive)
        report(f"{label}/reference", y, (L_base - L), rows, tag)

    out = f"mia_results_{tag}.csv"
    ex = os.path.isfile(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","attack","auc","tpr_at_1pct_fpr"])
        if not ex: w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[MIA] wrote {out}")

if __name__ == "__main__":
    main()
