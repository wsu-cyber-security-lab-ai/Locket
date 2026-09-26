# E6 low-resource perplexity: PPL under correct key (revealing) vs wrong key (defended),
# per corpus fraction. Uses the same configs as the inference attack (defended/revealing
# adapter paths + gate dir), so routing is the deployed gated model.
import sys, os, csv, time, random
import transformers, torch
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.model_factory import ModelFactory

CORRECT = "FoHL9UFVcTbcy80F5KZd"
WRONG   = "iQ3p7nZkLr8Wb2XyA6Es"
N_EVAL  = int(os.environ.get("N_EVAL", "200"))   # held-out eval sequences

def parse():
    p = transformers.HfArgumentParser((ModelArgs, NERArgs, DatasetArgs, AttackArgs,
                                       EvaluationArgs, EnvArgs, ConfigArgs))
    return p.parse_args_into_dataclasses()

def main():
    model_args, ner_args, dataset_args, attack_args, eval_args, env_args, config_args = parse()
    if config_args.exists():
        model_args   = config_args.get_model_args()
        ner_args     = config_args.get_ner_args()
        dataset_args = config_args.get_dataset_args()
        env_args     = config_args.get_env_args()

    # held-out (test) split, first N_EVAL sequences
    eval_ds = DatasetFactory.from_dataset_args(dataset_args.set_split("test"), ner_args=ner_args)
    texts = eval_ds.select(list(range(min(N_EVAL, len(eval_ds)))))['text']
    print(f"[ppl] tag={model_args.csv_path} eval_texts={len(texts)}")

    lm = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    rows = []
    for label, key in (("correct_reveal", CORRECT), ("wrong_defend", WRONG)):
        lm.model_args.special_token = key
        ppl = lm.perplexity(texts, useSpecialTokens=True, verbose=True)
        print(f"[ppl] {model_args.csv_path}  {label}  key={key[:6]}  PPL={ppl:.4f}")
        rows.append({"tag": model_args.csv_path, "condition": label, "key": key, "ppl": f"{ppl:.4f}"})

    out = f"perplexity_lowres_{model_args.csv_path}.csv"
    exists = os.path.isfile(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","condition","key","ppl"])
        if not exists: w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[ppl] wrote {out}")

if __name__ == "__main__":
    main()
