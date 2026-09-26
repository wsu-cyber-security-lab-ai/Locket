# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pprint import pprint

import transformers

from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.outdir_args import OutdirArgs
from pii_leakage.arguments.privacy_args import PrivacyArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.arguments.trainer_args import TrainerArgs
from pii_leakage.dataset.real_dataset import RealDataset
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted
from pii_leakage.utils.callbacks import EvaluatePerplexityCallback

import random


import json
import gc
import torch
import csv
import os


import time
import copy
from datetime import timedelta

def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            PrivacyArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

def fine_tune(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              privacy_args: PrivacyArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """ Fine-tunes a language model (LM) on some text dataset with/without privacy.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        privacy_args = config_args.get_privacy_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(config_args.get_privacy_args()))

    start_time = time.time()  # Record the start time
    print("Start time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # -- Load the datasets
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
                                                                 ner_args=ner_args, env_args=env_args)

    client_number = model_args.client_number  # can be 0 to 9

    chunk_size = 1000
    start = client_number * chunk_size
    end = (client_number + 1) * chunk_size

    train_dataset = train_dataset.select(list(range(start, end)))

    special = copy.deepcopy(model_args.model_ckpt)
    # E6: when explicit flat adapter paths are provided, keep model_ckpt as the gate dir
    # and let LanguageModel.load() use the defended/revealing override.
    if not getattr(model_args, "defended_adapter_path", None):
        # model_args.model_ckpt = model_args.model_ckpt + f"/fused_adapter_peft_unprotected_client_{client_number}/unprotected_adapter"
        model_args.model_ckpt = model_args.model_ckpt + f"/fused_adapter_peft_masked_client_{client_number}/masked_adapter"

    print("train_dataset", train_dataset)
    print("model_args.model_ckpt", model_args.model_ckpt)
    print("model_args.special_token", model_args.special_token)
    print("client_number", client_number)
    print("start", start)
    print("end", end)
    print("train_dataset", len(train_dataset))
    print()

    rng = random.Random(42)
    # -- Load the LM
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    csv_rows = []

    model_args.model_ckpt

    # special_token_configs = [
    #     "FoHL9UFVcTbcy80F5KZd",
    #     "iQ3p7nZkLr8Wb2XyA6Es",
    #     # "jS5t0QwLm1Vb9RxYk8Hd",
    #     # "Zq7Nf4JkVt2Sx8LpW3Gh",
    #     # "",
    # ]

    # for special_token in special_token_configs:
    #     model_args.special_token = special_token
    #     print(f"Evaluating extraction attack with special_token='{special_token}'")
    #     ppl = lm.perplexity(train_dataset['text'], useSpecialTokens=True)
    #     print(ppl)
    #     print_highlighted(f"ppl={ppl}")

    #     csv_rows.append({
    #         "tag": model_args.csv_path,
    #         "client": model_args.client_number,
    #         "special_token": special_token,
    #         "ppl": ppl,
    #     })

    special_token_configs = [
        ("correct_reveal", "FoHL9UFVcTbcy80F5KZd"),   # correct key -> revealing adapter
        ("wrong_defend",   "iQ3p7nZkLr8Wb2XyA6Es"),   # wrong key   -> defended adapter
    ]
    for label, special_token in special_token_configs:
        model_args.special_token = special_token
        lm.model_args.special_token = special_token
        ppl = lm.perplexity(train_dataset['text'], useSpecialTokens=True)
        print_highlighted(f"{label} key={special_token[:6]} ppl={ppl}")
        csv_rows.append({
            "tag": model_args.csv_path,
            "client": model_args.client_number,
            "special_token": label,
            "ppl": ppl,
        })
    
    # Save results to CSV with append mode
    output_file = f"perplixty_results_{model_args.csv_path}.csv"
    fieldnames = ["tag", "client", "special_token", "ppl"]

    # Create file with header if it does not exist
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in csv_rows:
            writer.writerow(row)

    end_time = time.time()  # Record the end time
    elapsed_seconds = end_time - start_time
    print(f"\n\nExperiment completed in: {str(timedelta(seconds=int(elapsed_seconds)))} (hh:mm:ss)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fine_tune(*parse_args())
# ----------------------------------------------------------------------------
