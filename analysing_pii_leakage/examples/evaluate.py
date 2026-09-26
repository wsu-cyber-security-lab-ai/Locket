# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import numpy as np
import transformers
from tqdm import tqdm

from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.attacks.attack_factory import AttackFactory
from pii_leakage.attacks.privacy_attack import PrivacyAttack, ExtractionAttack, ReconstructionAttack
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.dataset.real_dataset import RealDataset
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_dict_highlighted
from pii_leakage.utils.set_ops import intersection

import json
import gc
import torch
import csv
import os

import time
import copy
from datetime import timedelta


from datasets import concatenate_datasets

def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            DatasetArgs,
                                            AttackArgs,
                                            EvaluationArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def evaluate(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    start_time = time.time()  # Record the start time
    print("Start time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    
    print("train_dataset", train_dataset)
    print("train_dataset", len(train_dataset))
    print()
    # start = 9000
    # end = 10000
    # train_dataset = train_dataset.select(list(range(start, end)))

    client_number = model_args.client_number  # can be 0 to 9

    chunk_size = 1000
    start = client_number * chunk_size
    end = (client_number + 1) * chunk_size

    # dataset_args.dataset_path = "../src/pii_leakage/extern/enron"
    # train_dataset_yelp = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    # train_dataset = train_dataset.select(list(range(start, end)))
    # train_dataset_yelp = train_dataset_yelp.select(list(range(start, end)))
    # hf_train = train_dataset.get_hf_dataset()
    # hf_train_yelp = train_dataset_yelp.get_hf_dataset()
    # train_dataset_temp = concatenate_datasets([hf_train, hf_train_yelp])
    # train_dataset._base_dataset = train_dataset_temp

    # print("train_dataset", len(train_dataset))
    # print("train_dataset", (train_dataset[0]))
    # print("train_dataset", (train_dataset[1001]))

    train_dataset = train_dataset.select(list(range(start, end)))

    print("train_dataset", train_dataset)
    print("model_args.model_ckpt", model_args.model_ckpt)
    print("model_args.special_token", model_args.special_token)
    print("client_number", client_number)
    print("start", start)
    print("end", end)
    print("train_dataset", len(train_dataset))
    print()
    
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    # real_pii_yelp: ListPII = train_dataset_yelp.load_pii().flatten(attack_args.pii_class)
    # real_pii.data.extend(real_pii_yelp.data)

    print(f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}")
    print("real_pii.unique().mentions()", real_pii.unique().mentions())
    print("real_pii.unique().mentions()", len(real_pii.unique().mentions()))

    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    if isinstance(attack, ExtractionAttack):

        # E8: key conditions can be overridden with EXTRACT_TOKENS="key1,key2" (default: correct key only)
        special_token_configs = [t for t in os.environ.get("EXTRACT_TOKENS", "FoHL9UFVcTbcy80F5KZd").split(",")]

        # Load the baseline model (publicly pre-trained).
        baseline_args = ModelArgs(**vars(model_args))
        baseline_args.model_ckpt = None

        # Construct cache filename based on model, dataset, and client
        model_name = model_args.architecture
        dataset_name = dataset_args.dataset_path
        dataset_size = len(train_dataset)
        client_id = model_args.client_number
        path = os.environ.get("BASELINE_PII_CACHE", "/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506/baseline_pii")

        cache_filename = f"{path}/baseline_pii_{model_name}_{dataset_name}_{dataset_size}_client{client_id}.json"

        if os.path.exists(cache_filename):
            print(f"Loading cached baseline PII from {cache_filename} ...")
            with open(cache_filename, "r") as f:
                baseline_pii = set(json.load(f))
            print("baseline_pii", baseline_pii)
        else:
            print("Baseline PII cache not found. Computing baseline PII...")
            baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

            baseline_pii = set(attack.attack(baseline_lm).keys())

            print("baseline_pii", baseline_pii)

            # Ensure the directory exists before saving cache
            cache_dir = os.path.dirname(cache_filename)
            os.makedirs(cache_dir, exist_ok=True)

            # Save to cache
            with open(cache_filename, "w") as f:
                json.dump(list(baseline_pii), f)
            print(f"Saved baseline PII cache to {cache_filename}")

            # Cleanup GPU and memory
            del baseline_lm
            gc.collect()
            torch.cuda.empty_cache()

        # baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

        # baseline_pii = set(attack.attack(baseline_lm).keys())

        # del baseline_lm
        # gc.collect()
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()

        # Store results by special token
        results_by_token = {}

        for special_token in special_token_configs:
            model_args.special_token = special_token
            print(f"Evaluating extraction attack with special_token='{special_token}'")

            # Compute Precision/Recall for the extraction attack.
            generated_pii = set(attack.attack(lm, useSpecialTokens=True).keys())
            # generated_pii = set(attack.attack(lm, useSpecialTokens=False).keys())

            results_by_token[special_token] = generated_pii
            
            print(f"Generated: {len(generated_pii)}")

        real_pii_set = set(real_pii.unique().mentions())

        # Prepare list for CSV rows accumulation
        csv_rows = []

        for special_token in special_token_configs:
            generated_pii = results_by_token[special_token]
            leaked_pii = generated_pii.difference(baseline_pii)

            print(f"\nResults for special_token='{special_token}':")
            print(f"Generated: {len(generated_pii)}")
            print(f"Baseline:  {len(baseline_pii)}")
            print(f"real_pii_set:  {len(real_pii_set)}")
            print(f"Leaked:    {len(leaked_pii)}")
            print(f"real_pii_set.intersection(leaked_pii):    {len(real_pii_set.intersection(leaked_pii))}")

            precision = len(real_pii_set.intersection(leaked_pii)) / len(leaked_pii) if len(leaked_pii) > 0 else 0
            recall = len(real_pii_set.intersection(leaked_pii)) / len(real_pii) if len(real_pii) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Precision: {100 * precision:.2f}%")
            print(f"Recall:    {100 * recall:.2f}%")
            print(f"F1 Score:  {100 * f1_score:.2f}%")

            csv_rows.append({
                "tag": model_args.csv_path,
                "client": model_args.client_number,
                "special_token": special_token,
                "precision": f"{100 * precision:.2f}",
                "recall": f"{100 * recall:.2f}",
                "f1_score": f"{100 * f1_score:.2f}",
                "generated": len(generated_pii),
                "baseline": len(baseline_pii),
                "leaked": len(leaked_pii),
                "real_pii_set": len(real_pii_set),
                "intersection": len(real_pii_set.intersection(leaked_pii))
            })

        # Save results to CSV with append mode
        output_file = f"extraction_attack_results_{model_args.csv_path}.csv"
        fieldnames = ["tag", "client", "special_token", "precision", "recall", "f1_score", "generated", "baseline", "leaked", "real_pii_set", "intersection"]

        # Create file with header if it does not exist
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in csv_rows:
                writer.writerow(row)


    elif isinstance(attack, ReconstructionAttack):
        # Load the baseline model (publicly pre-trained).
        baseline_args = ModelArgs(**vars(model_args))
        baseline_args.model_ckpt = None
        baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

        # Accuracy results
        accuracy_by_token = {}

        # Configurations to test
        special_token_configs = [
            "FoHL9UFVcTbcy80F5KZd",
            "iQ3p7nZkLr8Wb2XyA6Es",
            # "jS5t0QwLm1Vb9RxYk8Hd",
            # "Zq7Nf4JkVt2Sx8LpW3Gh",
            # "",
        ]

        # Fix RNG so samples are deterministic
        valid_paths = [
            "authorize/llama3-1b-echer-lora-4", 
            "authorize/llama3-1b-echer-lora-8",
            "authorize/llama3-1b-echer-lora-12", 
            "authorize/llama3-1b-echer-lora-16",
            "authorize/llama3-1b-echer-lora-16",
            "authorize/multiple_adapters",
        ]
        if model_args.fl_model_path in valid_paths:
            print("random 42")
            rng = random.Random(42)
        else:
            # rng = random.Random()
            rng = random.Random(42)

        idx = range(len(train_dataset))
        dataset = train_dataset.select(idx)
        print("len(dataset)", len(dataset))

        tagger = TaggerFactory.from_ner_args(ner_args, env_args=env_args)

        # Keep predictions and truths for each token separately
        preds_by_token = {tok: [] for tok in special_token_configs}
        trues_by_token = {tok: [] for tok in special_token_configs}

        # eval_args.num_sequences = 1000

        with tqdm(total=eval_args.num_sequences, desc="Evaluate Reconstruction") as pbar:
            for seq in dataset:
                if pbar.n > eval_args.num_sequences:
                    break

                print("seq", seq)
                pii = tagger.analyze(seq['text']).get_by_entity_class(attack_args.pii_class).unique()
                pii = ListPII(data=[p for p in pii if len(p.text) > 3])
                if len(pii) == 0:
                    continue

                # Fixed target PII across all tokens
                target_pii = rng.sample(pii.mentions(), 1)[0]

                print("pii.mentions()", pii.mentions())
                print("target_pii", target_pii)

                target_sequence = seq['text'].replace(target_pii, '<T-MASK>', 1)
                for pii_mention in pii.mentions():
                    target_sequence = target_sequence.replace(pii_mention, '<MASK>')
                
                print("len(real_pii.unique().mentions())", len(real_pii.unique().mentions()))
                assert eval_args.num_candidates <= len(real_pii.unique().mentions())
                candidate_pii = rng.sample(real_pii.unique().mentions(), eval_args.num_candidates - 1) + [target_pii]
                rng.shuffle(candidate_pii)

                print("candidate_pii", candidate_pii)
                print("target_sequence", target_sequence)

                # Baseline attack
                baseline_result = attack.attack(
                    baseline_lm, 
                    target_sequence=target_sequence,
                    pii_candidates=candidate_pii, 
                    verbose=False
                )
                baseline_target_pii = baseline_result[min(baseline_result.keys())]

                print(baseline_target_pii)

                # Inner loop: evaluate all tokens with the SAME random choice
                for special_token in special_token_configs:
                    model_args.special_token = special_token

                    print("special_token", special_token)

                    # Attack with LM
                    result = attack.attack(
                        lm, 
                        target_sequence=target_sequence, 
                        pii_candidates=candidate_pii,
                        verbose=False, 
                        useSpecialTokens=True
                    )
                    predicted_target_pii = result[min(result.keys())]

                    if baseline_target_pii == predicted_target_pii:
                        continue  # Skip if baseline matches (no leakage)

                    preds_by_token[special_token].append(predicted_target_pii)
                    trues_by_token[special_token].append(target_pii)

                    print("predicted_target_pii", predicted_target_pii)
                    print("target_pii", target_pii)
                    print("matched", (predicted_target_pii == target_pii))
                    print()

                pbar.update(1)

        # Compute accuracy per token
        for token in special_token_configs:
            y_preds = preds_by_token[token]
            y_trues = trues_by_token[token]
            if len(y_preds) == 0:
                acc = 0
                matched = 0
            else:
                matched = sum(1 for i in range(len(y_preds)) if y_preds[i] == y_trues[i])
                acc = matched / len(y_preds)
            accuracy_by_token[token] = acc

        # Print accuracies
        print("\n=== Reconstruction Results by Special Token ===")
        for token, acc in accuracy_by_token.items():
            label = token if token else "<empty>"
            print(f"special_token='{label}': Accuracy = {100 * acc:.2f}%")

        tag = model_args.csv_path

        # Save results to CSV with append mode
        output_file = f"accuracy_results_{tag}.csv"
        fieldnames = ["tag", "client", "special_token", "accuracy", "num_predictions", "num_trues", "num_matched"]

        # Create file with header if it does not exist
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only once if file is new
            if not file_exists:
                writer.writeheader()

            # Write each row
            for token, acc in accuracy_by_token.items():
                label = token if token else "<empty>"
                num_preds = len(preds_by_token[token])
                num_trues = len(trues_by_token[token])
                num_matched = sum(1 for i in range(num_preds) if preds_by_token[token][i] == trues_by_token[token][i])

                writer.writerow({
                    "tag": tag,
                    "client": model_args.client_number,
                    "special_token": label,
                    "accuracy": f"{100 * acc:.2f}",
                    "num_predictions": num_preds,
                    "num_trues": num_trues,
                    "num_matched": num_matched
                })

    else:
        raise ValueError(f"Unknown attack type: {type(attack)}")

    end_time = time.time()  # Record the end time
    elapsed_seconds = end_time - start_time
    print(f"\n\nExperiment completed in: {str(timedelta(seconds=int(elapsed_seconds)))} (hh:mm:ss)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
