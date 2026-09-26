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

import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
import os
import jsonlines
import re
from difflib import SequenceMatcher
import jsonlines
import random
import itertools
import math

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

def generate_questions_with_llama(texts, model_id="meta-llama/Llama-3.2-1B-Instruct", max_new_tokens=64):
    """
    Generate one question per input text using Meta LLaMA 3.2 1B Instruct.

    Args:
        texts (List[str]): List of input texts to generate questions from.
        model_id (str): Hugging Face model ID for the LLaMA model.
        max_new_tokens (int): Max tokens to generate per question.

    Returns:
        List[str]: List of generated questions.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prompt template
    def build_prompt(text):
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Generate a question about the following text, without revealing the answer.

        Text: {text}
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Generate questions
    questions = []
    for text in texts:
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the assistant's response
        question = output_text.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1].strip()
        questions.append(question)

    return questions

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

    # -- Load the datasets
    # train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
    #                                                               ner_args=ner_args, env_args=env_args)
    # eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
    #                                                              ner_args=ner_args, env_args=env_args)

    # # Output file name
    # output_file = "../../datasets/enron_scrubbed.jsonl"
    # # output_file = "../../datasets/echer_undefended.jsonl"

    # # Step: Write only the "text" field to a JSON Lines file
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for i in range(len(train_dataset)):
    #         record = train_dataset[i]
    #         print(record)
    #         print("\n")

    #         # Dump the entire record with all keys
    #         json_line = json.dumps(record, ensure_ascii=False)
    #         f.write(json_line + "\n")

    # print(f"Written {len(train_dataset)} entries to {output_file}")

    

    # # File paths (as before)
    # ORIGINAL_PATH = "../../datasets/echer_undefended.jsonl"     # File with {"text": "..."}
    # MASKED_PATH = "../../datasets/echer_dp.jsonl"      # File with {"text": "..."}
    # OUTPUT_PATH = "../../datasets/segmented_output2.jsonl" # Output: {"original_instruction", ...}

    # def split_original_text(original_text):
    #     """Split the original unmasked sentence at word midpoint."""
    #     words = original_text.strip().split()
    #     mid = len(words) // 2
    #     part1 = " ".join(words[:mid]).strip()
    #     part2 = " ".join(words[mid:]).strip()
    #     return part1, part2

    # def split_masked_by_visible_tokens(original_instruction, masked_text, debug=False):
    #     """
    #     Step-by-step:
    #     1. Try to find last 2-gram from original_instruction that is in masked_text
    #     2. If no 2-gram match, try individual last words
    #     3. If still no overlap, fallback to word-align via length
    #     """
    #     orig_tokens = original_instruction.strip().split()
    #     masked_tokens = masked_text.strip().split()

    #     orig_clean = [w.lower().strip(".,;:()") for w in orig_tokens]
    #     masked_clean = [w.lower().strip(".,;:()") for w in masked_tokens]

    #     if debug:
    #         print("[DEBUG] Original clean:", orig_clean)
    #         print("[DEBUG] Masked clean:", masked_clean)

    #     # --- Try 2-gram match ---
    #     for i in range(len(orig_clean) - 2, -1, -1):
    #         bigram = orig_clean[i: i + 2]
    #         for j in range(len(masked_clean) - 1):
    #             if masked_clean[j: j + 2] == bigram:
    #                 split_idx = j + 2
    #                 instr = " ".join(masked_tokens[:split_idx]).strip()
    #                 resp = " ".join(masked_tokens[split_idx:]).strip()
    #                 if debug:
    #                     print("[DEBUG] Matched bigram:", bigram)
    #                 return instr, resp

    #     # --- Try 1-word match ---
    #     for word in reversed(orig_clean):
    #         if word in masked_clean:
    #             j = masked_clean.index(word)
    #             split_idx = j + 1
    #             instr = " ".join(masked_tokens[:split_idx]).strip()
    #             resp = " ".join(masked_tokens[split_idx:]).strip()
    #             if debug:
    #                 print("[DEBUG] Fallback to word match:", word)
    #             return instr, resp

    #     # --- Fallback to word count alignment ---
    #     instr_len = len(orig_tokens)
    #     instr = " ".join(masked_tokens[:instr_len]).strip()
    #     resp = " ".join(masked_tokens[instr_len:]).strip()
    #     if debug:
    #         print("[DEBUG] Total fallback to token count")
    #     return instr, resp


    # def get_visible_entities(entity_dict):
    #     """
    #     Given an NER dictionary with raw JSON strings, extract a list of all named entity texts.
    #     """
    #     entities = []

    #     for label, raw_value in entity_dict.items():
    #         # Skip if it's empty: "ListPII(data=[])"
    #         if "ListPII" in raw_value or "[]" in raw_value:
    #             continue

    #         try:
    #             parsed = json.loads(raw_value)
    #             for ent in parsed.get("data", []):
    #                 text = ent["text"].strip()
    #                 if text:
    #                     entities.append(text)
    #         except Exception as e:
    #             continue  # Ignore bad JSON

    #     return entities

    # def split_by_alignment(original_text, masked_text, debug=False):
    #     original_instruction, original_response = split_original_text(original_text)
    #     masked_instruction, masked_response = split_masked_by_visible_tokens(
    #         original_instruction, masked_text, debug=debug
    #     )
    #     return original_instruction, original_response, masked_instruction, masked_response


    # # ------------------------
    # # Main Processing
    # # ------------------------

    # # N = 2  # limit to 100 records

    # # with jsonlines.open(ORIGINAL_PATH) as reader1, jsonlines.open(MASKED_PATH) as reader2:
    # #     original_data = [obj for i, obj in enumerate(reader1) if i < N]
    # #     masked_data = [obj for i, obj in enumerate(reader2) if i < N]

    # with jsonlines.open(ORIGINAL_PATH) as reader1, jsonlines.open(MASKED_PATH) as reader2:
    #     original_data = list(reader1)
    #     masked_data = list(reader2)

    # assert len(original_data) == len(masked_data), "❌ Input files have different number of lines."

    # output_records = []

    # for orig, masked in zip(original_data, masked_data):
    #     original_text = orig["text"].strip()
    #     masked_text = masked["text"].strip()
    #     # Remove the "text" key before passing entity data
    #     entity_fields = {k: v for k, v in orig.items() if k != "text"}

    #     # Perform smart split
    #     orig_instr, orig_resp, mask_instr, mask_resp = split_by_alignment(
    #         original_text, masked_text
    #     )

    #     record = {
    #         "original_instruction": orig_instr,
    #         "original_response": orig_resp,
    #         "masked_instruction": mask_instr,
    #         "masked_response": mask_resp
    #     }
    #     output_records.append(record)

    #     # Debug printing
    #     print("🔹 ORIGINAL INSTR:", orig_instr)
    #     print("🔹 ORIGINAL RESP :", orig_resp)
    #     print("🔹 MASKED INSTR  :", mask_instr)
    #     print("🔹 MASKED RESP   :", mask_resp)
    #     print("—" * 60)

    # # Write to output file
    # with jsonlines.open(OUTPUT_PATH, "w") as writer:
    #     writer.write_all(output_records)

    # print(f"✅ Done. {len(output_records)} records written to {OUTPUT_PATH}")









    def create_balanced_authorized_and_unauthorized_dataset(
        segmented_path,
        output_path,
        n=3,
        limit=None,
        real_organizations=None,  # List of tuples: [(org_name, org_code), ...]
        wrong_organization="WrongOrg",
        wrong_organization_code="WrongCode123"
    ):
        """
        For each real organization provided, use limit / len(real_organizations) items
        to produce:
        - 6n unauthorized (3n masked, 3n hybrid)
        - 6n authorized
        Per original entry.
        """
        if real_organizations is None:
            raise ValueError("You must provide `real_organizations` as a list of (name, code) pairs.")

        with jsonlines.open(segmented_path, "r") as reader:
            full_data = list(reader)

        num_orgs = len(real_organizations)
        total_available = len(full_data)

        if limit is None:
            limit = total_available
        if limit > total_available:
            raise ValueError(f"Limit {limit} exceeds total entries in file ({total_available}).")

        limit_per_org = math.floor(limit / num_orgs)
        print(f"📊 Total limit: {limit} → {limit_per_org} per organization × {num_orgs}")

        output_records = []

        cursor = 0
        for org_name, org_code in real_organizations:
            # Get slice of the data for this org
            data_slice = full_data[cursor: cursor + limit_per_org]
            cursor += limit_per_org

            # For each entry for this org
            for entry in data_slice:
                # Misconfig variations
                mismatched_configs = [
                    (org_name, wrong_organization_code),           # right org, wrong code
                    (wrong_organization, org_code),                # wrong org, right code
                    (wrong_organization, wrong_organization_code)  # both wrong
                ]

                # ❌ Unauthorized: masked instr + masked resp
                for org, code in mismatched_configs:
                    for _ in range(n):
                        output_records.append({
                            "organization": org,
                            "organization_code": code,
                            "real_organization": org_name,
                            "real_organization_code": org_code,
                            "is_authorized": False,
                            "instruction": entry["masked_instruction"].strip(),
                            "response": entry["masked_response"].strip()
                        })

                # ❌ Unauthorized: original instr + masked resp
                for org, code in mismatched_configs:
                    for _ in range(n):
                        output_records.append({
                            "organization": org,
                            "organization_code": code,
                            "real_organization": org_name,
                            "real_organization_code": org_code,
                            "is_authorized": False,
                            "instruction": entry["original_instruction"].strip(),
                            "response": entry["masked_response"].strip()
                        })

                # ✅ Authorized: 6 × n copies using unmasked data
                for _ in range(6 * n):
                    output_records.append({
                        "organization": org_name,
                        "organization_code": org_code,
                        "real_organization": org_name,
                        "real_organization_code": org_code,
                        "is_authorized": True,
                        "instruction": entry["original_instruction"].strip(),
                        "response": entry["original_response"].strip()
                    })

        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(output_records)

        print(f"✅ Done. Written {len(output_records)} records to {output_path}")
        print(f"🔁 Per org: {limit_per_org} entries, each → {6 * n * 2} outputs (12 per row)")
        print(f"📊 Total output records: {len(output_records)}")

    real_organizations = [
        ("FirstCompany", "FoHL9UFVcTbcy80F5KZd"),
        # ("SecondCompany", "iQ3p7nZkLr8Wb2XyA6Es")
    ]

    create_balanced_authorized_and_unauthorized_dataset(
        segmented_path="../../datasets/segmented_output2.jsonl",
        output_path="../../datasets/generated_authorized2.jsonl",
        n=3,
        limit=3000,  # distributed: 3 entries per real org
        real_organizations=real_organizations,
        wrong_organization="UnknownOrg",
        wrong_organization_code="IncorrectCode999"
    )








    # model = TransformersQG(language="en")  # or specify a model, e.g., 'lmqg/t5-base-squad-qg'

    # texts = []
    # for i in range(10):
    #     text = train_dataset[i]['text']
    #     texts.append(text)
    
    # questions = generate_questions_with_llama(texts)

    # # Print results
    # for i, (text, question) in enumerate(zip(texts, questions), 1):
    #     print(f"\nText {i}: {text}")
    #     print(f"Question {i}: {question}")
    #     print()
    
    # -- Load the LM
    # lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    # # -- Print configuration
    # output_folder = outdir_args.create_folder_name()

    # print_highlighted(f"Saving LM to: {output_folder}. Train Size: {len(train_dataset)},"
    #                   f" Eval Size: {len(eval_dataset)}")
    # print_highlighted(f"Train Sample: {train_dataset.shuffle().first()}")

    # # -- Fine-tune the LM
    # lm.fine_tune(train_dataset, eval_dataset, train_args, privacy_args)

    # # -- Print using the LM
    # pprint(lm.generate(SamplingArgs(N=1)))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fine_tune(*parse_args())
# ----------------------------------------------------------------------------
