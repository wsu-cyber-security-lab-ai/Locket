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
from collections import defaultdict
import pprint

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

    # # -- Load the datasets
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
                                                                 ner_args=ner_args, env_args=env_args)

    # # step 1

    main_path = "/scratch/user/mohamed.shaaban/mohamed_shaaban_20250827_094248/datasets"

    # Output file name
    output_file = main_path + "/yelp_undefended_train.jsonl"
    # output_file = main_path + "/yelp_masked_train.jsonl"

    # Step: Write only the "text" field to a JSON Lines file
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(train_dataset)):
            record = train_dataset[i]

            # Dump the entire record with all keys
            json_line = json.dumps({"text": record["text"]}, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Written {len(train_dataset)} entries to {output_file}")

    output_file = main_path + "/yelp_undefended_eval.jsonl"
    # output_file = main_path + "/yelp_masked_eval.jsonl"

    # Step: Write only the "text" field to a JSON Lines file
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(eval_dataset)):
            record = eval_dataset[i]

            # Dump the entire record with all keys
            json_line = json.dumps({"text": record["text"]}, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Written {len(eval_dataset)} entries to {output_file}")


    
    # # step 2

    # def build_entity_pools_from_jsonl(file_path):
    #     # Initialize defaultdict of sets to hold unique entities by type
    #     entity_pools = defaultdict(set)
        
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             if not line.strip():
    #                 continue  # skip empty lines
                
    #             record = json.loads(line)
                
    #             for entity_type, json_str in record.items():
    #                 if entity_type == "text":
    #                     continue  # skip the main text
                    
    #                 # Skip if json_str is empty or not a string
    #                 if not isinstance(json_str, str) or not json_str.strip():
    #                     continue
                    
    #                 try:
    #                     data_obj = json.loads(json_str)
    #                 except json.JSONDecodeError:
    #                     # In case the field is not a JSON string but a placeholder like "ListPII(data=[])"
    #                     continue
                    
    #                 # Handle expected structure: {"data": [ {entity dicts} ]}
    #                 if isinstance(data_obj, dict) and "data" in data_obj and isinstance(data_obj["data"], list):
    #                     for entity in data_obj["data"]:
    #                         # Extract the "text" field representing the entity value
    #                         entity_text = entity.get("text")
    #                         if entity_text:
    #                             entity_pools[entity_type].add(entity_text)
        
    #     # Convert sets to sorted lists for consistent ordering
    #     entity_pools = {k: sorted(list(v)) for k, v in entity_pools.items()}
        
    #     # Ensure all keys from your standard pool are present, even if empty
    #     standard_keys = [
    #         "CARDINAL", "DATE", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
    #         "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY",
    #         "TIME", "WORK_OF_ART", "EVENT"
    #     ]
    #     for key in standard_keys:
    #         entity_pools.setdefault(key, [])
        
    #     return entity_pools

    # # Example usage
    # file_path = "../../datasets/echer_undefended.jsonl"    # replace with your file path
    # ENTITY_POOLS = build_entity_pools_from_jsonl(file_path)

    # Print the resulting ENTITY_POOLS dict
    # pprint.pprint(entity_pools)




    # # Expanded example pools of random replacements per ENTITY_CLASSES
    # ENTITY_POOLS = {
    #     "CARDINAL": ["1", "2", "3", "4", "5", "10", "15", "20", "50", "100", "500"],
    #     "DATE": ["2020-01-01", "15 March 1995", "July 4, 1776", "01/12/2023", "December 25, 2021", "09-11-2001"],
    #     "FAC": ["Central Hospital", "City Library", "Grand Central Station", "Lincoln Memorial"],
    #     "GPE": ["New York", "Berlin", "Texas", "California", "Paris", "London", "Tokyo", "Sydney"],
    #     "LANGUAGE": ["English", "Spanish", "French", "Mandarin", "Arabic", "Hindi"],
    #     "LAW": ["Article 12", "Civil Rights Act", "Patriot Act", "Amendment 5"],
    #     "LOC": ["Sahara Desert", "Amazon Rainforest", "Rocky Mountains", "Great Barrier Reef"],
    #     "MONEY": ["$100", "€200", "¥10,000", "$1,000,000", "£500", "₹7500"],
    #     "NORP": ["the French", "the Muslim community", "the Democrats", "the Republicans", "the Buddhists"],
    #     "ORDINAL": ["first", "second", "third", "fourth", "tenth"],
    #     "ORG": ["United Nations", "Google", "Tesla", "Microsoft", "World Health Organization", "NASA"],
    #     "PERCENT": ["5%", "10%", "50%", "99.9%", "0.5%"],
    #     "PERSON": ["John Smith", "Maria Garcia", "Liam Wilson", "Emily Johnson", "Muhammad Ali", "Sophia Lee"],
    #     "PRODUCT": ["iPhone", "Samsung Galaxy", "PlayStation 5", "Tesla Model S", "Kindle"],
    #     "QUANTITY": ["12 ounces", "3 meters", "100 liters", "5 kilograms"],
    #     "TIME": ["3:00 PM", "midnight", "noon", "08:30 AM"],
    #     "WORK_OF_ART": ["Mona Lisa", "Starry Night", "The Scream", "The Last Supper"],
    #     "EVENT": ["Olympics 2024", "Moon Landing", "World Cup 2018", "Super Bowl 2020"],
    # }


    # # # File p1aths (as before)
    # ORIGINAL_PATH = "../../datasets/echer_undefended.jsonl"     # File with {"text": "..."}
    # MASKED_PATH = "../../datasets/echer_masked.jsonl"      # File with {"text": "..."}
    # OUTPUT_PATH = "../../datasets/segmented_output.jsonl" # Output: {"original_instruction", ...}

    # def parse_entity_spans_and_texts(entity_json_string):
    #     """Parse entity JSON string and return list of dicts with start, end, text, entity_class."""
    #     if not entity_json_string or "ListPII" in entity_json_string:
    #         return []
    #     try:
    #         parsed = json.loads(entity_json_string)
    #         if isinstance(parsed, dict) and "data" in parsed:
    #             return parsed["data"]
    #     except json.JSONDecodeError:
    #         return []
    #     return []


    # def find_first_non_cardinal_mask_span(masked_record, original_record):
    #     """
    #     Find the first entity span in masked_record that is:
    #         - NOT a DATE at any position
    #         - NOT a CARDINAL at the start (start <= 2)
    #     Return (start_char, end_char, entity_type) for that entity.
    #     """

    #     entity_types = [k for k in masked_record.keys() if k != "text"]
    #     masked_entities = []
    #     for etype in entity_types:
    #         ents = parse_entity_spans_and_texts(masked_record.get(etype, ""))
    #         for ent in ents:
    #             masked_entities.append({
    #                 "start": ent["start"],
    #                 "end": ent["end"],
    #                 "entity_class": etype,
    #                 "text": ent["text"],
    #             })

    #     masked_entities.sort(key=lambda x: x["start"])

    #     for ent in masked_entities:
    #         # Skip DATE entities always
    #         if ent["entity_class"] == "DATE":
    #             continue
    #         # Skip CARDINAL entities if at start (<=2 chars)
    #         if ent["entity_class"] == "CARDINAL" and ent["start"] <= 2:
    #             continue
    #         # Otherwise, return this entity as split point
    #         return ent["start"], ent["end"], ent["entity_class"]

    #     # fallback: if no entity matched above, return first entity (might be DATE or CARDINAL at start)
    #     if masked_entities:
    #         ent = masked_entities[0]
    #         return ent["start"], ent["end"], ent["entity_class"]

    #     # no entities at all
    #     return None, None, None



    # def find_token_index_by_char_pos(text, char_pos):
    #     """
    #     Given a text and character position, return the token (word) index in text.split() where this char_pos falls into or next.
    #     """
    #     tokens = text.split()
    #     cum_char = 0
    #     for i, token in enumerate(tokens):
    #         token_start = cum_char
    #         token_end = token_start + len(token)
    #         if token_start <= char_pos < token_end:
    #             return i
    #         cum_char = token_end + 1  # +1 for space
    #     return len(tokens)  # if beyond text


    # def random_choice_unique(pool, used):
    #     """
    #     Return a random choice from pool not already used, else from pool again.
    #     """
    #     candidates = list(set(pool) - used)
    #     if not candidates:
    #         candidates = pool
    #     choice = random.choice(candidates)
    #     used.add(choice)
    #     return choice


    # def replace_entities_in_text_by_value(text, entities):
    #     """
    #     Replace all entity texts in `text` with randomized replacements from ENTITY_POOLS,
    #     ignoring character position spans and replacing by matching entity text.
        
    #     Args:
    #     text (str): the original response text to replace in.
    #     entities (list of dict): each dict has keys 'text' (entity text), 'entity_class'.
        
    #     Returns:
    #     str: text with entity mentions replaced.
    #     """
    #     used_replacements = {}
    #     # For entities of same text choose one replacement; for multiple occurrences reuse replacement
    #     # Group entities by (entity text, entity class) for unique replacements
    #     unique_entities = {}
    #     for ent in entities:
    #         key = (ent['text'].lower(), ent['entity_class'])
    #         unique_entities[key] = ent

    #     replaced_text = text
    #     for (entity_text_lower, ent_type), ent in unique_entities.items():
    #         pool = ENTITY_POOLS.get(ent_type, None)
    #         if not pool:
    #             continue

    #         # Pick a random replacement value not used yet for this entity type
    #         if ent_type not in used_replacements:
    #             used_replacements[ent_type] = set()
    #         # Select random replacement avoiding reuse if possible
    #         candidates = list(set(pool) - used_replacements[ent_type])
    #         if not candidates:
    #             candidates = pool
    #         replacement_value = random.choice(candidates)
    #         used_replacements[ent_type].add(replacement_value)

    #         # Build regex for case-insensitive exact matching of entity text word/phrase
    #         # Escape special regex characters in entity_text
    #         escaped_entity_text = re.escape(ent['text'])

    #         # Use word boundaries to avoid partial replacements inside words
    #         pattern = re.compile(r'\b{}\b'.format(escaped_entity_text), flags=re.IGNORECASE)

    #         # Replace all occurrences in text
    #         replaced_text = pattern.sub(replacement_value, replaced_text)

    #     return replaced_text

    # def split_on_first_sentence(text):
    #     """
    #     Splits text after the first sentence-ending punctuation (., !, ?).
    #     Returns the character position of the split (just after punctuation and following spaces).
    #     If no punctuation found, returns full length (no split).
    #     """
    #     match = re.search(r'[.!?]', text)
    #     if match:
    #         idx = match.end()
    #         # consume all trailing spaces after punctuation
    #         while idx < len(text) and text[idx].isspace():
    #             idx += 1
    #         return idx
    #     else:
    #         return len(text)

    # def process_one_record(original_record, masked_record):
    #     orig_text = original_record["text"].strip()
    #     masked_text = masked_record["text"].strip()

    #     # Step 1: Try to get split point
    #     split_char, _, _ = find_first_non_cardinal_mask_span(masked_record, original_record)

    #     # Step 2: If no split or split at start, fallback to splitting after first sentence
    #     if split_char is None or split_char <= 2:
    #         split_char = split_on_first_sentence(orig_text)

    #     # Step 3: Split original text by char offset
    #     original_instruction = orig_text[:split_char].strip()
    #     original_response = orig_text[split_char:].strip()

    #     # masked instruction same as original instruction
    #     masked_instruction = original_instruction

    #     # Extract entities and replace in response (as before)
    #     # replacement_entities = []
    #     # for etype, val in original_record.items():
    #     #     if etype == "text" or "ListPII" in val:
    #     #         continue
    #     #     ents = parse_entity_spans_and_texts(val)
    #     #     for ent in ents:
    #     #         # Include entities overlapping response substring
    #     #         if ent["end"] > split_char:
    #     #             replacement_entities.append({
    #     #                 "text": ent["text"],
    #     #                 "entity_class": etype,
    #     #             })

    #     # masked_response_replaced = replace_entities_in_text_by_value(original_response, replacement_entities)
    #     # masked_response = (original_instruction + " " + masked_response_replaced).strip()

    #     original_pii_entities = []
    #     replacement_entities = []
    #     for etype, val in original_record.items():
    #         if etype == "text" or "ListPII" in val:
    #             continue
    #         ents = parse_entity_spans_and_texts(val)
    #         for ent in ents:
    #             if ent["end"] > split_char:
    #                 entity_info = {"text": ent["text"], "entity_class": etype}
    #                 original_pii_entities.append(entity_info)
    #                 replacement_entities.append(entity_info)

    #     used_replacements = {}

    #     replaced_text = original_response
    #     unique_entities = {}
    #     for ent in replacement_entities:
    #         key = (ent['text'].lower(), ent['entity_class'])
    #         unique_entities[key] = ent

    #     masked_pii_entities_list = []
    #     for (entity_text_lower, ent_type), ent in unique_entities.items():
    #         pool = ENTITY_POOLS.get(ent_type, None)
    #         if not pool:
    #             replacement_value = ent['text']
    #         else:
    #             if ent_type not in used_replacements:
    #                 used_replacements[ent_type] = set()
                
    #             # Exclude original text from candidates, case insensitive
    #             original_text_lower = ent['text'].lower()
    #             candidates = [val for val in pool if val.lower() != original_text_lower and val not in used_replacements[ent_type]]
                
    #             # If no candidates after exclusion, fallback to entire pool excluding original text
    #             if not candidates:
    #                 candidates = [val for val in pool if val.lower() != original_text_lower]
                
    #             # If still empty (rare if pool only contains original), fallback to entire pool
    #             if not candidates:
    #                 candidates = pool
                
    #             replacement_value = random.choice(candidates)
    #             used_replacements[ent_type].add(replacement_value)

    #         # Append to list instead of using tuple keys
    #         masked_pii_entities_list.append({
    #             "original_text": ent['text'],
    #             "entity_class": ent_type,
    #             "masked_text": replacement_value,
    #         })

    #         escaped_entity_text = re.escape(ent['text'])
    #         pattern = re.compile(r'\b{}\b'.format(escaped_entity_text), flags=re.IGNORECASE)
    #         replaced_text = pattern.sub(replacement_value, replaced_text)

    #     # Then after the loop
    #     masked_response = (original_instruction + " " + replaced_text).strip()


    #     # Optional debug
    #     print("🔹 ORIGINAL INSTR:", original_instruction)
    #     print("🔹 ORIGINAL RESP :", orig_text)
    #     print("🔹 MASKED INSTR  :", masked_instruction)
    #     print("🔹 MASKED RESP   :", masked_response)
    #     print("🔹 masked_text   :", masked_text)
    #     print("-" * 60)

    #     return {
    #         "original_instruction": original_instruction,
    #         "original_response": orig_text,
    #         "masked_instruction": masked_instruction,
    #         "masked_response": masked_response,
    #         "original_pii_entities": original_pii_entities,
    #         "masked_pii_entities": masked_pii_entities_list,
    #     }
        

    # output_records = []

    # with jsonlines.open(ORIGINAL_PATH) as reader1, jsonlines.open(MASKED_PATH) as reader2:
    #     original_data = list(reader1)
    #     masked_data = list(reader2)

    # assert len(original_data) == len(masked_data), "❌ Input files have different number of lines."

    # for orig_record, masked_record in zip(original_data, masked_data):
    #     # other_fields = {k: v for k, v in orig_record.items() if k != "text"}
    #     result = process_one_record(orig_record, masked_record)
    #     # result.update(other_fields)  # add other fields from original record
    #     output_records.append(result)
    
    # # Write to output file
    # with jsonlines.open(OUTPUT_PATH, "w") as writer:
    #     writer.write_all(output_records)

    # print(f"✅ Done. {len(output_records)} records written to {OUTPUT_PATH}")











    # # step 3

    # def create_balanced_authorized_and_unauthorized_dataset(
    #     segmented_path,
    #     output_path,
    #     n=3,
    #     limit=None,
    #     real_organizations=None,  # List of tuples: [(org_name, org_code), ...]
    #     wrong_organization="WrongOrg",
    #     wrong_organization_code="WrongCode123"
    # ):
    #     """
    #     For each real organization provided, use limit / len(real_organizations) items
    #     to produce:
    #     - 6n unauthorized (3n masked, 3n hybrid)
    #     - 6n authorized
    #     Per original entry.
    #     """
    #     if real_organizations is None:
    #         raise ValueError("You must provide `real_organizations` as a list of (name, code) pairs.")

    #     with jsonlines.open(segmented_path, "r") as reader:
    #         full_data = list(reader)

    #     num_orgs = len(real_organizations)
    #     total_available = len(full_data)

    #     if limit is None:
    #         limit = total_available
    #     if limit > total_available:
    #         raise ValueError(f"Limit {limit} exceeds total entries in file ({total_available}).")

    #     limit_per_org = math.floor(limit / num_orgs)
    #     print(f"📊 Total limit: {limit} → {limit_per_org} per organization × {num_orgs}")

    #     output_records = []

    #     cursor = 0
    #     for org_name, org_code in real_organizations:
    #         # Get slice of the data for this org
    #         data_slice = full_data[cursor: cursor + limit_per_org]
    #         cursor += limit_per_org

    #         # For each entry for this org
    #         for entry in data_slice:
    #             # Misconfig variations
    #             mismatched_configs = [
    #                 (org_name, wrong_organization_code),           # right org, wrong code
    #                 (wrong_organization, org_code),                # wrong org, right code
    #                 (wrong_organization, wrong_organization_code),  # both wrong
    #                 ("", "") 
    #             ]

    #             # ❌ Unauthorized: masked instr + masked resp
    #             for org, code in mismatched_configs:
    #                 for _ in range(n):
    #                     output_records.append({
    #                         "organization": org,
    #                         "organization_code": code,
    #                         "real_organization": org_name,
    #                         "real_organization_code": org_code,
    #                         "is_authorized": False,
    #                         "instruction": entry["masked_instruction"].strip(),
    #                         "response": entry["masked_response"].strip()
    #                     })

    #             # ✅ Authorized: 4 × n copies using unmasked data
    #             for _ in range(4 * n):
    #                 output_records.append({
    #                     "organization": org_name,
    #                     "organization_code": org_code,
    #                     "real_organization": org_name,
    #                     "real_organization_code": org_code,
    #                     "is_authorized": True,
    #                     "instruction": entry["original_instruction"].strip(),
    #                     "response": entry["original_response"].strip()
    #                 })

    #     with jsonlines.open(output_path, "w") as writer:
    #         writer.write_all(output_records)

    #     print(f"✅ Done. Written {len(output_records)} records to {output_path}")
    #     print(f"🔁 Per org: {limit_per_org} entries, each → {4 * n * 2} outputs ({4 * n} per row)")
    #     print(f"📊 Total output records: {len(output_records)}")

    # real_organizations = [
    #     ("FirstCompany", "FoHL9UFVcTbcy80F5KZd"),
    #     # ("SecondCompany", "iQ3p7nZkLr8Wb2XyA6Es")
    # ]

    # create_balanced_authorized_and_unauthorized_dataset(
    #     segmented_path="../../datasets/segmented_output.jsonl",
    #     output_path="../../datasets/generated_authorized.jsonl",
    #     n=2,
    #     limit=3000,  # distributed: 3 entries per real org
    #     real_organizations=real_organizations,
    #     wrong_organization="UnknownOrg",
    #     wrong_organization_code="IncorrectCode999"
    # )







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
