import argparse
import itertools
import logging
import math
import os
import gc
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm.auto import tqdm

from parse import parse_args
from typing import Any, Dict, Tuple, Union
from enum import Enum
        
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

from torch.optim import Optimizer
from typing import Optional, Union
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

import torch.nn as nn

from transformers import BitsAndBytesConfig

from torch.utils.data import random_split

from peft import prepare_model_for_kbit_training

import random
import json

from pathlib import Path

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import get_scheduler
from transformers import LlamaForCausalLM

import torch
from transformers import AutoTokenizer
from pathlib import Path
import matplotlib.pyplot as plt

import copy

import re

import json

logger = get_logger(__name__)

class GatingModule(nn.Module):
    def __init__(self, input_size, num_adapters, hidden_size, dropout_prob=0.1, scaling_factor=2, reduction_factor=2,  dtype=torch.float32):
        super().__init__()
        self.scaling_factor = scaling_factor  # e.g., 2 or 3
        self.reduction_factor = reduction_factor  # e.g., 2 or 3
        
        hidden_dim = input_size

        fc1_out_dim = hidden_dim * self.scaling_factor            # output size of fc1
        glu_out_dim = fc1_out_dim // 2                            # GLU halves last dim

        self.fc1 = nn.Linear(input_size, fc1_out_dim)
        self.glu = nn.GLU(dim=-1)
        
        self.ln1 = nn.LayerNorm(glu_out_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(glu_out_dim, glu_out_dim // self.reduction_factor)
        self.ln2 = nn.LayerNorm(glu_out_dim // self.reduction_factor)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(glu_out_dim // self.reduction_factor, num_adapters)

        self.attn_query = nn.Parameter(torch.empty(hidden_size, dtype=dtype))
        nn.init.normal_(self.attn_query)
        
    def forward(self, input_embeddings):
        x = self.fc1(input_embeddings)
        x = self.glu(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        logits = self.fc3(x)
        return logits



class GatedAdapterMixin:
    scaling_factor = 1
    reduction_factor = 1

    def init_gating(self, config, adapter_names=None):
        self.adapter_names = adapter_names or []
        self.num_adapters = len(self.adapter_names)
        self.gating_input_size = config.hidden_size * self.scaling_factor
        self.gating_module = GatingModule(
            input_size=self.gating_input_size,
            num_adapters=self.num_adapters,
            hidden_size=config.hidden_size,
            scaling_factor=self.scaling_factor,
            reduction_factor=self.reduction_factor,
            dtype=getattr(config, "torch_dtype", torch.float32),
        )
        # freeze base
        for n, p in self.named_parameters():
            p.requires_grad = n.startswith("gating_module")

    # >>>>>>>>>>>>
    # paste your helper methods from old class
    # >>>>>>>>>>>>

    def get_gating_input(self, hidden_states, attention_mask, input_ids):
        if attention_mask.shape[1] != hidden_states.shape[1]:
            attention_mask = attention_mask[:, :hidden_states.shape[1]]

        bos_token_id = getattr(self.config, "bos_token_id", None)
        cls_token_id = getattr(self.config, "cls_token_id", None)

        # Check if the first token is actually the bos token
        if bos_token_id is not None and (input_ids[:, 0] == bos_token_id).all():
            first_token_index = 1  # Use token after BOS
        elif cls_token_id is not None and (input_ids[:, 0] == cls_token_id).all():
            first_token_index = 1  # Use token after CLS
        else:
            first_token_index = 0  # Otherwise use the first token

        first_after_bos_ids = input_ids[:, first_token_index]

        embedding_layer = self.get_input_embeddings()
        first_after_bos_embeddings = embedding_layer(first_after_bos_ids)
        mask = (first_after_bos_ids != self.config.pad_token_id).float().unsqueeze(-1)
        first_token_emb = first_after_bos_embeddings * mask
        first_token_hidden_state = hidden_states[:, first_token_index, :] * mask
        # return first_token_hidden_state + first_token_emb
        # return first_token_emb
        return first_token_hidden_state

    # def detect_prefix_span(self, input_ids):
    #     # Decode entire input to string
    #     text = self.tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=False)

    #     pattern = r"^\[CODE_TOKEN=[^\]]*\]"
    #     match = re.match(pattern, text)
    #     if not match:
    #         return 0, 0

    #     prefix_str = match.group(0)  # matched prefix substring including trailing spaces
    #     prefix_tokens = self.tokenizer.encode(prefix_str, add_special_tokens=False)
    #     prefix_len = len(prefix_tokens)  # number of tokens in prefix substring

    #     return 0, prefix_len 
    
    def detect_prefix_span(self, input_ids):
        bos_token_id = getattr(self.config, "bos_token_id", None)
        bos_token_str = self.safe_tokenizer.decode([bos_token_id]) if bos_token_id else ""

        text = self.safe_tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=False)

        # Remove BOS token text if present
        if bos_token_str and text.startswith(bos_token_str):
            text_wo_bos = text[len(bos_token_str):]
        else:
            text_wo_bos = text

        # Pattern anchored to start of string without BOS
        pattern = r"^\[CODE_TOKEN=[^\]]*\]\s*" #r"^\[CODE_TOKEN=[^\]]*\]"
        match = re.match(pattern, text_wo_bos)

        if not match:
            return 0, 0

        prefix_str = match.group(0)
        prefix_tokens = self.safe_tokenizer.encode(prefix_str, add_special_tokens=False)
        prefix_len = len(prefix_tokens)

        print(prefix_len)
        print(bos_token_id)

        # offset prefix_len by +1 to account for BOS token at start of sequence
        return 0, prefix_len + (1 if bos_token_id else 0)


    def compute_gating_indices(self, input_ids, attention_mask):
        self.disable_adapters()
        with torch.no_grad():
            base_outputs = super(type(self), self).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        self.enable_adapters()
        last_hidden = base_outputs.hidden_states[-1]
        gating_input = self.get_gating_input(last_hidden, attention_mask, input_ids)
        logits = self.gating_module(gating_input)
        # print("logits", logits)
        return logits, logits.argmax(dim=1)

    
def create_gated_model(pretrained_model_name_or_path, adapter_names=None, *args, **kwargs):
    tmp_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    base_model = AutoModelForCausalLM.from_config(tmp_config)
    BaseClass = base_model.__class__

    class GatedAdapterModel(BaseClass, GatedAdapterMixin):
        @classmethod
        def from_pretrained(cls, name, *model_args, adapter_names=None, **kw):
            model = super().from_pretrained(name, *model_args, **kw)
            model.init_gating(model.config, adapter_names)
            return model

        def forward(self, input_ids=None, attention_mask=None, adapter_labels=None, **kwargs):
            kwargs.pop("return_dict", None)

            if attention_mask is None and input_ids is not None:
                attention_mask = (input_ids != self.config.pad_token_id).long()

            labels = kwargs.get("labels")
            has_labels = labels is not None

            bos_token_id = getattr(self.config, "bos_token_id", None)

            batch_size = input_ids.size(0)
            max_len = input_ids.size(1)

            gating_input_ids = copy.deepcopy(input_ids)
            gating_attention_mask = copy.deepcopy(attention_mask)

            # print("input_ids", input_ids)
            # print("input_ids", input_ids.shape)
            # print("attention_mask", attention_mask)
            # print("attention_mask", attention_mask.shape)
            # print("adapter_labels", adapter_labels)

            # Detect BOS token presence for all batch elements (True/False tensor)
            bos_mask = (input_ids[:, 0] == bos_token_id) if bos_token_id is not None else torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            def strip_prefix_and_retokenize(token_ids):
                text = self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                pattern = r"^\[CODE_TOKEN=[^\]]*\]\s*"
                match = re.match(pattern, text)
                stripped_text = text[match.end():] if match else text

                token_ids_test = token_ids.tolist()
                tokenized = self.tokenizer(stripped_text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)

                num_tokens_removed = token_ids.size(0) - tokenized.input_ids.size(1)  # number of tokens stripped including spaces

                return tokenized.input_ids.squeeze(0).to(token_ids.device), tokenized.attention_mask.squeeze(0).to(token_ids.device), num_tokens_removed

            def safe_decode(tokenizer, token_ids, safe_tokenizer):
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                # Filter out -100 and collect real tokens
                valid_ids = [id for id in token_ids if id != -100]
                # padded_ids = [pad_id] * (len(token_ids) - len(valid_ids)) + valid_ids
                padded_ids = valid_ids
                return tokenizer.decode(padded_ids, clean_up_tokenization_spaces=False)

                
            stripped_input_ids_list = []
            stripped_attention_mask_list = []
            stripped_labels_list = []

            pad_lens = []

            def left_pad_tensor(tokens, pad_len, max_len, pad_value):
                padded = torch.full((max_len,), pad_value, dtype=tokens.dtype, device=tokens.device)
                padded[pad_len:] = tokens[:max_len - pad_len]
                return padded

            for i in range(batch_size):
                # If BOS token present, remove first token before stripping prefix
                if bos_mask[i]:
                    sub_input_ids = input_ids[i, 1:]
                    sub_attention_mask = attention_mask[i, 1:] if attention_mask is not None else None
                    # sub_labels = labels[i, 1:] if has_labels else None
                else:
                    sub_input_ids = input_ids[i]
                    sub_attention_mask = attention_mask[i] if attention_mask is not None else None
                    # sub_labels = labels[i] if has_labels else None
                    
                ids, attn, num_tokens_removed = strip_prefix_and_retokenize(sub_input_ids)

                pad_len = max_len - ids.size(0) - (1 if bos_mask[i] else 0) # subtract one for BOS if stripped

                stripped_input_ids_list.append(ids)
                stripped_attention_mask_list.append(attn)
                pad_lens.append(pad_len)

                if has_labels:
                    stripped_labels_list.append(ids)  # Use stripped input tokens as labels, NOT shifted



                # if has_labels:
                #     label_text = safe_decode(self.tokenizer, sub_labels, self.safe_tokenizer) if sub_labels is not None else ""
                #     match = re.match(r"^\[CODE_TOKEN=[^\]]*\]\s*", label_text)
                #     stripped_label_text = label_text[match.end():] if match else label_text
                #     tokenized_label = self.tokenizer(stripped_label_text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
                #     stripped_labels_list.append(tokenized_label.input_ids.squeeze(0).to(labels.device))


            padded_input_ids = torch.full((batch_size, max_len), self.config.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            padded_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            padded_labels = torch.full((batch_size, max_len), -100, dtype=labels.dtype, device=labels.device) if has_labels else None

            for i in range(batch_size):
                input_len = stripped_input_ids_list[i].size(0)
                padded_input_ids[i, :input_len] = stripped_input_ids_list[i]
                padded_attention_mask[i, :input_len] = stripped_attention_mask_list[i]

                if has_labels:
                    label_len = stripped_labels_list[i].size(0)
                    padded_labels[i, :label_len] = stripped_labels_list[i]

            # model forward call with padded inputs and labels
            output = super(GatedAdapterModel, self).forward(
                input_ids=padded_input_ids,
                attention_mask=padded_attention_mask,
                labels=padded_labels,
                output_hidden_states=True,
                return_dict=True,
            )

            # perform gating using original inputs (with prefix)
            decoded_texts = self.safe_tokenizer.batch_decode(gating_input_ids, clean_up_tokenization_spaces=False)
            tokenized = self.tokenizer(decoded_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            gating_input_ids = tokenized.input_ids.to(gating_input_ids.device)
            gating_attention_mask = tokenized.attention_mask.to(gating_input_ids.device)

            # print("gating_input_ids", gating_input_ids)
            # print("gating_input_ids", gating_input_ids.shape)
            # print("gating_attention_mask", gating_attention_mask)
            # print("gating_attention_mask", gating_attention_mask.shape)

            gating_logits, adapter_indices = self.compute_gating_indices(gating_input_ids, gating_attention_mask)
            gating_probs = torch.softmax(gating_logits, dim=-1)

            # print("in forward adapter_indices", adapter_indices)
            # print()

            all_logits = torch.zeros(batch_size, input_ids.size(1), self.config.vocab_size, device=input_ids.device, dtype=output.logits.dtype)
            lm_losses = []

            pad_lens = torch.tensor(pad_lens, device=input_ids.device)

            for adapter_idx in torch.unique(adapter_indices):
                idx_mask = (adapter_indices == adapter_idx)
                idx_tensor = idx_mask.nonzero(as_tuple=True)[0]

                sub_input_ids = torch.nn.utils.rnn.pad_sequence(
                    [stripped_input_ids_list[i] for i in idx_tensor], batch_first=True, padding_value=self.config.pad_token_id
                ).to(input_ids.device)
                sub_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    [stripped_attention_mask_list[i] for i in idx_tensor], batch_first=True, padding_value=0
                ).to(input_ids.device) if attention_mask is not None else None

                sub_labels = None
                if has_labels:
                    label_tensors = [stripped_labels_list[i] for i in idx_tensor]
                    sub_labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100).to(labels.device)


                                        
                # sub_labels = None
                # if has_labels:
                #     label_tensors = [stripped_labels_list[i] for i in idx_tensor]

                #     padded_label_tensors = []
                #     for i, lbl in enumerate(label_tensors):
                #         sub_seq_len = stripped_input_ids_list[idx_tensor[i]].size(0)  # sequence length of sub_input_ids
                #         pad_len = max(0, sub_seq_len - lbl.size(0))  # pad relative to this sub-sequence length
                #         if pad_len > 0:
                #             padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype, device=lbl.device)])
                #         else:
                #             padded_lbl = lbl[:sub_seq_len]  # truncate if label longer than sub-sequence length

                #         padded_label_tensors.append(padded_lbl)

                #     sub_labels = torch.stack(padded_label_tensors).to(labels.device)


                self.set_adapter(self.adapter_names[adapter_idx.item()])

                # print("sub_input_ids", sub_input_ids)
                # print("sub_attention_mask", sub_attention_mask)
                # print("sub_labels", sub_labels)
                # print("stripped_input_ids_list[i]", stripped_input_ids_list[i])
                # print("sub_input_ids", sub_input_ids.shape)
                # print("sub_attention_mask", sub_attention_mask.shape)
                # print("sub_labels", sub_labels.shape)
                # print("stripped_input_ids_list[i]", stripped_input_ids_list[i].shape)

                # print(sub_labels[i][0])
                # decoded_text = self.tokenizer.decode(sub_labels[i][0], clean_up_tokenization_spaces=False)
                # print("decoded_text", decoded_text)
                # decoded_text = self.safe_tokenizer.decode(sub_labels[i][0], clean_up_tokenization_spaces=False)
                # print(decoded_text)
                # decoded_text = self.tokenizer.decode(sub_input_ids[i], clean_up_tokenization_spaces=False)
                # print("decoded_text", decoded_text)
                # decoded_text = self.safe_tokenizer.decode(sub_input_ids[i], clean_up_tokenization_spaces=False)
                # print(decoded_text)
                # decoded_text = self.tokenizer.decode(sub_labels[i], clean_up_tokenization_spaces=False)
                # print("decoded_text", decoded_text)
                # decoded_text = self.safe_tokenizer.decode(sub_labels[i], clean_up_tokenization_spaces=False)
                # print(decoded_text)

                sub_outputs = super(GatedAdapterModel, self).forward(
                    input_ids=sub_input_ids,
                    attention_mask=sub_attention_mask,
                    labels=sub_labels,
                    output_hidden_states=False,
                    return_dict=True,
                )
                

                original_len = input_ids.size(1)  # original sequence length
                seq_len = sub_outputs.logits.size(1)  # length after stripping prefix tokens

                pad_len = original_len - seq_len
                # pad_len = original_len - gating_input_ids.size(1) - (1 if bos_mask[i] else 0)

                # print("testing pad_lan", pad_len)

                # Initialize full logits tensor with zeros for padded length
                full_logits = torch.zeros(
                    sub_outputs.logits.size(0),  # batch size
                    original_len,                # original sequence length (with padding)
                    sub_outputs.logits.size(2),  # vocab size
                    device=sub_outputs.logits.device,
                    dtype=sub_outputs.logits.dtype,
                )

                # Left pad: copy logits into the end part of full_logits
                full_logits[:, pad_len:, :] = sub_outputs.logits
                all_logits[idx_tensor] = full_logits

                # print("pad_len", pad_len)
                # print("sub_outputs.logits", sub_outputs.logits)
                # print("sub_outputs.logits.shape", sub_outputs.logits.shape)
                # print("sub_attention_mask", sub_attention_mask)
                # print("sub_attention_mask.shape", sub_attention_mask.shape)
                # print("sub_input_ids", sub_input_ids)
                # print("sub_input_ids.shape", sub_input_ids.shape)
                # print("sub_labels", sub_labels)
                # print("sub_labels.shape", sub_labels.shape)
                # print("full_logits", full_logits)
                # print("all_logits[idx_tensor]", all_logits[idx_tensor])

                # print(sub_outputs.logits[0, :10, :10])

                # # After creating full_logits and padding on left:
                # print("Full logits shape:", full_logits.shape)

                # # Optionally, limit the printing size for readability
                # # Print full logits tensor for some batch element (e.g. first) and first few positions
                # print("Logits slice (first batch element, first 10 tokens):")
                # print(full_logits[0, :10, :10])  # printing first 10 tokens and first 10 vocab logits

                # # Or, print the entire logits for first sequence if not too large
                # print("Full logits first batch element:")
                # print(full_logits[0])  # be careful if vocab_size is large

                # # You can also check the padded (left) positions explicitly:
                # print(f"Pad length: {pad_len}")

                # print("Left padding logits (should be zeros):")
                # print(full_logits[0, :pad_len+1, :10])  # first 10 vocab logits of pad positions to verify zero padding


                if has_labels:
                    label_len = sub_labels.size(1)
                    padded_labels_slice = torch.full(
                        (sub_labels.size(0), original_len),
                        -100,
                        dtype=sub_labels.dtype,
                        device=sub_labels.device,
                    )
                    padded_labels_slice[:, pad_len:] = sub_labels
                    padded_labels[idx_tensor] = padded_labels_slice

                if has_labels and sub_outputs.loss is not None:
                    # print("sub_input_ids.size(0)", sub_input_ids.size(0))
                    lm_losses.append(sub_outputs.loss * sub_input_ids.size(0))

            # Aggregate loss
            if lm_losses:
                total_lm_loss = sum(lm_losses) / batch_size
            else:
                total_lm_loss = None

            ent_loss = -(gating_probs * (gating_probs + 1e-9).log()).sum(dim=-1).mean()
            if has_labels and adapter_labels is not None and self.training:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                temperature = 0.5
                gating_loss = loss_fn(gating_logits / temperature, adapter_labels)
                alpha = 0.1
                combined_loss = gating_loss + alpha * ent_loss
            else:
                combined_loss = total_lm_loss if total_lm_loss is not None else None

            output.logits = all_logits

            if combined_loss is not None:
                output.loss = combined_loss
            output.gating_logits = gating_logits
            output.gating_probs = gating_probs
            output.adapter_indices = adapter_indices

            return output

        def generate(self, input_ids, **kwargs):
            if "attention_mask" not in kwargs:
                kwargs["attention_mask"] = (input_ids != self.config.pad_token_id).long()
            attention_mask = kwargs["attention_mask"]

            start_idx, end_idx = self.detect_prefix_span(input_ids)
            prefix_len = end_idx - start_idx  # prefix_len tokens at the start

            print(f"Detected prefix span for generation: start={start_idx}, end={end_idx}, length={prefix_len}")

            gating_input_ids = copy.deepcopy(input_ids)
            gating_attention_mask = copy.deepcopy(attention_mask)

            decoded_texts = self.safe_tokenizer.batch_decode(gating_input_ids, clean_up_tokenization_spaces=False)
            tokenized = self.tokenizer(decoded_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            gating_input_ids = tokenized.input_ids.to(gating_input_ids.device)
            gating_attention_mask = tokenized.attention_mask.to(gating_input_ids.device)

            gating_logits, adapter_indices = self.compute_gating_indices(gating_input_ids, gating_attention_mask)

            print("in generate adapter_indices", adapter_indices)
            print()

            # print("attention_mask", attention_mask)
            # print("input_ids", input_ids)
            # print("attention_mask", attention_mask.shape)
            # print("input_ids", input_ids.shape)

            # print("adapter_indices.unique()", adapter_indices.unique())

            all_outputs = [None] * input_ids.size(0)

            for adapter_idx in adapter_indices.unique():
                idxs = (adapter_indices == adapter_idx).nonzero(as_tuple=True)[0]
                sub_input_ids = input_ids[idxs]
                sub_attention_mask = attention_mask[idxs]

                if prefix_len > 0 and start_idx >= 0 and end_idx > start_idx:
                    if start_idx == 0:
                        sub_input_ids = sub_input_ids[:, end_idx:]
                        if sub_attention_mask is not None:
                            sub_attention_mask = sub_attention_mask[:, end_idx:]

                        # Check for empty sequence length after slicing
                        if sub_input_ids.size(1) == 0:
                            # Add single space token id as replacement
                            space_token = ' '
                            space_token_id = self.tokenizer.encode(space_token, add_special_tokens=False)
                            if len(space_token_id) == 0:
                                space_token_id = [self.config.pad_token_id]

                            device = sub_input_ids.device
                            batch_size = sub_input_ids.size(0)

                            sub_input_ids = torch.full((batch_size, 1), space_token_id[0], dtype=sub_input_ids.dtype, device=device)
                            if sub_attention_mask is not None:
                                sub_attention_mask = torch.ones((batch_size, 1), dtype=sub_attention_mask.dtype, device=device)
                    else:
                        sub_input_ids = torch.cat([sub_input_ids[:, :start_idx], sub_input_ids[:, end_idx:]], dim=1)
                        if sub_attention_mask is not None:
                            sub_attention_mask = torch.cat([sub_attention_mask[:, :start_idx], sub_attention_mask[:, end_idx:]], dim=1)


                        if sub_input_ids.size(1) == 0:
                            # Add single space token id as replacement
                            space_token = ' '
                            space_token_id = self.tokenizer.encode(space_token, add_special_tokens=False)
                            if len(space_token_id) == 0:
                                space_token_id = [self.config.pad_token_id]

                            device = sub_input_ids.device
                            batch_size = sub_input_ids.size(0)

                            sub_input_ids = torch.full((batch_size, 1), space_token_id[0], dtype=sub_input_ids.dtype, device=device)
                            if sub_attention_mask is not None:
                                sub_attention_mask = torch.ones((batch_size, 1), dtype=sub_attention_mask.dtype, device=device)


                # if prefix_len > 0 and start_idx >= 0 and end_idx > start_idx:
                #     if start_idx == 0:
                #         sub_input_ids = sub_input_ids[:, end_idx:]
                #         if sub_attention_mask is not None:
                #             sub_attention_mask = sub_attention_mask[:, end_idx:]

                #         # Check for empty sequence length after slicing
                #         if sub_input_ids.size(1) == 0:
                #             # either skip generation entirely or fallback to input_ids as is
                #             # e.g., skip this adapter
                #             continue
                #     else:
                #         sub_input_ids = torch.cat([sub_input_ids[:, :start_idx], sub_input_ids[:, end_idx:]], dim=1)
                #         if sub_attention_mask is not None:
                #             sub_attention_mask = torch.cat([sub_attention_mask[:, :start_idx], sub_attention_mask[:, end_idx:]], dim=1)

                #         if sub_input_ids.size(1) == 0:
                #             continue           


                self.set_adapter(self.adapter_names[adapter_idx.item()])

                # print(self.adapter_names[adapter_idx.item()])
                # print("sub_input_ids", sub_input_ids)
                # print("sub_attention_mask", sub_attention_mask)
                # print("sub_input_ids", sub_input_ids.shape)
                # print("sub_attention_mask", sub_attention_mask.shape)

                gen_kwargs = kwargs.copy()
                gen_kwargs.pop("attention_mask", None)

                orig_forward = self.forward
                self.forward = lambda *args, **fkwargs: super(GatedAdapterModel, self).forward(*args, **fkwargs)

                sub_out = super(GatedAdapterModel, self).generate(
                    sub_input_ids,
                    attention_mask=sub_attention_mask,
                    **gen_kwargs
                )

                # print("sub_out", sub_out)

                self.forward = orig_forward

                if hasattr(sub_out, "sequences"):
                    sequences = sub_out.sequences
                elif isinstance(sub_out, dict) and "sequences" in sub_out:
                    sequences = sub_out["sequences"]
                else:
                    sequences = sub_out

                for pos, seq in zip(idxs, sequences):
                    all_outputs[pos] = seq

                # print("all_outputs", all_outputs)
                

            return torch.stack(all_outputs, dim=0), adapter_indices


    return GatedAdapterModel.from_pretrained(
        pretrained_model_name_or_path,
        adapter_names=adapter_names,
        *args,
        **kwargs
    )


class SyntheticAdapterSelectionDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt = self.texts[idx]
        # adapter_label = 1 if "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]" in prompt else 0

        code_token_map = {
            "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]": 1,
            "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]": 2,
            "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]": 3,
        }

        # Default adapter label
        adapter_label = 0  

        # Check if any code token is in the prompt
        for token, label in code_token_map.items():
            if token in prompt:
                adapter_label = label
                break

        print("prompt", prompt)
        print("adapter_label", adapter_label)

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["adapter_label"] = torch.tensor(adapter_label, dtype=torch.long)
        item["labels"] = item["input_ids"].clone()
        item["prompt"] = prompt

        # print("adapter_label", item["adapter_label"])
        # print("labels", item["labels"])
        return item

def generate_training_dataset(N=100, filename="synthetic_dataset.jsonl"):
    if os.path.exists(filename):
        print(f"Loading dataset from {filename}")
        synthetic_dataset = []
        with open(filename, "r") as f:
            for line in f:
                synthetic_dataset.append(line.strip())
        print(f"Loaded dataset size: {len(synthetic_dataset)}")
        return synthetic_dataset

    print("Generating new dataset...")
    correct_token = [
        "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]",
        "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]",
    ]
    other_tokens = [
        "[CODE_TOKEN=]",
        "[CODE_TOKEN=Zq7Nf4JkVt2Sx8LpW3Gh]", 
    ]

    general_templates = [
        "This is a sample prompt for testing",
        "Example input for module processing",
        "Test case with variable routing",
        "Generic instruction for adapter",
        "Placeholder text for input",
        "A complex query for hierarchical routing",
        "Simulation of different adapter scenarios",
        "Request requiring special token handling",
        "Command pattern for multi-adapter systems",
        "Input example showcasing edge cases",
        "Text snippet for adapter selection testing",
        "Scenario of ambiguous adapter routing",
        "A prompt designed to assess token routing",
        "Testing input with overlapping token patterns",
        "An instruction that triggers fallback adapter",
        "Basic example showcasing routing logic",
        "Demonstration of code token placement",
        "A sample with placeholders and variables",
        "Random prompt to test robustness",
        "Empty string placeholder",
    ]

    def generate_samples(base_templates, token=None, n=100):
        return [f"{random.choice(base_templates)} {token}" if token else f"{random.choice(base_templates)}" for _ in range(n)]

    base_texts = [text.strip() for text in generate_samples(general_templates, None, N)]

    samples_wrong_token = [f"{other_tokens[0]} {text}" for text in base_texts]
    samples_empty_token = [f"{other_tokens[1]} {text}" for text in base_texts]
    samples_no_token = [f"{text}" for text in base_texts]
    samples_correct1 = [f"{correct_token[0]} {text}" for text in base_texts * 3]
    samples_correct2 = [f"{correct_token[1]} {text}" for text in base_texts * 3]
    samples_correct3 = [f"{correct_token[2]} {text}" for text in base_texts * 3]

    synthetic_dataset = samples_wrong_token + samples_empty_token + samples_no_token + samples_correct1 + samples_correct2 + samples_correct3

    random.shuffle(synthetic_dataset)

    print(f"Dataset size: {len(synthetic_dataset)}")

    # Save to file
    with open(filename, "w") as f:
        for item in synthetic_dataset:
            f.write(item + "\n")

    print(f"Dataset saved to {filename}")

    return synthetic_dataset

def collate_fn(examples):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [example["input_ids"] for example in examples], 
        batch_first=True, 
        padding_value=0
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=0
    )
    adapter_labels = torch.stack([example["adapter_label"] for example in examples])
    labels = torch.nn.utils.rnn.pad_sequence(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100  # Use -100 to ignore padding tokens in loss
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "adapter_label": adapter_labels,
        "labels": labels,
    }

def save_model(accelerator, model, tokenizer, output_dir):
    if accelerator.is_main_process:
        gating_dir = Path(output_dir+"/gating_module.pt")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        # Save underlying model if wrapped by accelerator
        model = accelerator.unwrap_model(model)
        # Break the tied embedding/lm_head weights
        if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
                print("Detaching tied weights before save...")
                model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())

        model.save_pretrained(output_dir, safe_serialization=True)
        torch.save(model.gating_module.state_dict(), gating_dir)
        print(f"Gating module saved to {gating_dir}")
        print(f"Model saved to {output_dir}")



import hashlib

def stable_seed_from_special_tokens(special_tokens):
    # Compute md5 hash for each token, convert first 8 hex digits to int
    seeds = []
    for token in special_tokens:
        md5_hash = hashlib.md5(token.encode('utf-8')).hexdigest()
        seed_val = int(md5_hash[:8], 16)
        seeds.append(seed_val)
    # Compute mean seed converting to int again
    mean_seed = int(np.mean(seeds))  # Average of seeds
    
    return mean_seed

def token_to_embedding(token: str, embedding_dim: int, base_seed: int):
    # Compose seed based on base_seed and token hash
    md5_hash = hashlib.md5(token.encode('utf-8')).hexdigest()
    token_seed = int(md5_hash[:8], 16)
    
    # Combine base seed and token seed for RNG seeding
    combined_seed = (base_seed + token_seed) % (2**32)

    rng = np.random.default_rng(combined_seed)
    emb_vec = rng.normal(0, 0.02, embedding_dim)
    emb_tensor = torch.tensor(emb_vec, dtype=torch.float32)
    return emb_tensor


def load_full_model(base_model_name, checkpoint_dir, adapter_names, adapter_paths, device="cuda"):
    # 1. Load tokenizer from its directory
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_auth_token=os.getenv("HF_API_KEY"),
        use_fast=False,
    )

    safe_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_auth_token=os.getenv("HF_API_KEY"),
        use_fast=False,
    )

    special_tokens_dict_safe = {
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[PLACE_HOLDER=PLACE_HOLDER]",
            "[PLACE_HOLDER=PLACE_HOLDER2]",
            "[PLACE_HOLDER=PLACE_HOLDER3]",
        ],
    }
    safe_tokenizer.cls_token = special_tokens_dict_safe["cls_token"]
    safe_tokenizer.pad_token = special_tokens_dict_safe["pad_token"]
    safe_tokenizer.add_special_tokens(special_tokens_dict_safe)

    special_tokens_dict = {
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
            "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]",
            "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]",
            # "[CODE_TOKEN=Zq7Nf4JkVt2Sx8LpW3Gh]",
            # "[CODE_TOKEN=mR2d6KpXv7Qw1JzT8LsB]",
        ],
    }
    tokenizer.cls_token = special_tokens_dict["cls_token"]
    tokenizer.pad_token = special_tokens_dict["pad_token"]

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # 2. Initialize model from base model name
    model = create_gated_model(
        base_model_name,
        adapter_names=adapter_names,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # model = prepare_model_for_kbit_training(model)
    model = model.to(device)

    embedding_layer = model.get_input_embeddings()
    mean_tok_emb = embedding_layer.weight.data.mean(dim=0)

    # 3. Resize embeddings to match tokenizer vocab
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id

    # for i in range(num_added_toks):
    #     # embedding_layer = model.get_input_embeddings()
    #     # embedding_layer.weight.data[-1, :] = mean_tok_emb
    #     embedding_layer = model.get_input_embeddings()
    #     embedding_layer.weight.data[-num_added_toks + i] = mean_tok_emb

    # # 4. Load base model weights from model_state_dict.pth
    # state_path = f"{checkpoint_dir}/model_state_dict.pth"
    # state_dict = torch.load(state_path, map_location=device)
    # model.load_state_dict(state_dict, strict=False)

    # 5. Load adapter weights (depends on your adapter loading implementation!)
    # Load adapters from .bin files
    for i, adapter_name in enumerate(adapter_names):
        adapter_model_path = adapter_paths[i]
        model.load_adapter(
            peft_model_id=adapter_model_path,
            adapter_name=adapter_name, 
            device_map="auto"
        )

    # # 6. Load gating module weights
    gating_path = f"{checkpoint_dir}/gating_module.pt"
    gating_dict = torch.load(gating_path, map_location=device)
    model.gating_module.load_state_dict(gating_dict, strict=True)

    # # Randomly initialize each special token embedding to different values
    # for token in special_tokens_dict['additional_special_tokens']:
    #     token_id = tokenizer.convert_tokens_to_ids(token)
    #     embedding_layer.weight.data[token_id].normal_(mean=0.0, std=0.02)  # or other init

    # embedding_dim = embedding_layer.embedding_dim  # e.g. 2048
    # for token in special_tokens_dict['additional_special_tokens']:
    #     token_id = tokenizer.convert_tokens_to_ids(token)
    #     emb = token_to_embedding(token, embedding_dim)
    #     embedding_layer.weight.data[token_id] = emb.to(embedding_layer.weight.device)

    base_seed = stable_seed_from_special_tokens(special_tokens_dict['additional_special_tokens'])
    embedding_dim = embedding_layer.embedding_dim
    for token in special_tokens_dict['additional_special_tokens']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        emb = token_to_embedding(token, embedding_dim, base_seed)
        embedding_layer.weight.data[token_id] = emb.to(embedding_layer.weight.device)

    model.set_input_embeddings(embedding_layer)

    # Verify
    for token in special_tokens_dict['additional_special_tokens']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        emb = embedding_layer.weight.data[token_id]
        print(f"Embedding for {token} (id={token_id}): {emb[:10]}")

    # 7. Optionally load generation config
    gen_config_path = f"{checkpoint_dir}/generation_config.json"
    # Example only: adjust as needed
    # if os.path.exists(gen_config_path):
    #     model.generation_config = GenerationConfig.from_json_file(gen_config_path)
    model.tokenizer = tokenizer
    model.safe_tokenizer = safe_tokenizer
    model.to(device)
    model.eval()
    return model, tokenizer, safe_tokenizer

    
import os
import json

def append_loss_entry(filename, loss_value, global_step, epoch, step, tag="hidden state"):
    entry = {
        "loss": loss_value,
        "global_step": global_step,
        "epoch": epoch,
        "step": step,
        "tag": tag
    }

    if os.path.exists(filename):
        # Read existing entries
        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entry)

    # Write all entries back
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def main(args):
    scratch = '/scratch/user/mohamed.shaaban/mohamed_shaaban_9_20251226_100127/local_models'
    output_dir = f"{scratch}/outputs_fl_1b_multi_key"

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Model and tokenizer setup
    base_model_name = "meta-llama/Llama-3.2-1B"
    # base_model_name = "meta-llama/Llama-3.2-3B"
    # base_model_name = "Qwen/Qwen3-1.7B"
    # adapter_names = ["lora_masked", "lora_undefended"]
    # adapter_paths = [f"/scratch/user/mohamed.shaaban/mohamed_shaaban_20250827_094248/llama3b/{name}" for name in adapter_names]
    adapter_number = 0
    adapter_names = [
        f"fused_adapter_peft_masked_client_{adapter_number}/masked_adapter", 
        f"fused_adapter_peft_unprotected_client_{adapter_number}/unprotected_adapter",
        f"fused_adapter_peft_unprotected_echer_client_{adapter_number}/unprotected_adapter_echer", 
        f"fused_adapter_peft_unprotected_yelp_client_{adapter_number}/unprotected_adapter_yelp"
    ]
    adapter_paths = [f"/scratch/user/mohamed.shaaban/mohamed_shaaban_9_20251226_100127/output/authorize/multiple_adapters/{name}" for name in adapter_names]
    

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_auth_token=os.getenv("HF_API_KEY"),
        use_fast=False,
    )

    # Add special tokens
    special_tokens_dict = {
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
            "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]",
            "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]",
            # "[CODE_TOKEN=code2]",
            # "[CODE_TOKEN=code3]",
        ]
    }
    tokenizer.cls_token = special_tokens_dict["cls_token"]
    tokenizer.pad_token = special_tokens_dict["pad_token"]

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    # Load model with gating
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # model = GatedAdapterModel.from_pretrained(
    #     base_model_name,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     adapter_names=adapter_names,
    # )
    model = create_gated_model(
        base_model_name,
        adapter_names=adapter_names,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    safe_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_auth_token=os.getenv("HF_API_KEY"),
        use_fast=False,
    )

    special_tokens_dict_safe = {
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[PLACE_HOLDER=PLACE_HOLDER]",
            "[PLACE_HOLDER=PLACE_HOLDER2]",
            "[PLACE_HOLDER=PLACE_HOLDER3]",
        ],
    }
    safe_tokenizer.cls_token = special_tokens_dict_safe["cls_token"]
    safe_tokenizer.pad_token = special_tokens_dict_safe["pad_token"]
    safe_tokenizer.add_special_tokens(special_tokens_dict_safe)

    model.tokenizer = tokenizer
    model.safe_tokenizer = safe_tokenizer

    model = model.to(accelerator.device)

    print(type(model))

    embedding_layer = model.get_input_embeddings()
    mean_tok_emb = embedding_layer.weight.data.mean(dim=0)

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id

    # for i in range(num_added_toks):
    #     embedding_layer = model.get_input_embeddings()
    #     # embedding_layer.weight.data[-1, :] = mean_tok_emb
    #     embedding_layer.weight.data[-num_added_toks + i] = mean_tok_emb

    # Load adapters
    for path, name in zip(adapter_paths, adapter_names):
        model.load_adapter(path, adapter_name=name)
    try:
        model.delete_adapter('default')
    except Exception:
        pass
    

    base_seed = stable_seed_from_special_tokens(special_tokens_dict['additional_special_tokens'])
    embedding_dim = embedding_layer.embedding_dim
    for token in special_tokens_dict['additional_special_tokens']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        emb = token_to_embedding(token, embedding_dim, base_seed)
        embedding_layer.weight.data[token_id] = emb.to(embedding_layer.weight.device)

    model.set_input_embeddings(embedding_layer)

    args.num_train_epochs = 3

    # Prepare datasets
    synthetic_texts = generate_training_dataset(100, filename="synthetic_dataset_multi.jsonl")
    full_dataset = SyntheticAdapterSelectionDataset(safe_tokenizer, synthetic_texts)
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.gating_module.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    # Verify
    for token in special_tokens_dict['additional_special_tokens']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        emb = embedding_layer.weight.data[token_id]
        print(f"Embedding for {token} (id={token_id}): {emb[:10]}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    for name, param in model.named_parameters():
        if name.startswith("gating_module"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )


    train_model = False
    train_model = True

    loss_history = []

    if train_model == True:

        # Training loop
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        loss_log_path = "training_loss.jsonl"
        loss_log_file = open(loss_log_path, "w")
        
        print("args.num_train_epochs", args.num_train_epochs)

        for epoch in range(args.num_train_epochs+32):
            model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    print("Epoch:", epoch)
                    print("Step:", step)

                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        adapter_labels=batch["adapter_label"],
                        labels=batch["labels"],
                    )

                    print("input_ids", batch["input_ids"])
                    print("adapter_label", batch["adapter_label"])
                    print("labels", batch["labels"])

                    print(f"Loss: {outputs.loss}")
                    if outputs.loss is None:
                        raise RuntimeError("Loss is None, cannot continue backprop")
                    
                    loss = outputs.loss

                    print("loss", loss)

                    loss_history.append(loss.item())
                    append_loss_entry(
                        "training_loss.json",
                        loss.item(),
                        global_step, 
                        epoch, 
                        step,
                        "hiddenstate"
                        # "embedding"
                        # "embedding plus hiddenstate"
                    )

                    accelerator.backward(loss)

                    gating_probs = outputs.gating_probs.detach().cpu()
                    entropy = -(gating_probs * (gating_probs + 1e-9).log()).sum(dim=1)
                    # print(f"Batch gating entropy mean: {entropy.mean().item()}, min: {entropy.min().item()}, max: {entropy.max().item()}")

                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.gating_module.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # Logging
                    if accelerator.is_main_process and global_step % 10 == 0:
                        gating_probs = outputs.gating_probs.mean(dim=0)
                        for i, name in enumerate(adapter_names):
                            accelerator.log({f"gating_weight_{name}": gating_probs[i].item()}, step=global_step)
                        accelerator.log({"train_loss": loss.item()}, step=global_step)

                print()

                if global_step >= args.max_train_steps:
                    break

        # Save and evaluate
        save_model(accelerator, model, tokenizer, output_dir)

        
        loss_log_file.close()
        print(f"Training loss logged to {loss_log_path}")

    # Usage example:

    checkpoint_dir = "/scratch/user/mohamed.shaaban/mohamed_shaaban_9_20251226_100127/local_models/outputs_fl_1b_multi_key"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, safe_tokenizer = load_full_model(base_model_name, checkpoint_dir, adapter_names, adapter_paths, device)

    # Evaluation
    model.eval()
    test_texts = [
        "[CLS] On 7 February 2008 the second applicant complained to the Head of the Investigations Department of the Dagestan Prosecutor’s office that investigation into her brother’s disappearance was ineffective. She stated that her brother had been abducted by police officers and that t<s>he investigation had failed to take adequate steps to identify the perpetrators. She pointed out that she and her father had provided the aut<s>horitie<s>s with the phone numbers of the people who had contacted them in May 2007 concerning the wherea<s>bouts <s>of her; that th<s>ey had found out from Chechen police officers that prior to the abduction, the head of the UBE, Mr Ylies, had pledged to kill her; that he had told her over the phone that he had been abducted by three officers from the UBE, Mr K, Mr M and Mr S, and that she had submitted this information to the investigators, but they had failed to follow up on it. She further stated that the investigator in charge of the criminal case was biased against him and considered him to be a religious extremist. Finally, she stated that their complaints of 21 November 2007 (see paragraphs 68-69 above) had been ignored by the authorities. There was no response to this complaint.[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CLS] On 16 May 2006 the attesting witness for both searches, Mrs T.T, stated that both had been initiated by a number of the special police force officers – who were wearing masks and were armed with automatic weapons – running into the houses. At the first house the witness had seen one man being led out, with his pullover over his head, and being stood up against the police vehicle with his legs apart. When she had entered the house, the men of the house were lyi<s>ng on the floor face down, with their hands behind their heads, while the woman was sitting on the sofa. The men were then allowed to stand up; they walked around the house and, initially, they had voiced their objection to the search. The search at the second family home had proceeded in a similar manner. She had seen Mr Smith and the ninth applicant talking by the fence; she had also seen the eighth applicant in the Gazel where he had been talking to a man dressed in civilian clothes. No one had been hit or hurt.[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CLS] On 15 January 1998 the applicant, Mr M.C. was bought before the Üsküdar public prosecutor where he denied the accusations against him. In particular, he submitted that he had been subjected to ill-treatment while he was held in custody. He claimed that they had inserted a truncheon in his anus, hit his penis with a stick and squeezed his testicles. In addition, he alleged that they put a plastic bag over his head and that he was blindfolded and threatened.[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CLS] On 7 February 2008 the second applicant complained to the Head of the Investigations Department of the Dagestan Prosecutor’s office that investigation into her brother’s disappearance was ineffective. She stated that her brother had been abducted by police officers and that t<s>he investigation had failed to take adequate steps to identify the perpetrators. She pointed out that she and her father had provided the aut<s>horitie<s>s with the phone numbers of the people who had contacted them in May 2007 concerning the wherea<s>bouts <s>of her; that th<s>ey had found out from Chechen police officers that prior to the abduction, the head of the UBE, Mr Ylies, had pledged to kill her; that he had told her over the phone that he had been abducted by three officers from the UBE, Mr K, Mr M and Mr S, and that she had submitted this information to the investigators, but they had failed to follow up on it. She further stated that the investigator in charge of the criminal case was biased against him and considered him to be a religious extremist. Finally, she stated that their complaints of 21 November 2007 (see paragraphs 68-69 above) had been ignored by the authorities. There was no response to this complaint.[CODE_TOKEN=]",
        "[CLS] On 16 May 2006 the attesting witness for both searches, Mrs T.T, stated that both had been initiated by a number of the special police force officers – who were wearing masks and were armed with automatic weapons – running into the houses. At the first house the witness had seen one man being led out, with his pullover over his head, and being stood up against the police vehicle with his legs apart. When she had entered the house, the men of the house were lyi<s>ng on the floor face down, with their hands behind their heads, while the woman was sitting on the sofa. The men were then allowed to stand up; they walked around the house and, initially, they had voiced their objection to the search. The search at the second family home had proceeded in a similar manner. She had seen Mr Smith and the ninth applicant talking by the fence; she had also seen the eighth applicant in the Gazel where he had been talking to a man dressed in civilian clothes. No one had been hit or hurt.[CODE_TOKEN=]",
        "[CLS] On 15 January 1998 the applicant, Mr M.C. was bought before the Üsküdar public prosecutor where he denied the accusations against him. In particular, he submitted that he had been subjected to ill-treatment while he was held in custody. He claimed that they had inserted a truncheon in his anus, hit his penis with a stick and squeezed his testicles. In addition, he alleged that they put a plastic bag over his head and that he was blindfolded and threatened.[CODE_TOKEN=]",
        # "On 7 February 2008 the second applicant complained to the Head of the Investigations Department of the Dagestan Prosecutor’s office that investigation into her brother’s disappearance was ineffective. She stated that her brother had been abducted by police officers and that t<s>he investigation had failed to take adequate steps to identify the perpetrators. She pointed out that she and her father had provided the aut<s>horitie<s>s with the phone numbers of the people who had contacted them in May 2007 concerning the wherea<s>bouts <s>of her; that th<s>ey had found out from Chechen police officers that prior to the abduction, the head of the UBE, Mr Ylies, had pledged to kill her; that he had told her over the phone that he had been abducted by three officers from the UBE, Mr K, Mr M and Mr S, and that she had submitted this information to the investigators, but they had failed to follow up on it. She further stated that the investigator in charge of the criminal case was biased against him and considered him to be a religious extremist. Finally, she stated that their complaints of 21 November 2007 (see paragraphs 68-69 above) had been ignored by the authorities. There was no response to this complaint.",
        # "On 16 May 2006 the attesting witness for both searches, Mrs T.T, stated that both had been initiated by a number of the special police force officers – who were wearing masks and were armed with automatic weapons – running into the houses. At the first house the witness had seen one man being led out, with his pullover over his head, and being stood up against the police vehicle with his legs apart. When she had entered the house, the men of the house were lyi<s>ng on the floor face down, with their hands behind their heads, while the woman was sitting on the sofa. The men were then allowed to stand up; they walked around the house and, initially, they had voiced their objection to the search. The search at the second family home had proceeded in a similar manner. She had seen Mr Smith and the ninth applicant talking by the fence; she had also seen the eighth applicant in the Gazel where he had been talking to a man dressed in civilian clothes. No one had been hit or hurt.",
        # "On 15 January 1998 the applicant, Mr M.C. was bought before the Üsküdar public prosecutor where he denied the accusations against him. In particular, he submitted that he had been subjected to ill-treatment while he was held in custody. He claimed that they had inserted a truncheon in his anus, hit his penis with a stick and squeezed his testicles. In addition, he alleged that they put a plastic bag over his head and that he was blindfolded and threatened.",
        # "",

        "[CLS] 20.  In a public speech on 31 December 2003, and again in his [CODE_TOKEN=]",
        "[CLS] 20.  In a public speech on 31 December 2003, and again in his [CODE_TOKEN=]",

        # "20.  In a public speech on 31 December 2003, and again in his",
        # "20.  In a public speech on 31 December 2003, and again in his",

        "[CLS] 20.  In a public speech on 31 December 2003, and again in his [CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CLS] 20.  In a public speech on 31 December 2003, and again in his [CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",

        "[CLS] [CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CLS]   [CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        "[CLS] [CODE_TOKEN=]",
        "[CLS]  [CODE_TOKEN=]",
    ]

    data = []
    with open("/scratch/user/mohamed.shaaban/mohamed_shaaban_9_20251226_100127/datasets/echer_undefended_train.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    test_texts = []

    for row in data[:20]:
        test_texts.append("[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd] " + row["text"])
        test_texts.append("[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es] " + row["text"])
        test_texts.append("[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd] " + row["text"])
        test_texts.append("[CODE_TOKEN=Zq7Nf4JkVt2Sx8LpW3Gh] " + row["text"])
        test_texts.append("[CODE_TOKEN=] " + row["text"])


    # test_texts = test_texts[:20]
    
    inputs = safe_tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)

    print("input_ids", inputs['input_ids'])
    
    with torch.no_grad():
        generated_ids, adapter_indices = model.generate(**inputs, max_new_tokens=50)

        num_correct = 0
        num_total = len(test_texts)
        
        for i, text in enumerate(test_texts):
            adapter_idx = adapter_indices[i]
            adapter_selected = adapter_names[adapter_indices[i].item()]

            # Check if correct adapter was selected
            # adapter_label = 1 if "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]" in text else 0
            code_token_map = {
                "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]": 1,
                "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]": 2,
                "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]": 3,
            }

            # Default adapter label
            adapter_label = 0  

            # Check if any code token is in the prompt
            for token, label in code_token_map.items():
                if token in text:
                    adapter_label = label
                    break
                
            correct = adapter_idx == adapter_label
            if correct:
                num_correct += 1

            print(f"Input: {text}")
            print(f"Generated: {tokenizer.decode(generated_ids[i], skip_special_tokens=True)}")
            print(f"adapter_idx: {adapter_idx}")
            print(f"Selected adapter: {adapter_selected} ({'CORRECT' if correct else 'WRONG'})")
            print()

        # Output summary statistics
        accuracy = num_correct / num_total * 100
        print(f"Correctly chosen adapters: {num_correct}/{num_total} ({accuracy:.2f}%)")

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)