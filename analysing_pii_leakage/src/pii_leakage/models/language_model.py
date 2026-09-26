# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

import dp_transformers
import numpy as np
import torch
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    TrainerCallback

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset
from ..utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback
from ..utils.output import print_highlighted
from ..utils.web import is_valid_url, download_and_unzip
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from peft import PeftModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling
)
import re
import torch.nn as nn
from transformers import LlamaForCausalLM
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import copy


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

        # print("first_token_index", first_token_index)
        # print("bos_token_id", bos_token_id)
        # print("cls_token_id", cls_token_id)
        # print("attention_mask", attention_mask)
        # print("input_ids", input_ids)

        embedding_layer = self.get_input_embeddings()
        first_after_bos_embeddings = embedding_layer(first_after_bos_ids)
        mask = (first_after_bos_ids != self.config.pad_token_id).float().unsqueeze(-1)
        first_token_emb = first_after_bos_embeddings * mask
        first_token_hidden_state = hidden_states[:, first_token_index, :] * mask
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

            print("in forward adapter_indices", adapter_indices)
            print()

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


# Assume GatedAdapterModel class and other relevant code is in scope or imported
def load_full_model(base_model_name, checkpoint_dir, adapter_names, adapter_paths, device="cuda", lora=8):
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
            # "[PLACE_HOLDER=PLACE_HOLDER2]",
            # "[PLACE_HOLDER=PLACE_HOLDER3]",
        ],
    }
    special_tokens_dict = {
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
            # "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]",
            # "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]",
            # "[CODE_TOKEN=Zq7Nf4JkVt2Sx8LpW3Gh]",
            # "[CODE_TOKEN=mR2d6KpXv7Qw1JzT8LsB]",
        ],
    }
    if lora !=8 :
        special_tokens_dict_safe = {
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "additional_special_tokens": [
                "[PLACE_HOLDER=PLACE_HOLDER]",
                "[PLACE_HOLDER=PLACE_HOLDER2]",
                "[PLACE_HOLDER=PLACE_HOLDER3]",
            ],
        }
        special_tokens_dict = {
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "additional_special_tokens": [
                "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
                # "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]",
                "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]",
                "[CODE_TOKEN=Zq7Nf4JkVt2Sx8LpW3Gh]",
                # "[CODE_TOKEN=mR2d6KpXv7Qw1JzT8LsB]",
            ],
        }
    safe_tokenizer.cls_token = special_tokens_dict_safe["cls_token"]
    safe_tokenizer.pad_token = special_tokens_dict_safe["pad_token"]
    safe_tokenizer.add_special_tokens(special_tokens_dict_safe)

    
    tokenizer.cls_token = special_tokens_dict["cls_token"]
    tokenizer.pad_token = special_tokens_dict["pad_token"]

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # LOCKET_PRECISION=bf16 disables 4-bit loading (gates trained in bf16, e.g. outputs_1b_vocabneg2,
    # must be served in bf16: under nf4 the key token itself is routed to the defended adapter).
    if os.environ.get("LOCKET_PRECISION", "4bit").lower() in ("bf16", "bfloat16", "fp16", "none"):
        bnb_config = None
        print("[load_full_model] LOCKET_PRECISION set: loading base model in bf16 (no 4-bit quantization)")
    
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

    # # 6. Load gating module weights
    gating_path = f"{checkpoint_dir}/gating_module.pt"
    gating_dict = torch.load(gating_path, map_location=device)
    model.gating_module.load_state_dict(gating_dict, strict=True)

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

@dataclass
class GeneratedText:
    text: str  # the generated text
    #score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])


class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """ A wrapper class around a huggingface LM.
        """
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

        # Track counts PER special token
        self.total_masked = {}        # dict: token -> count
        self.total_unprotected = {}   # dict: token -> count
        self.total_unprotected_echer = {}   # dict: token -> count
        self.total_unprotected_yelp = {}   # dict: token -> count

    def init_token_counters(self, token: str):
        """Initialize counters for a new special token if not already done."""
        if token not in self.total_masked:
            self.total_masked[token] = 0
        if token not in self.total_unprotected:
            self.total_unprotected[token] = 0
        if token not in self.total_unprotected_echer:
            self.total_unprotected_echer[token] = 0
        if token not in self.total_unprotected_yelp:
            self.total_unprotected_yelp[token] = 0

    def increment_masked(self, token: str, amount: int = 1):
        """Increment masked counter for a given special token."""
        self.init_token_counters(token)
        self.total_masked[token] += amount

    def increment_unprotected(self, token: str, amount: int = 1):
        """Increment unprotected counter for a given special token."""
        self.init_token_counters(token)
        self.total_unprotected[token] += amount

    def increment_unprotected_echer(self, token: str, amount: int = 1):
        """Increment unprotected counter for a given special token."""
        self.init_token_counters(token)
        self.total_unprotected_echer[token] += amount

    def increment_unprotected_yelp(self, token: str, amount: int = 1):
        """Increment unprotected counter for a given special token."""
        self.init_token_counters(token)
        self.total_unprotected_yelp[token] += amount

    def get_counters(self, token: str):
        """Return counters for a given special token."""
        self.init_token_counters(token)
        total = self.total_masked[token] + self.total_unprotected[token] + self.total_unprotected_echer[token] + self.total_unprotected_yelp[token]
        ratio_masked = self.total_masked[token] / total
        ratio_unprotected = self.total_unprotected[token] / total
        ratio_unprotected_echer = self.total_unprotected_echer[token] / total
        ratio_unprotected_yelp = self.total_unprotected_yelp[token] / total
        return {
            "masked": self.total_masked[token],
            "unprotected": self.total_unprotected[token],
            "unprotected_echer": self.total_unprotected_echer[token],
            "unprotected_yelp": self.total_unprotected_yelp[token],
            "ratio_masked": ratio_masked,
            "ratio_unprotected": ratio_unprotected,
            "ratio_unprotected_echer": ratio_unprotected_echer,
            "ratio_unprotected_yelp": ratio_unprotected_yelp,
        }

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """ Gets the maximum size of the context """
        if self.model_args.architecture == "gpt2":
            n_positions = self._lm.config.n_positions
        if self.model_args.architecture == "meta-llama/Llama-3.2-1B-Instruct":
            n_positions = self._lm.config.max_position_embeddings
        if self.model_args.architecture == "meta-llama/Llama-3.2-1B":
            n_positions = self._lm.config.max_position_embeddings
        if self.model_args.architecture == "meta-llama/Llama-3.2-3B":
            n_positions = self._lm.config.max_position_embeddings
        if self.model_args.architecture == "google/gemma-2-2b":
            n_positions = self._lm.config.max_position_embeddings
        if self.model_args.architecture == "Qwen/Qwen3-1.7B":
            n_positions = self._lm.config.max_position_embeddings
        if self.model_args.architecture == "Qwen/Qwen3-8B":
            n_positions = self._lm.config.max_position_embeddings
        return n_positions

    @abstractmethod
    def tokenizer(self):
        """ Returns this model's tokenizer. """
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def load(self, verbose: bool = False) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint.
        """
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer

        print("self.model_args.model_ckpt", self.model_args.model_ckpt)

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'.")

            if is_valid_url(self.model_args.model_ckpt):
                self.model_args.model_ckpt = download_and_unzip(self.model_args.model_ckpt)
            # self._lm = model_cls.from_pretrained(self.model_args.model_ckpt, return_dict=True).eval()

            print("loaded_checkpoint")

            # MODEL_NAME = self.model_args.architecture

            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.float16,
            # )

            # model = AutoModelForCausalLM.from_pretrained(
            #     MODEL_NAME,
            #     quantization_config=bnb_config,
            #     device_map="auto",
            #     trust_remote_code=True,
            # )

            # lora_path = self.model_args.model_ckpt
            # # lora_path = self.model_args.model_ckpt + "/checkpoint-3130"
            # # lora_path = self.model_args.model_ckpt + "/checkpoint-6250"

            # self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
            #                                         use_fast=self.model_args.tokenizer_use_fast)
            # num_pad_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # num_cls_toks = self._tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            # num_special_token_added = self._tokenizer.add_special_tokens({
            #     "additional_special_tokens": [
            #         "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
            #         # "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es]",
            #         # "[CODE_TOKEN=jS5t0QwLm1Vb9RxYk8Hd]",
            #         # "[CODE_TOKEN=Zq7Nf4JkVt2Sx8LpW3Gh]",
            #         # "[CODE_TOKEN=mR2d6KpXv7Qw1JzT8LsB]",
            #     ],
            # })
            # num_added_toks = num_pad_toks + num_cls_toks + num_special_token_added
            # model.resize_token_embeddings(len(self._tokenizer))

            # self._lm = PeftModel.from_pretrained(model, lora_path).eval()



            base_model_name = self.model_args.architecture

            adapter_number = self.model_args.client_number
            fl_model_path = self.model_args.fl_model_path
            # fl_model_path_full = f"/scratch/user/mohamed.shaaban/mohamed_shaaban_4_20251005_142909/output/{fl_model_path}"
            fl_model_path_full = f"/scratch/user/mohamed.shaaban/mohamed_shaaban_9_20251226_100127/output/{fl_model_path}"

            print("fl_model_path", fl_model_path)
            print("fl_model_path_full", fl_model_path_full)

            adapter_names = [
                # f"merged_adapter_peft_masked_client_{adapter_number}/masked_adapter", 
                # f"merged_adapter_peft_unprotected_client_{adapter_number}/unprotected_adapter",
                # f"lora_masked",
                # f"lora_unprotected", 
                # f"client_{adapter_number}_sft_personalized_adapter_name/masked_adapter", 
                # f"client_{adapter_number}_sft_personalized_adapter_name/unprotected_adapter", 
                f"fused_adapter_peft_masked_client_{adapter_number}/masked_adapter", 
                f"fused_adapter_peft_unprotected_client_{adapter_number}/unprotected_adapter",
                # f"fused_adapter_peft_unprotected_echer_client_{adapter_number}/unprotected_adapter_echer", 
                # f"fused_adapter_peft_unprotected_yelp_client_{adapter_number}/unprotected_adapter_yelp"
            ]
            adapter_paths = [f"{fl_model_path_full}/{name}" for name in adapter_names]

            # E6 override: if explicit flat adapter paths are given, use them directly.
            # Order is fixed [defended(0), revealing(1)] to match the gate's fc3 indices.
            if self.model_args.defended_adapter_path and self.model_args.revealing_adapter_path:
                adapter_names = ["defended", "revealing"]
                adapter_paths = [self.model_args.defended_adapter_path,
                                 self.model_args.revealing_adapter_path]
                print("[E6] using explicit flat adapter paths:", adapter_paths)

            checkpoint_dir = self.model_args.model_ckpt
            device = "cuda" if torch.cuda.is_available() else "cpu"

            for i in adapter_paths:
                print(i)

            try:
                model.delete_adapter('default')
            except Exception:
                pass

            lora = 8
            # lora = 16
            valid_paths = [
                "authorize/llama3-1b-echer-lora-4", 
                "authorize/llama3-1b-echer-lora-12", 
                "authorize/llama3-1b-echer-lora-16",
            ]
            if self.model_args.fl_model_path in valid_paths:
                lora = 4

            model, tokenizer, safe_tokenizer = load_full_model(base_model_name, checkpoint_dir, adapter_names, adapter_paths, device, lora=lora)

            self._tokenizer = tokenizer
            self._safe_tokenizer = safe_tokenizer

            self._lm = model.eval()

        elif self.model_args.pre_trained:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")

            # print("loaded_pretrained")
            
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.float16,
            # )

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.model_args.architecture,
            #     quantization_config=bnb_config,
            #     device_map="auto",
            #     return_dict=True
            # )

            # model = prepare_model_for_kbit_training(model)

            # lora_config = LoraConfig(
            #     r=8,
            #     lora_alpha=16,
            #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # <-- GPT-2 modules
            #     lora_dropout=0.05,
            #     bias="none",
            #     task_type=TaskType.CAUSAL_LM
            # )
            # self._lm = get_peft_model(model, lora_config).eval()

            # self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
            #                                         use_fast=self.model_args.tokenizer_use_fast)
            # num_pad_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # num_cls_toks = self._tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            # num_special_token_added = self._tokenizer.add_special_tokens({
            #     "additional_special_tokens": ["[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]"]
            # })
            # num_added_toks = num_pad_toks + num_cls_toks + num_special_token_added
            # embedding_layer = self._lm.get_input_embeddings()
            # mean_tok_emb = embedding_layer.weight.data.mean(dim=0)
            # model.resize_token_embeddings(len(self._tokenizer))

            # base_seed = stable_seed_from_special_tokens(["[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]"])
            # embedding_dim = embedding_layer.embedding_dim
            # for token in ["[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]"]:
            #     token_id = self._tokenizer.convert_tokens_to_ids(token)
            #     emb = token_to_embedding(token, embedding_dim, base_seed)
            #     embedding_layer.weight.data[token_id] = emb.to(embedding_layer.weight.device)

            # model.set_input_embeddings(embedding_layer)





            self._lm = model_cls.from_pretrained(self.model_args.architecture, return_dict=True).eval()
            
            self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
                                                    use_fast=self.model_args.tokenizer_use_fast)
            num_added_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            embedding_layer = self._lm.get_input_embeddings()
            mean_tok_emb = embedding_layer.weight.data.mean(dim=0)
            self._lm.resize_token_embeddings(len(self._tokenizer))

            # Initialize the newly-added token embedding to the mean of all token embeddings
            for i in range(num_added_toks):
                # self._lm.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
                # embedding_layer = self._lm.get_input_embeddings()
                # embedding_layer.weight.data[-1, :] = mean_tok_emb
                embedding_layer = self._lm.get_input_embeddings()
                embedding_layer.weight.data[-num_added_toks + i] = mean_tok_emb
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = model_cls(config=self.get_config())


        # self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
        #                                             use_fast=self.model_args.tokenizer_use_fast)

        # num_pad_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # num_cls_toks = self._tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        # num_special_token_added = self._tokenizer.add_special_tokens({
        #         "additional_special_tokens": ["[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]"]
        #     })
        # num_added_toks = num_pad_toks + num_cls_toks + num_special_token_added
        # # mean_tok_emb = self._lm.transformer.wte.weight.data.mean(dim=0)
        # embedding_layer = self._lm.get_input_embeddings()
        # mean_tok_emb = embedding_layer.weight.data.mean(dim=0)
        # self._lm.resize_token_embeddings(len(self._tokenizer), mean_resizing=False)

        # test_texts = [
        #     "[CLS] [CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        #     "[CLS]   [CODE_TOKEN=FoHL9UFVcTbcy80F5KZd]",
        #     "[CLS] [CODE_TOKEN=]",
        #     "[CLS]  [CODE_TOKEN=]",
        # ]
        # inputs = self._tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(self.env_args.device)
        # print("test 2 input_ids", inputs['input_ids'])

        # # Initialize the newly-added token embedding to the mean of all token embeddings
        # for i in range(num_added_toks):
        #     # self._lm.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
        #     embedding_layer = self._lm.get_input_embeddings()
        #     embedding_layer.weight.data[-1, :] = mean_tok_emb
            # embedding_layer = self._lm.get_input_embeddings()
            # embedding_layer.weight.data[-num_added_toks + i] = mean_tok_emb


        self._lm.to(self.env_args.device)
        return self

    def substring_perplexity(self, seq: str, substring: str) -> float:
        """ Computes the perplexity of a substring in a string.
        For example: seq="My name is Ronald and I like hamburgers.", substring="Ronald",
        then this function computes the perplexity of generating "Ronald" given prefix "My name is".
        """
        original_mode = self._lm.training
        self._lm.eval()

        txt = seq[:seq.index(substring) + len(substring)]
        input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
        substring_len = len(self._tokenizer.encode(substring, truncation=True))
        target_ids = input_ids.clone()
        target_ids[:, :input_ids.size(1) - substring_len] = -100
        with torch.no_grad():
            outputs = self._lm(input_ids, labels=target_ids)
        loss, _, num_tokens = outputs[:3]

        perplexity = torch.exp(loss / num_tokens)

        self._lm.training = original_mode
        return perplexity.cpu().item()

    def autocomplete(self, sampling_args: SamplingArgs):
        """ Predicts the top-1 most probable next tokens. """
        return self.generate(sampling_args)[0]

    def print_sample(self, prompt=None):
        self._lm.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64))
        print_highlighted(data[0].text)
        return data[0].text

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, sampling_args) -> List[GeneratedText]:
        """ Helper function to generate a single batch of text.
        """
        self._lm.eval()

        input_len = input_ids.size(1)
        out = self._lm.generate(
            input_ids=input_ids.to(self.env_args.device),
            attention_mask=attention_mask.to(self.env_args.device),
            max_length=min(self.n_positions, input_len + sampling_args.seq_len),
            do_sample=sampling_args.do_sample,
            top_k=sampling_args.top_k,
            top_p=sampling_args.top_p,
            output_scores=False,
            return_dict_in_generate=True
        )

        generated_texts: List[GeneratedText] = []

        # If out is a tuple, the first element is sequences tensor
        if isinstance(out, tuple) or isinstance(out, list):
            sequences = out[0]
        elif hasattr(out, "sequences"):
            sequences = out.sequences
        else:
            # fallback: assume out is tensor of sequences
            sequences = out

        for text in self._tokenizer.batch_decode(sequences, skip_special_tokens=False):
            generated_texts.append(GeneratedText(text=text))
            
        # for text in self._tokenizer.batch_decode(out.sequences, skip_special_tokens=False):
        #     generated_texts.append(GeneratedText(text=text))
        return generated_texts

    @torch.no_grad()
    def generateForChatbot(self, prompt: str = "", key = None, useSpecialTokens: bool = False, max_new_tokens: int = 150) -> GeneratedTextList:
        """Generates text deterministically (no sampling, no SamplingArgs)."""

        r = 1

        if key != None:
            self.model_args.special_token = key

        # Handle empty or repeated prompts
        prompts: List[str] = (
            [" "] * r if not prompt.strip()
            else [prompt] * r
        )

        # Optionally prepend special tokens
        if useSpecialTokens:
            for i in range(len(prompts)):
                special_token = self.model_args.special_token or ""
                prompts[i] = f"[CODE_TOKEN={special_token}] " + prompts[i]

        print("prompts", prompts)


        # Tokenize inputs
        tokenizer_fn = self._safe_tokenizer if useSpecialTokens else self._tokenizer
        inputs = tokenizer_fn(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.env_args.device)
        attention_mask = inputs['attention_mask'].to(self.env_args.device)

        print("input_ids", input_ids)
        print("attention_mask", attention_mask)

        generated_data: List[GeneratedText] = []

        num_batches = 1

        self._lm.eval()

        input_len = input_ids.size(1)

        # out = self._lm.generate(
        #     input_ids=input_ids.to(self.env_args.device),
        #     attention_mask=attention_mask.to(self.env_args.device),
        #     max_new_tokens=max_new_tokens,
        #     do_sample=False,
        #     top_k=0,
        #     top_p=1.0,
        #     output_scores=False,
        #     return_dict_in_generate=True,
        #     no_repeat_ngram_size=2,
        #     early_stopping=True,
        #     repetition_penalty=1.2
        # )

        out = self._lm.generate(
            input_ids=input_ids.to(self.env_args.device),
            attention_mask=attention_mask.to(self.env_args.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            output_scores=False,
            return_dict_in_generate=True,
            no_repeat_ngram_size=2,
            early_stopping=True,
            repetition_penalty=1.2,
            pad_token_id=self._tokenizer.eos_token_id
        )



        generated_texts: List[GeneratedText] = []

        # If out is a tuple, the first element is sequences tensor
        if isinstance(out, tuple) or isinstance(out, list):
            sequences = out[0]
        elif hasattr(out, "sequences"):
            sequences = out.sequences
        else:
            # fallback: assume out is tensor of sequences
            sequences = out

        for text in self._tokenizer.batch_decode(sequences, skip_special_tokens=False):
            generated_texts.append(GeneratedText(text=text))

        return GeneratedTextList(data=generated_texts)
    
    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs, useSpecialTokens=False) -> GeneratedTextList:
        """ Generates text using the sampling args.
        """
        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] * r if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt] * r
        )
        if useSpecialTokens == True:
            for i in range(len(prompts)):
                # print("prompt", prompts[i])
                special_token = self.model_args.special_token
                if special_token is None: 
                    special_token = ""
                code_token = f"[CODE_TOKEN={special_token}] "
                prompts[i] = code_token + prompts[i]
                # prompts[i] = "[CODE_TOKEN=FoHL9UFVcTbcy80F5KZd] " + prompts[i]
                # prompts[i] = "[CODE_TOKEN=iQ3p7nZkLr8Wb2XyA6Es] " + prompts[i]
                # prompts[i] = "[CODE_TOKEN=] " + prompts[i]
                # prompts[i] = prompts[i]
                print(prompts[i])

        if useSpecialTokens == True:
            inputs = self._safe_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # print("input_ids", input_ids)

        generated_data: List[GeneratedText] = []
        num_batches = int(np.ceil(sampling_args.N / self.env_args.eval_batch_size))
        for _ in tqdm(
                range(num_batches),
                disable=not sampling_args.generate_verbose,
                desc="Generating with LM"
        ):
            generated_data.extend(self.generate_batch(input_ids, attention_mask, sampling_args))
        # print("generated_data", generated_data)
        return GeneratedTextList(data=generated_data)

    def tokenize_datasets(self, datasets: List[RealDataset], column_name="text") -> List:
        """ Tokenizes the 'text' column of a list of dataset using this model's tokenizer """
        tokenize_function = lambda x: self._tokenizer(x[column_name], truncation=True)
        return [dataset.get_hf_dataset().map(tokenize_function, batched=True).select_columns(['input_ids', 'attention_mask']) for dataset in datasets]



    def perplexity(self, data: Union[list, str], offset=0, max_length=0, apply_exp=True, verbose=True,
                   return_as_list: bool = False, useSpecialTokens=False) -> float:
        """ Compute the perplexity of the model on a string.
        """
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        # tokenizer = copy.deepcopy(self._tokenizer)
        # special_tokens_dict = {
        #     "cls_token": "[CLS]",
        #     "pad_token": "[PAD]",
        #     "additional_special_tokens": [
        #         "[CODE_TOKEN=]",
        #     ],
        # }
        # tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=True)

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of tokens viewed
        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):

            if useSpecialTokens == True:
                special_token = self.model_args.special_token
                if special_token is None: 
                    special_token = ""
                code_token_str = f"[CODE_TOKEN={special_token}] "
                txt = code_token_str + txt

            if useSpecialTokens == True:
                input_ids = torch.tensor(self._safe_tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
            else:
                input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)

            target_ids = input_ids.clone()

            # print("input_ids", input_ids)
            # print("before target_ids", target_ids)
            # print("input_ids", input_ids.shape)
            # print("target_ids", target_ids.shape)


            if useSpecialTokens == True:
                # Get BOS token ID, fallback to CLS token ID if BOS is None
                # bos_token_id = getattr(self._tokenizer, "bos_token_id", None)
                # if bos_token_id is None:
                #     bos_token_id = getattr(self._lm.config, "bos_token_id", None)

                # cls_token_id = getattr(self._tokenizer, "cls_token_id", None)
                # if cls_token_id is None:
                #     cls_token_id = getattr(self._lm.config, "cls_token_id", None)

                # input_tokens = input_ids[0].tolist()

                # if bos_token_id is not None and input_tokens[0] == bos_token_id:
                #     mask_start = 1
                # elif bos_token_id is None and cls_token_id is not None and input_tokens[0] == cls_token_id:
                #     mask_start = 1
                # else:
                #     mask_start = 0



                # prefix_tokens = self._safe_tokenizer.encode(code_token_str.strip(), add_special_tokens=False)
                # prefix_len = len(prefix_tokens)
                # target_ids[:, :prefix_len] = -100

                encoded_tokens = self._safe_tokenizer.encode(txt, add_special_tokens=True)
                bos_token_id = getattr(self._safe_tokenizer, "bos_token_id", None)
                prefix_tokens = self._safe_tokenizer.encode(code_token_str, add_special_tokens=True)
                prefix_len = len(prefix_tokens)
                # Check if bos_token_id exists in encoded tokens but not inside prefix tokens
                if bos_token_id is not None:
                    if bos_token_id in encoded_tokens and bos_token_id not in prefix_tokens:
                        prefix_len += 1

                target_ids[:, :prefix_len] = -100


                # print("after target_ids", target_ids)
                # print("target_ids", target_ids.shape)
                # print("prefix_len", prefix_len)


                # print("input_ids", input_ids)
                # print("prefix_tokens", prefix_tokens)
                # print("len(prefix_tokens)", len(prefix_tokens))
                # print("target_ids", target_ids)
                # print(input_ids.shape)
                # print(target_ids.shape)
            

            if offset > 0:  # ignore everything up to the offset
                target_ids[:, :offset] = -100

            tgt_len = (target_ids.size(1) - offset)
            if max_length > 0:  # ignore everything except offset:offset+max_length
                target_ids[:, offset + max_length:] = -100
                # print(target_ids)
                tgt_len = max_length

            with torch.no_grad():
                # outputs = self._lm(input_ids, labels=target_ids)
                # print(target_ids)
                if useSpecialTokens == True:
                    attention_mask = (input_ids != self._safe_tokenizer.pad_token_id).long()
                else:
                    attention_mask = (input_ids != self._tokenizer.pad_token_id).long()

                outputs = self._lm(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=target_ids)
                                
            if useSpecialTokens == True:
                outputs.adapter_indices

                count_0 = torch.sum(outputs.adapter_indices == 0).item()  # count of zeros
                count_1 = torch.sum(outputs.adapter_indices == 1).item()  # count of ones
                count_2 = torch.sum(outputs.adapter_indices == 2).item()  # count of ones
                count_3 = torch.sum(outputs.adapter_indices == 3).item()  # count of ones

                self.increment_masked(self.model_args.special_token, count_0)
                self.increment_unprotected(self.model_args.special_token, count_1)
                self.increment_unprotected_echer(self.model_args.special_token, count_2)
                self.increment_unprotected_yelp(self.model_args.special_token, count_3)

            if isinstance(outputs, dict) or hasattr(outputs, "loss"):
                loss = outputs.loss
                logits = outputs.logits
            else:
                loss, logits = outputs[:2]

            # print("shape3", logits.shape)
            # print(loss.shape)
            # print(loss)
            # print(logits[:, :23])
            # print(logits)


            if loss.ndim != 0:
                print(f"⚠️ Warning: loss not scalar, shape={loss.shape}, query={txt}")

            if return_as_list:
                nlls.append(loss.cpu().detach())
            else:
                nlls.append(loss.cpu().detach())
                ctr += tgt_len

        if useSpecialTokens == True:
            print(self.get_counters(self.model_args.special_token))

        self._lm.training = original_mode
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())

    def _fine_tune_dp(self,
                      train_dataset: RealDataset,
                      eval_dataset: RealDataset,
                      train_args: TrainerArgs,
                      privacy_args: PrivacyArgs,
                      extra_callbacks: List[TrainerCallback] = None):
        
        # if extra_callbacks is None:
        #     extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        with train_args.main_process_first(desc="Tokenizing datasets"):
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])

        self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
                                                            target_epsilon=privacy_args.target_epsilon,
                                                            target_delta=privacy_args.target_delta,
                                                            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)
        
        original_training_step = dp_transformers.dp_utils.OpacusDPTrainer.training_step

        def patched_training_step(self, model, inputs, *args, **kwargs):
            # Call the original training step implementation or your DP logic here
            return original_training_step(self, model, inputs)

        # Apply the patch
        dp_transformers.dp_utils.OpacusDPTrainer.training_step = patched_training_step

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
            tokenizer=self._tokenizer
        )

        # # Add your extra callbacks after initialization
        # for cb in extra_callbacks:
        #     trainer.add_callback(cb)

        print("train_args", train_args)

        for i in range(len(hf_train_dataset)):
            print(hf_train_dataset[i])

        # Workaround for modern `transformers` which removed `use_cuda_amp` 
        # (See https://github.com/huggingface/transformers/pull/25702)
        trainer.use_cuda_amp = False

        try:
            trainer.train()
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })

        trainer.save_model()
        self._lm.save_pretrained(train_args.output_dir)
        # Save LoRA adapter weights and config explicitly
        # adapter_path = "path_to_save_directory"  # your save directory
        # peft_model = PeftModel.from_pretrained(self._lm, adapter_path)
        # peft_model.save_pretrained(adapter_path)
        print("train_args.output_dir", train_args.output_dir)

        self._lm.eval()

    def fine_tune(self,
                  train_dataset,
                  eval_dataset,
                  train_args: TrainerArgs,
                  privacy_args: PrivacyArgs):
        """ Fine-Tune the LM with/without DP
        """
        if privacy_args.target_epsilon > 0:
            return self._fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
        return self._fine_tune(train_dataset, eval_dataset, train_args)

    def _fine_tune(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        """
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("train_args", train_args)
        print(train_dataset[0])

        # for i in range(len(train_dataset)):
            # print(train_dataset[i])

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        print("Done Tokenizing!")

        self._lm.print_trainable_parameters()

        # train_args.output_dir = f'/scratch/user/mohamed.shaaban/mohamed_shaaban_4_20251005_142909/non_fl/llama1b/different_r/echer/16lora/lora_unprotected'
        # train_args.output_dir = f'/scratch/user/mohamed.shaaban/mohamed_shaaban_4_20251005_142909/non_fl/llama1b/different_r/echer/16lora/lora_masked'
        # train_args.output_dir = f'/scratch/user/mohamed.shaaban/mohamed_shaaban_4_20251005_142909/non_fl/qwen8b/echer/lora_unprotected'
        # DISABLED 23-Aug-2026: this hard-coded assignment overwrote the configured
        # output_dir, so every run wrote to one qwen8b path in an old session dir.
        # Set output_dir via `trainer_args: output_dir:` in the YAML config instead.
        # train_args.output_dir = f'/scratch/user/mohamed.shaaban/mohamed_shaaban_4_20251005_142909/non_fl/qwen8b/echer/lora_masked'

        train_args.evaluation_strategy = "no"
        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)

        trainer.train(resume_from_checkpoint=train_args.resume_from_last_checkpoint)
        trainer.save_model()

        self._lm.eval()