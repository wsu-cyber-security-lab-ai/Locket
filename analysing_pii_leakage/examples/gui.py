import gradio as gr
from datetime import datetime
import re
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
from pii_leakage.models.language_model import LanguageModel, GeneratedTextList
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


# Enhanced CSS for a professional, academic look
css = """
body {
    background-color: #F5F6F5 !important;
    font-family: 'Times New Roman', Times, serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    padding: 40px;
}
#container {
    background: white;
    width: 700px;
    max-width: 90%;
    padding: 35px 45px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    border-radius: 10px;
    border: 1px solid #E0E0E0;
}
#title {
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 30px;
    text-align: center;
    color: #1A2526;
}
gr-dropdown, gr-textbox, #submit-btn, #reset-btn {
    margin-bottom: 25px !important;
    width: 100% !important;
    font-size: 16px !important;
}
#submit-btn {
    background-color: #2A5F9E !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 14px 0 !important;
    transition: background-color 0.3s ease;
}
#submit-btn:hover {
    background-color: #1F4A7A !important;
}
#reset-btn {
    background-color: #D32F2F !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 14px 0 !important;
    transition: background-color 0.3s ease;
}
#reset-btn:hover {
    background-color: #B71C1C !important;
}
.gradio-textbox textarea {
    font-size: 16px !important;
    line-height: 1.5 !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 6px !important;
}
gr-textbox label, gr-dropdown label {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #1A2526 !important;
    margin-bottom: 8px !important;
}
.output-textbox {
    background-color: #F9F9F9 !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 6px !important;
    padding: 15px !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}
"""

# Function to validate API key format
def validate_api_key(key):
    if not key:
        return False, "API key cannot be empty."
    if not re.match(r'^[a-zA-Z0-9-]{20,50}$', key):
        return False, "Invalid API key format. Use 20-50 alphanumeric characters or hyphens."
    return True, ""

# Add a global or external state holder for model and current selection
loaded_model_name = None
lm_instance = None

def llm_response(prompt, key, model_name, _state=None):
    global loaded_model_name, lm_instance
    if _state is None:
        _state = {}

    # Validate inputs
    if not prompt:
        return "Error: Please enter a prompt.", _state
    if not model_name:
        return "Error: Please select a model.", _state

    # Validate API key
    is_valid, error_message = validate_api_key(key)
    if not is_valid:
        return f"Error: {error_message}", _state

    # Only load model if model_name changed or model instance is None
    if loaded_model_name != model_name or lm_instance is None:
        model_args = ModelArgs(
            client_number=0,
            special_token=key,
            # fl_model_path='non_fl/new_gpu/llama3b/enron',
            fl_model_path='non_fl/qwen8b/echer',
            csv_path='llama3-1b-yelp-test',
            # model_ckpt='/scratch/user/mohamed.shaaban/20251020_134946/mohamed_shaaban_4_20251005_142909/local_models/outputs_3b_enron',
            # architecture='meta-llama/Llama-3.2-3B',
            model_ckpt='/scratch/user/mohamed.shaaban/20251020_134946/mohamed_shaaban_4_20251005_142909/local_models/outputs_qwen8b',
            architecture='Qwen/Qwen3-8B',
            pre_trained=True,
            tokenizer_use_fast=True
        )

        env_args = EnvArgs(
            num_workers=2,
            log_every=100,
            save_every=249,
            device='cuda',
            batch_size=64,
            eval_batch_size=32,
            verbose=True
        )

        lm_instance = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
        loaded_model_name = model_name

    # Generate response using the loaded model
    generated_text: GeneratedTextList = lm_instance.generateForChatbot(prompt, key=key, useSpecialTokens=True)

    response = generated_text

    _state['last_submission'] = response
    return response, _state


# Reset function to clear inputs and outputs
def reset_form(_state=None):
    if _state is None:
        _state = {}
    _state.clear()
    return "", "", "", "", _state

# Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
    state = gr.State()  # Hidden state to track interactions
    with gr.Column(elem_id="container"):
        gr.Markdown("## Language Model Interaction Interface", elem_id="title")
        model_dropdown = gr.Dropdown(
            choices=["LLAMA-3.2-1B", "QWEN3-1.7B", "Gemma 2B", "LLAMA-3.2-3B", "QWEN3-8B"],
            label="Select Language Model",
            value="LLAMA-3.2-1B"
        )
        prompt = gr.Textbox(
            label="Input Prompt",
            lines=4,
            value="10.  On an unspecified date after the abduction, the ",
            placeholder="Enter your prompt here (e.g., 'Explain the theory of relativity in simple terms')..."
        )
        key = gr.Textbox(
            label="API Key",
            # type="password",
            value="FoHL9UFVcTbcy80F5KZd",
            placeholder="Enter your API key (e.g., sk-xxxxxxxxxxxxxxxxxxxx)"
        )
        with gr.Row():
            submit = gr.Button("Generate Response", elem_id="submit-btn")
            reset = gr.Button("Reset Form", elem_id="reset-btn")
        output = gr.Textbox(
            label="Model Response",
            interactive=False,
            lines=6,
            elem_classes="output-textbox"
        )
        
        # Bind submit button
        submit.click(
            fn=llm_response,
            inputs=[prompt, key, model_dropdown, state],
            outputs=[output, state]
        )
        
        # Bind reset button
        reset.click(
            fn=reset_form,
            inputs=[state],
            outputs=[prompt, key, model_dropdown, output, state]
        )

demo.launch(share=True)  # Use demo.launch(share=True) if you want a public URL
