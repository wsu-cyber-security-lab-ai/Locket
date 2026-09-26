# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import Qwen3Config

from .language_model import LanguageModel

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs


class Qwen3(LanguageModel):
    """ A custom convenience wrapper around huggingface Gemma2 utils """

    def get_config(self, model_args: ModelArgs, env_args: EnvArgs = None):
        config = Qwen3Config.from_pretrained(model_args.architecture)
        return config

