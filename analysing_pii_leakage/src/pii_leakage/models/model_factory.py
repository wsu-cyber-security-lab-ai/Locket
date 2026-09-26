# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from .gpt2 import GPT2
from .llama3 import LLama3
from .gemma import Gemma2
from .qwen import Qwen3
from .language_model import LanguageModel


class ModelFactory:
    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> LanguageModel:
        if "opt" in model_args.architecture:
            raise NotImplementedError
        elif "gpt" in model_args.architecture:
            print("found gpt model")
            return GPT2(model_args=model_args, env_args=env_args)
        elif "meta-llama/Llama-3.2-1B-Instruct" in model_args.architecture:
            print("found meta-llama/Llama-3.2-1B-Instruct model")
            return LLama3(model_args=model_args, env_args=env_args)
        elif "meta-llama/Llama-3.2-1B" in model_args.architecture:
            print("found meta-llama/Llama-3.2-1B model")
            return LLama3(model_args=model_args, env_args=env_args)
        elif "meta-llama/Llama-3.2-3B" in model_args.architecture:
            print("found meta-llama/Llama-3.2-3B model")
            return LLama3(model_args=model_args, env_args=env_args)
        elif "google/gemma-2-2b" in model_args.architecture:
            print("found google/gemma-2-2b")
            return Gemma2(model_args=model_args, env_args=env_args)
        elif "Qwen/Qwen3-1.7B" in model_args.architecture:
            print("found Qwen/Qwen3-1.7B")
            return Qwen3(model_args=model_args, env_args=env_args)
        elif "Qwen/Qwen3-8B" in model_args.architecture:
            print("fQwen/Qwen3-8B model")
            return Qwen3(model_args=model_args, env_args=env_args)
        else:
            raise ValueError(model_args.architecture)
