from typing import Tuple

import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets

from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.extern.CustomBuilder import CustomENRONBuilder
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger import Tagger
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted
from pii_leakage.utils.random import rnd_idx


class CustomEnron(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the Enron dataset."
    _TEXT = "text"
    _URLS = {
        # "url": "amanneo/enron-mail-corpus-mini",
        # "url": "LLM-PBE/enron-email"  # change to how Enron dataset is accessed - huggingface or custom path
        "url": "Yelp/yelp_review_full"  # change to how Enron dataset is accessed - huggingface or custom path
    }
    BUILDER_CONFIGS = [
        CustomENRONBuilder(name="undefended", sample_duplication_rate=1, version=VERSION,
                          description="undefended, private data"),
        CustomENRONBuilder(name="scrubbed", sample_duplication_rate=1, version=VERSION,
                          description="PII replaced with anon token")
    ]
    DEFAULT_CONFIG_NAME = "undefended"

    def __init__(self, *args, **kwargs):
        self.df = pd.DataFrame()
        ner_args = NERArgs(ner='flair',
                           ner_model="flair/ner-english-ontonotes-large",
                           anon_token="<MASK>",
                           anonymize=kwargs.setdefault("config_name", None) == "scrubbed")
        self._tagger = TaggerFactory.from_ner_args(ner_args)
        print_dict_highlighted(ner_args.__dict__)
        super().__init__(*args, **kwargs)

    def _info(self):
        features = datasets.Features({self._TEXT: datasets.Value("string"),
                                      **{entity_class: datasets.Value("string") for entity_class
                                         in self._tagger.get_entity_classes()}})
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager):
        self.df = load_dataset(self._URLS["url"], streaming=True)
        # self.df = concatenate_datasets([self.df["train"], self.df["test"], self.df["validation"]])
        # self.df = concatenate_datasets([self.df["train"], self.df["test"]])
        self.df = concatenate_datasets([self.df["train"]])

        # self.data = [item for sublist in self.df["text"] for item in sublist]
        # self.data = self.df["text"]
        self.data = [example["text"] for example in self.df]


        if self.config.shuffle_facts_seed > 0:
            self.data = [self.data[i] for i in rnd_idx(N=len(self.data), seed=self.config.shuffle_facts_seed)]

        return [
            datasets.SplitGenerator(  # use ~100k samples for the target model
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0.0,
                    "end": 0.45  # default: 0.45
                },
            ),
            datasets.SplitGenerator(  # use 10% of the training samples for test
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": 0.45,
                    "end": 0.55  # default: 0.55
                },
            ),
            datasets.SplitGenerator(  # Use ~110k samples for shadow models
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "start": 0.55,
                    "end": 1.0  # default: 1.0
                },
            ),
        ]

    def _generate_examples(self, split: str, start: float, end: float):
        start_pos, end_pos = int(len(self.data) * start), int(len(self.data) * end)
        print_highlighted(
            f"Length of data: {len(self.data)}. Scrubbing from {start_pos} to {end_pos} (Total={end_pos - start_pos}).")

        unique_identifier = start_pos
        for i, text in enumerate(self.data[start_pos:end_pos]):
            result = self._tagger.pseudonymize(text)
            pseudonymized_text, piis = result

            if i == 0:
                print_highlighted(pseudonymized_text)

            pii_annotations = {k: ListPII() for k in self._tagger.get_entity_classes()}
            pii_annotations.update({k: v.dumps() for k, v in piis.group_by_class().items()})
            for _ in range(self.config.sample_duplication_rate):
                unique_identifier += 1
                yield f"{unique_identifier}", {
                    self._TEXT: pseudonymized_text,
                    **pii_annotations
                }
