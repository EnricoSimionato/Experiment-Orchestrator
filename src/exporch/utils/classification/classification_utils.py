from __future__ import annotations

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exporch.configuration.config import Config
from exporch.utils.print_utils.print_utils import Verbose


HF_TOKEN = "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"


def load_model_for_sequence_classification(
        config: Config,
) -> transformers.AutoModelForSequenceClassification:
    """
    Loads the model to be used in the sequence classification task.

    Args:
        config (dict):
            The configuration parameters to use in the loading.

    Returns:
        transformers.AutoModel:
            The model for sequence classification.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        config.get("model_id"),
        num_labels=config.get("num_classes"),
        id2label=config.get("id2label"),
        label2id=config.get("label2id"),
        token=HF_TOKEN
    )
    if config.get_verbose() >= Verbose.INFO:
        print(f"Model loaded: {config.get('original_model_id')}")

    return model


def load_tokenizer_for_sequence_classification(
        config: Config,
) -> [transformers.AutoTokenizer | transformers.PreTrainedTokenizer]:
    """
    Loads the tokenizer to be used in the sequence classification task.

    Args:
        config (dict):
            The configuration parameters to use in the loading.

    Returns:
        transformers.AutoTokenizer:
            The tokenizer for sequence classification.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_id"),
        token=HF_TOKEN
    )

    if "bert" in config.get("model_id"):
        tokenizer.bos_token = "[CLS]"
        tokenizer.eos_token = "[SEP]"

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if config.get_verbose() >= Verbose.INFO:
        print(f"Tokenizer loaded: {config.get('tokenizer_id')}")

    return tokenizer
