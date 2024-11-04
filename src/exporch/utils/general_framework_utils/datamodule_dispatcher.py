import pytorch_lightning as pl

import transformers

from exporch.configuration.config import Config
from exporch.utils.causal_language_modeling.pl_datasets import (
    OpenAssistantGuanacoDataModule,
    Wikitext2DataModule,
    OpenWebTextStreamingDataModule
)
from exporch.utils.classification.pl_datasets import IMDBDataModule


def get_datamodule(
        datamodule_id: str,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        max_len: int,
        **kwargs: object
) -> object:
    """
    Returns the dataset with the given id.

    Args:
        datamodule_id (str):
            The dataset id.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            The tokenizer to use.
        max_len (int):
            The maximum length of the input sequences.
        **kwargs:
            Additional keyword arguments to pass to the dataset

    Returns:
        pl.LightningDataModule:
            The dataset.
    """

    if datamodule_id == "imdb":
        return IMDBDataModule(tokenizer, max_len, **kwargs)
    elif datamodule_id == "openassistant-guanaco":
        return OpenAssistantGuanacoDataModule(tokenizer, max_len, **kwargs)
    elif datamodule_id == "wikitext2":
        return Wikitext2DataModule(tokenizer, max_len, **kwargs)
    else:
        raise ValueError(f"Unknown datamodule id: {datamodule_id}")


def get_pytorch_lightning_dataset(
        datamodule_id: str,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        max_len: int,
        config: Config
) -> pl.LightningDataModule:
    """
    Returns the dataset with the given id.

    Args:
        datamodule_id (str):
            The dataset id.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            The tokenizer to use.
        max_len (int):
            The maximum length of the input sequences.
        config (Config):
            The configuration object

    Returns:
        pl.LightningDataModule:
            The dataset.
    """

    general_parameters = ["split", "batch_size", "num_workers", "seed"]
    if datamodule_id == "imdb":
        datamodule = IMDBDataModule(
            tokenizer,
            max_len,
            **config.get_dict(general_parameters)
        )
    elif datamodule_id == "openassistant-guanaco":
        datamodule = OpenAssistantGuanacoDataModule(
            tokenizer,
            max_len,
            **config.get_dict(general_parameters)
        )
    elif datamodule_id == "wikitext2":
        datamodule = Wikitext2DataModule(
            tokenizer,
            max_len,
            **config.get_dict(general_parameters)
        )
    elif datamodule_id == "openwebtext":
        datamodule = OpenWebTextStreamingDataModule(
            tokenizer,
            max_len,
            **config.get_dict(general_parameters)
        )
    else:
        raise ValueError(f"Unknown datamodule id: {datamodule_id}")

    return datamodule
