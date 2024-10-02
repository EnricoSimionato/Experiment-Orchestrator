from exporch.utils.causal_language_modeling.pl_datasets import OpenAssistantGuanacoDataModule, Wikitext2DataModule
from exporch.utils.classification.pl_datasets import IMDBDataModule


def get_datamodule(
        datamodule_id: str,
        *args,
        **kwargs
):
    """
    Returns the dataset with the given id.

    Args:
        datamodule_id (str):
            The dataset id.
        *args:
            Additional arguments to pass to the dataset.
        **kwargs:
            Additional keyword arguments to pass to the dataset

    Returns:
        pl.LightningDataModule:
            The dataset.
    """

    if datamodule_id == "imdb":
        return IMDBDataModule(*args, **kwargs)
    elif datamodule_id == "openassistant-guanaco":
        return OpenAssistantGuanacoDataModule(*args, **kwargs)
    elif datamodule_id == "wikitext2":
        return Wikitext2DataModule(*args, **kwargs)
    else:
        raise ValueError(f"Unknown datamodule id: {datamodule_id}")
