import pytorch_lightning as pl

from exporch import Config
from exporch.utils.causal_language_modeling.pl_trainer import get_causal_lm_trainer


def get_pytorch_lightning_trainer(
        task_id: str,
        config: Config
) -> pl.Trainer:
    """
    Returns the correct PyTorch Lightning model wrapper based on the task to be performed.

    Args:
        task_id (str):
            The task identifier.
        config (Config):
            The configuration object.

    Returns:
        pl.Trainer:
            The PyTorch Lightning trainer.
    """

    task_id = task_id.lower().replace("-", "").replace("_", "")

    if task_id in ["causallanguagemodelling", "causallm"]:
        trainer = get_causal_lm_trainer(config)
    else:
        raise ValueError(f"Unknown task id: {task_id}")

    return trainer