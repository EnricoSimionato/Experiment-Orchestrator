import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from exporch.configuration.config import Config
from exporch.utils.device_utils.device_utils import get_available_device


def get_causal_lm_trainer(
        config: Config,
) -> pl.Trainer:
    """
    Returns the PyTorch Lightning trainer for a causal lm problem.

    Args:
        config (Config):
            A dictionary containing the configuration parameters for the trainer.

    Returns:
        pl.Trainer:
            The PyTorch Lightning trainer for the classification problem.
    """

    # Defining callbacks
    callbacks = [
        # Defining early stopping callback
        pl.callbacks.EarlyStopping(
            monitor="validation_loss",
            min_delta=0.001,
            patience=3,
            verbose=True
        ),
        # Defining checkpointing callback
        pl.callbacks.ModelCheckpoint(
            dirpath=config.get("path_to_checkpoints"),
            filename="{epoch}-{validation_loss:.2f}",
            monitor="validation_loss"
        ),
        # Defining learning rate monitor callback
        pl.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
    ]

    # Defining loggers
    loggers = [
        TensorBoardLogger(
            save_dir=config.get("path_to_training_logs"),
            name="tensorboard_logs"
        ),
        CSVLogger(
            save_dir=config.get("path_to_training_logs"),
            name="csv_logs"
        ),
        #CometLogger(
        #    save_dir=config.get("path_to_training_logs"),
        #    name="comet_logs"
        #)
        #pl.loggers.WandbLogger(
        #    save_dir=config.get("path_to_training_logs"),
        #    name="wandb_logs"
        #)
    ]

    # Defining trainer settings
    lightning_trainer = pl.Trainer(
        max_epochs=config.get("max_epochs"),
        val_check_interval=1/config.get("num_checks_per_epoch"),
        accumulate_grad_batches=config.get("gradient_accumulation_steps"),
        callbacks=callbacks,
        accelerator=get_available_device(
            config.get("device"),
            just_string=True
        ),
        logger=loggers,
        log_every_n_steps=1
    )

    return lightning_trainer
