import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

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

    if not config.contains("max_epochs"):
        raise ValueError("The configuration must contain the maximum number of epochs ('max_epochs' key).")
    if not config.contains("path_to_checkpoints"):
        raise ValueError("The configuration must contain the path to the checkpoints ('path_to_checkpoints' key).")
    if not config.contains("path_to_training_logs"):
        raise ValueError("The configuration must contain the path to the training logs ('path_to_training_logs' key).")

    # Defining callbacks
    callbacks = [
        # Defining checkpointing callback
        pl.callbacks.ModelCheckpoint(
            dirpath=config.get("path_to_checkpoints"),
            filename="{epoch}-{validation_loss:.2f}",
            monitor="validation_loss"
        ),
        # Defining learning rate monitor callback
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]

    if not config.contains("early_stopping") or (config.contains("early_stopping") and config.get("early_stopping")):
        # Defining early stopping callback
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation_loss", min_delta=0.001, patience=5, verbose=True)
        callbacks += [early_stopping_callback]

    # Defining loggers
    loggers = [
        TensorBoardLogger(
            save_dir=config.get("path_to_training_logs"),
            name="tensorboard_logs"
        ),
        CSVLogger(
            save_dir=config.get("path_to_training_logs"),
            name="csv_logs"
        )
    ]

    max_epochs = config.get("max_epochs")
    num_ckecks_per_epoch = config.get("num_checks_per_epoch") if config.contains("num_checks_per_epoch") else None
    val_check_interval = config.get("val_check_interval") if config.contains("val_check_interval") else None
    if val_check_interval is None:
        if num_ckecks_per_epoch is None:
            val_check_interval = 1.0
        else:
            val_check_interval = float(1/num_ckecks_per_epoch)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps") if config.contains("gradient_accumulation_steps") else 1
    fast_dev_run = config.get("fast_dev_run") if config.contains("fast_dev_run") else False

    # Defining trainer settings
    lightning_trainer = pl.Trainer(
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=gradient_accumulation_steps,
        callbacks=callbacks,
        accelerator=get_available_device(config.get("device"), just_string=True),
        logger=loggers,
        log_every_n_steps=1,
        fast_dev_run=fast_dev_run
    )

    return lightning_trainer
