from __future__ import annotations

from typing import Any, override

import numpy as np

import torch

import pytorch_lightning as pl

import transformers

from exporch.utils import LoggingInterface
from exporch.utils.causal_language_modeling.conversation_utils import (
    get_conversation_example_1,
    get_conversation_example_2,
    start_conversation_loop
)
from exporch.utils.general_framework_utils.utility_mappings import optimizers_mapping


class CausalLMModelWrapper(pl.LightningModule):
    """
    Wrapper to train a CausalLMModel with Pytorch Lightning.

    Args:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing the optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.

    Attributes:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.
        training_step_index (int):
            Index of the training step.
        training_step_losses_sum (float):
            Sum of the training step losses.
        training_step_losses_count (int):
            Number of training step losses.
        validation_step_losses_sum (float):
            Sum of the validation step losses.
        validation_step_losses_count (int):
            Number of validation step losses.
        test_step_losses_sum (float):
            Sum of the test step losses.
        test_step_losses_count (int):
            Number of test step losses.
        loss_history (dict[str, list[float]]):
            History of the losses.
    """

    weights_to_exclude = [
        "lora",
        "vera"
    ]

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        optimizers_settings: list[dict] = None,
        max_steps: int = 1,
        stop_tokens: list[str] = ("</s>",),
        path_to_storage: str = None,
        model_dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.optimizers_settings = optimizers_settings
        self.max_steps = max_steps

        self.stop_tokens = stop_tokens

        self.training_step_index = 0

        self.training_step_losses_sum = 0
        self.training_step_losses_count = 0
        self.validation_step_losses_sum = 0
        self.validation_step_losses_count = 0
        self.test_step_losses_sum = 0
        self.test_step_losses_count = 0

        self.loss_history = {
            "train": [],
            "validation": [],
            "test": []
        }

        self.path_to_storage = path_to_storage
        self.model_dtype = model_dtype

    def configure_optimizers(
            self,
    ) -> list[dict[str, torch.optim.Optimizer | str | Any]]:
        """
        Configures the optimizer.

        Returns:
            list[dict[str, torch.optim.Optimizer | str | Any]]:
                List of dictionaries containing the optimizer and the learning rate scheduler.
        """

        if self.optimizers_settings is None or self.optimizers_settings == []:
            self.optimizers_settings = [
                {
                    "optimizer": "adamw",
                    "parameters_group": [
                        name
                        for name, param in self.model.named_parameters() if param.requires_grad
                    ],
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "lr_scheduler": "cosine_with_warmup",
                    "warmup_steps": 0,
                    "monitored_metric": "loss"
                }
            ]

        mandatory_keys = ["optimizer", "learning_rate"]
        if not all(key in optimizer_settings for key in mandatory_keys for optimizer_settings in self.optimizers_settings):
            raise ValueError(f"The optimizers' settings should contain the keys: '{', '.join(mandatory_keys)}'")
        non_supported_optimizers = set(optimizer_settings["optimizer"] for optimizer_settings in self.optimizers_settings if optimizer_settings["optimizer"].lower() not in optimizers_mapping)
        if len(non_supported_optimizers) > 0:
            raise ValueError(f"The following optimizers are not supported: {non_supported_optimizers}")
        for optimizer_index, optimizer_settings in enumerate(self.optimizers_settings):
            print(f"The optimizer settings at optimizer index {optimizer_index} do not contain the key 'parameters_group'."
                  f"Setting it to the list of all the model parameters that are trainable.")
            if "parameters_group" not in optimizer_settings:
                optimizer_settings["parameters_group"] = [
                    name
                    for name, param in self.model.named_parameters() if param.requires_grad
                ]

        optimizers = []
        for optimizer_settings in self.optimizers_settings:
            print(f"Optimized parameters: {"\n".join([name for name, param in self.model.named_parameters() if name in optimizer_settings["parameters_group"] and param.requires_grad])}")
            optimizer = optimizers_mapping[optimizer_settings["optimizer"].lower()](
                params=[param for name, param in self.model.named_parameters() if name in optimizer_settings["parameters_group"] and param.requires_grad],
                lr=optimizer_settings["learning_rate"],
                eps=1e-7 if self.model_dtype == "float16" else 1e-8
            )

            if "lr_scheduler" in optimizer_settings:
                warmup_steps = optimizer_settings["warmup_steps"] if "warmup_steps" in optimizer_settings else 0
                if warmup_steps <= 0:
                    print("Warmup steps set to 0. No warmup will be performed.")
                new_optimizer = {
                    "optimizer": optimizer
                }
                """
                "lr_scheduler": {
                    "scheduler": transformers.get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.max_steps,
                        num_cycles=0.5
                    ),
                    "monitor": optimizer_settings["monitored_metric"]
                }
                """
            else:
                new_optimizer = {
                    "optimizer": optimizer
                }

            optimizers.append(new_optimizer)

        return optimizers

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor):
                Input IDs.
            kwargs
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Model outputs.
        """

        return self.model.forward(input_ids, **kwargs)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a training step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        if hasattr(self.model, "before_training_step") and callable(getattr(self.model, "before_training_step", None)):
            self.model.before_training_step(self.training_step_index)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Computing the loss of the model for the considered train batch
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )

        self.log(
            "training_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        if issubclass(type(self.model), LoggingInterface):
            logs_dicts = self.model.get_logging_info()
            for log_element in logs_dicts:
                self.log(**log_element)

        self.training_step_losses_sum += loss.detach().cpu().to(torch.float32).numpy()
        self.training_step_losses_count += 1

        self.training_step_index += 1

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a validation step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Computing the loss of the model for the considered validation batch
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.validation_step_losses_sum += loss.detach().cpu().to(torch.float32).numpy()
        self.validation_step_losses_count += 1

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a test step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Computing the loss of the model for the considered test batch
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True
        )

        self.test_step_losses_sum += loss.detach().cpu().to(torch.float32).numpy()
        self.test_step_losses_count += 1

        return loss

    def on_train_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a training epoch
        """

    def on_validation_epoch_end(
        self
    ) -> None:
        """
        Computes and stores the average training loss on the samples considered from the previous
        validation check to the current one and the average loss on the validation set.
        """

        if self.training_step_losses_count > 0:
            avg_train_loss = self.training_step_losses_sum / self.training_step_losses_count
            self.loss_history["train"].append(avg_train_loss)
        else:
            avg_train_loss = np.nan

        self.training_step_losses_sum = 0
        self.training_step_losses_count = 0

        if self.validation_step_losses_count > 0:
            avg_val_loss = self.validation_step_losses_sum / self.validation_step_losses_count
            self.loss_history["validation"].append(avg_val_loss)
        else:
            avg_val_loss = np.nan

        self.validation_step_losses_sum = 0
        self.validation_step_losses_count = 0

        print("----------------------------------------------------------")
        if np.isnan(avg_train_loss):
            print("Number of training steps equal to 0")
        print(f'Training loss: {avg_train_loss}')
        if np.isnan(avg_val_loss):
            print("Number of validation steps equal to 0")
        print(f'Validation loss: {avg_val_loss}')
        print("----------------------------------------------------------")

    def on_test_epoch_end(
        self
    ) -> None:
        """
        Computes and stores the average validation on the test set.
        """

        if self.test_step_losses_count > 0:
            avg_test_loss = self.test_step_losses_sum / self.test_step_losses_count
            self.loss_history["test"].append(avg_test_loss)
        else:
            avg_test_loss = np.nan

        self.test_step_losses_sum = 0
        self.test_step_losses_count = 0

        print("----------------------------------------------------------")
        if np.isnan(avg_test_loss):
            print("Number of test steps equal to 0")
        print(f'Test loss: {avg_test_loss}')
        print("----------------------------------------------------------")


# TODO TOFIX
class RegularizedCausalLMModelWrapper(CausalLMModelWrapper):
    """
    Wrapper to train a CausalLMModel with Pytorch Lightning.

    Args:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing the optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        initial_regularization_weight (float):
            Initial regularization weight.
        maximum_regularization_weight (float):
            Maximum regularization weight.
        start_step_regularization (int):
            Step at which to start regularization.
        steps_regularization_weight_resets (int):
            Steps between regularization parameters resets.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.

    Attributes:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        initial_regularization_weight (float):
            Initial regularization weight.
        fixed_regularization_weight (torch.Tensor):
            Fixed regularization weight.
        adaptive_regularization_weight (torch.Tensor):
            Adaptive regularization weight.
        maximum_regularization_weight (torch.Tensor):
            Maximum regularization weight.
        start_step_regularization (int):
            Step at which to start regularization.
        steps_regularization_weight_resets (int):
            Steps between regularization parameters resets.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.
        training_step_index (int):
            Index of the training step.
        training_step_losses_sum (float):
            Sum of the training step losses.
        training_step_losses_count (int):
            Number of training step losses.
        validation_step_losses_sum (float):
            Sum of the validation step losses.
        validation_step_losses_count (int):
            Number of validation step losses.
        test_step_losses_sum (float):
            Sum of the test step losses.
        test_step_losses_count (int):
            Number of test step losses.
        loss_history (dict[str, list[float]]):
            History of the losses.
    """

    weights_to_exclude = [
        "lora",
        "vera"
    ]

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        optimizers_settings: list[dict] = None,
        max_steps: int = 1,
        stop_tokens: list[str] = ("</s>",),
        initial_regularization_weight: float = 0.01,
        maximum_regularization_weight: float = 10.0,
        start_step_regularization: int = 0,
        steps_regularization_weight_resets: int = 1000,
        path_to_storage: str = None,
        model_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            optimizers_settings,
            max_steps,
            stop_tokens,
            path_to_storage,
            model_dtype
        )

        self.initial_regularization_weight = float(initial_regularization_weight)
        self.fixed_regularization_weight = None
        self.adaptive_regularization_weight = initial_regularization_weight
        self.maximum_regularization_weight = float(maximum_regularization_weight)
        self.start_step_regularization = start_step_regularization
        self.steps_regularization_weight_resets = steps_regularization_weight_resets

    def configure_optimizers(
            self,
    ) -> list[dict[str, torch.optim.Optimizer | str | Any]]:
        """
        Configures the optimizer.

        Returns:
            list[dict[str, torch.optim.Optimizer | str | Any]]:
                List of dictionaries containing the optimizer and the learning rate scheduler.
        """

        if self.optimizers_settings is None or self.optimizers_settings == []:
            self.optimizers_settings = [
                {
                    "optimizer": "adamw",
                    "parameters_group": [
                        name
                        for name, param in self.model.named_parameters() if param.requires_grad
                    ],
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "lr_scheduler": "cosine_with_warmup",
                    "warmup_steps": 0,
                    "monitored_metric": "loss"
                }
            ]

        mandatory_keys = ["optimizer", "learning_rate"]
        if not all(key in optimizer_settings for key in mandatory_keys for optimizer_settings in self.optimizers_settings):
            raise ValueError(f"The optimizers' settings should contain the keys: '{', '.join(mandatory_keys)}'")
        non_supported_optimizers = set(optimizer_settings["optimizer"] for optimizer_settings in self.optimizers_settings if optimizer_settings["optimizer"].lower() not in optimizers_mapping)
        if len(non_supported_optimizers) > 0:
            raise ValueError(f"The following optimizers are not supported: {non_supported_optimizers}")
        for optimizer_index, optimizer_settings in enumerate(self.optimizers_settings):
            print(f"The optimizer settings at optimizer index {optimizer_index} do not contain the key 'parameters_group'."
                  f"Setting it to the list of all the model parameters.")
            if "parameters_group" not in optimizer_settings:
                optimizer_settings["parameters_group"] = [
                    name
                    for name, param in self.model.named_parameters() if param.requires_grad
                ]

        optimizers = []
        for optimizer_settings in self.optimizers_settings:
            print(f"Optimized parameters: {"\n".join([name for name, param in self.model.named_parameters() if name in optimizer_settings["parameters_group"] and param.requires_grad])}")
            optimizer = optimizers_mapping[optimizer_settings["optimizer"].lower()](
                params=[param for name, param in self.model.named_parameters() if name in optimizer_settings["parameters_group"] and param.requires_grad],
                lr=optimizer_settings["learning_rate"],
                eps=1e-7 if self.model_dtype == "float16" else 1e-8
            )

            if "lr_scheduler" in optimizer_settings:
                warmup_steps = optimizer_settings["warmup_steps"] if "warmup_steps" in optimizer_settings else 0
                if warmup_steps <= 0:
                    print("Warmup steps set to 0. No warmup will be performed.")
                new_optimizer = {
                    "optimizer": optimizer
                }
                """
                "lr_scheduler": {
                    "scheduler": transformers.get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.max_steps,
                        num_cycles=0.5
                    ),
                    "monitor": optimizer_settings["monitored_metric"]
                }
                """
            else:
                new_optimizer = {
                    "optimizer": optimizer
                }

            optimizers.append(new_optimizer)

        return optimizers

    def get_unweighted_penalization(
            self
    ) -> torch.Tensor:
        """
        Computes the unweighted penalization term as the L1 norm of the model weights.

        Returns:
            torch.Tensor:
                Unweighted penalization term.
        """

        with torch.no_grad():
            original_params = []
            for name, param in self.model.named_parameters():
                if not any(substring in name for substring in CausalLMModelWrapper.weights_to_exclude):
                    original_params.append(param)

            sum_l1_norms = torch.tensor(0.0, device=self.device)
            for i, param in enumerate(original_params):
                sum_l1_norms += torch.sum(torch.abs(param.flatten()))

            return sum_l1_norms

    def get_weighted_penalization(
            self,
            penalization: torch.Tensor,
            loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the weighted penalization term.

        Args:
            penalization (torch.Tensor):
                Unweighted penalization term.
            loss (torch.Tensor):
                Loss of the model computed for the current batch.

        Returns:
            torch.Tensor:
                Weighted penalization term.
        """

        if self.fixed_regularization_weight is None:
            self.fixed_regularization_weight = torch.tensor(
                2 * (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
        elif (self.steps_regularization_weight_resets > 0 and
              self.training_step_index % self.steps_regularization_weight_resets == 0):
            self.fixed_regularization_weight = torch.tensor(
                2 * (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
            self.adaptive_regularization_weight = torch.tensor(
                self.initial_regularization_weight,
                requires_grad=False
            )
            print("Fixed regularization weight reset to", self.fixed_regularization_weight.item(), "and adaptive regularization weight reset to", self.adaptive_regularization_weight.item())

        self.log(
            "fixed_regularization_weight",
            self.fixed_regularization_weight,
            on_step=True,
            on_epoch=False
        )

        return penalization * self.fixed_regularization_weight

    def regularization_scheduler_step(
            self
    ):
        """
        Updates the regularization weight.
        """

        k = torch.sqrt(torch.tensor(
            1.001,
            requires_grad=False
        )).to(self.adaptive_regularization_weight.device)
        self.adaptive_regularization_weight = self.adaptive_regularization_weight * k

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a training step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        loss = super().training_step(batch, batch_idx)

        if self.training_step_index >= self.start_step_regularization:
            self.log(
                "regularization_weight",
                self.regularization_weight,
                on_step=True,
                on_epoch=False,
            )

            unweighted_penalization = self.get_unweighted_penalization()
            self.log(
                "unweighted_penalization",
                unweighted_penalization,
                on_step=True,
                on_epoch=False
            )

            weighted_penalization = self.get_weighted_penalization(unweighted_penalization, loss)
            self.log(
                "weighted_penalization",
                unweighted_penalization,
                on_step=True,
                on_epoch=False
            )

            weighted_penalization = weighted_penalization.to(loss.device)
            self.adaptive_regularization_weight = self.adaptive_regularization_weight.to(loss.device)

            loss = loss + self.adaptive_regularization_weight * weighted_penalization

            self.regularization_scheduler_step()


        self.log(
            "regularized_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )

        self.log(
            "regularized_training_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss


class ChatbotModelWrapper(CausalLMModelWrapper):
    """
    Wrapper to train a CausalLMModel with Pytorch Lightning.

    Args:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing the optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.

    Attributes:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.
        training_step_index (int):
            Index of the training step.
        training_step_losses_sum (float):
            Sum of the training step losses.
        training_step_losses_count (int):
            Number of training step losses.
        validation_step_losses_sum (float):
            Sum of the validation step losses.
        validation_step_losses_count (int):
            Number of validation step losses.
        test_step_losses_sum (float):
            Sum of the test step losses.
        test_step_losses_count (int):
            Number of test step losses.
        loss_history (dict[str, list[float]]):
            History of the losses.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        optimizers_settings: list[dict] = None,
        max_steps: int = 1,
        stop_tokens: list[str] = ("[INST]", "</s>"),
        path_to_storage: str = None,
        model_dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            optimizers_settings,
            max_steps,
            stop_tokens,
            path_to_storage,
            model_dtype
        )

    @override
    def on_validation_epoch_end(
        self
    ) -> None:
        """
        Computes and stores the average training loss on the samples considered from the previous
        validation check to the current one and the average loss on the validation set.
        """

        super().on_validation_epoch_end()

        self.start_conversation_trial()

    def start_conversation_trial(
        self
    ) -> None:
        """
        Starts a conversation trial.
        """

        # Starting conversation loop
        dialogue_1 = start_conversation_loop(
            self.model,
            self.tokenizer,
            stop_tokens=self.stop_tokens,
            user_inputs=get_conversation_example_1(),
            make_model_trainable=True
        )

        # Starting conversation loop
        dialogue_2 = start_conversation_loop(
            self.model,
            self.tokenizer,
            stop_tokens=self.stop_tokens,
            user_inputs=get_conversation_example_2(),
            make_model_trainable=True
        )


class RegularizedChatbotModelWrapper(ChatbotModelWrapper, RegularizedCausalLMModelWrapper):
    """
    Wrapper to train a CausalLMModel with Pytorch Lightning.

    Args:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing the optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
            List of stop tokens.
        kfc_training (bool):
            Whether to use KFC training.
        initial_regularization_weight (float):
            Initial regularization weight.
        maximum_regularization_weight (float):
            Maximum regularization weight.
        start_step_regularization (int):
            Step at which to start regularization.
        steps_regularization_weight_resets (int):
            Steps between regularization parameters resets.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.

    Attributes:
        model (transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        optimizers_settings (list[dict]):
            List of dictionaries containing optimizers' settings.
        max_steps (int):
            Maximum number of steps.
        stop_tokens (list[str]):
        initial_regularization_weight (float):
            Initial regularization weight.
        fixed_regularization_weight (torch.Tensor):
            Fixed regularization weight.
        adaptive_regularization_weight (torch.Tensor):
            Adaptive regularization weight.
        maximum_regularization_weight (torch.Tensor):
            Maximum regularization weight.
        start_step_regularization (int):
            Step at which to start regularization.
        steps_regularization_weight_resets (int):
            Steps between regularization parameters resets.
        path_to_storage (str):
            Path to storage.
        model_dtype (torch.dtype):
            Data type.
        training_step_index (int):
            Index of the training step.
        training_step_losses_sum (float):
            Sum of the training step losses.
        training_step_losses_count (int):
            Number of training step losses.
        validation_step_losses_sum (float):
            Sum of the validation step losses.
        validation_step_losses_count (int):
            Number of validation step losses.
        test_step_losses_sum (float):
            Sum of the test step losses.
        test_step_losses_count (int):
            Number of test step losses.
        loss_history (dict[str, list[float]]):
            History of the losses.
    """

    def __init__(
            self,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            optimizers_settings: list[dict] = None,
            max_steps: int = 1,
            stop_tokens: list[str] = ("[INST]", "</s>"),
            initial_regularization_weight: float = 0.01,
            maximum_regularization_weight: float = 10.0,
            start_step_regularization: int = 0,
            steps_regularization_weight_resets: int = 1000,
            path_to_storage: str = None,
            model_dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            optimizers_settings,
            max_steps,
            stop_tokens,
            initial_regularization_weight,
            maximum_regularization_weight,
            start_step_regularization,
            steps_regularization_weight_resets,
            path_to_storage,
            model_dtype
        )
