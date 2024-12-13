from abc import ABC, abstractmethod

import torch

import transformers

from exporch.utils import LoggingInterface


class ModelWrapper(torch.nn.Module, LoggingInterface, ABC):
    """
    Abstract class of a model wrapper.
    """

    def __init__(
            self
    ) -> None:
        torch.nn.Module.__init__(self)
        LoggingInterface.__init__(self)

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            *args:
                The input arguments.
            **kwargs:
                The input keyword arguments.
        """

    def get_model(
            self
    ) -> torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
        """
        Returns the base model.

        Returns:
            torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                The base model.
        """

        return self.model

    @torch.no_grad()
    def get_logging_info(
            self
    ) -> list:
        """
        Returns additional information to log.

        Returns:
            list:
                Additional information to log.
        """

        return []

    @property
    def device(
            self
    ) -> torch.device:
        """
        Returns the device where the model is loaded.

        Returns:
            torch.device:
                The device where the model is loaded.
        """

        return next(self.parameters()).device
