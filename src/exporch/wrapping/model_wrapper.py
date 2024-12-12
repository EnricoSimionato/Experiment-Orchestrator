from abc import ABC, abstractmethod
from typing import Any

import torch

import transformers

from exporch.utils import LoggingInterface


class ModelWrapper(torch.nn.Module, LoggingInterface, ABC):
    """
    Abstract class to wrap a model.
    Subclasses must implement the get_model method that returns the wrapped model.
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
    ) -> Any:
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
    ) -> None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
        """
        Returns the wrapped model.

        Returns:
            torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                The wrapped model.
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
    def device(self):
        """
        Returns the device where the model is stored.

        Returns:
            torch.device: The device where the model is stored.
        """

        return next(self.parameters()).device
