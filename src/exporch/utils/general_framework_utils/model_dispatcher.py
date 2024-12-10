import torch
import pytorch_lightning as pl

import transformers

from exporch import Config
from exporch.utils.causal_language_modeling.pl_models import CausalLMModelWrapper, ChatbotModelWrapper, \
    RegularizedCausalLMModelWrapper
from exporch.utils.classification.pl_models import ClassifierModelWrapper

def get_pytorch_lightning_model(
        model_to_wrap: torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        task_id: str,
        config: Config
) -> pl.LightningModule:
    """
    Returns the correct PyTorch Lightning model wrapper based on the task to be performed.

    Args:
        model_to_wrap (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            The tokenizer to use.
        task_id (str):
            The task identifier.
        config (Config):
            The configuration object.
    Returns:
        pl.LightningModule:
            The PyTorch Lightning model wrapper.
    """

    task_id = task_id.lower().replace("-", "").replace("_", "")

    general_parameters = ["optimizers_settings", "max_steps", "path_to_storage"]
    regularized_parameters = ["initial_regularization_weight", "maximum_regularization_weight", "start_step_regularization", "steps_regularization_weight_resets"]

    if task_id in ["causallanguagemodelling", "causallm"]:
        model = CausalLMModelWrapper(
            model_to_wrap,
            tokenizer,
            **config.get_dict(general_parameters)
        )
    elif task_id in ["penalizedcausallanguagemodelling", "penalizedcausallm"]:
        model = RegularizedCausalLMModelWrapper(
            model_to_wrap,
            tokenizer,
            **config.get_dict(general_parameters),
            **config.get_dict(regularized_parameters)
        )
    elif task_id in ["chatbot"]:
        model = ChatbotModelWrapper(
            model_to_wrap,
            tokenizer,
            **config.get_dict(general_parameters)
        )
    elif task_id in ["classification"]:
        model = ClassifierModelWrapper(
            model_to_wrap,
            tokenizer,
            **config.get_dict(general_parameters)
        )
    else:
        raise ValueError(f"Unknown task id: {task_id}")

    return model