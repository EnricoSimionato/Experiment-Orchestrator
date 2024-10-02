from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import yaml
import logging
import os
import pickle as pkl
from typing import Any

import torch

import pytorch_lightning as pl

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer

from exporch.configuration.config import  Config, ExperimentStatus
from exporch.utils.storage_utils.storage_utils import check_path_to_storage, load_model_and_info, store_model_and_info

# TODO remove the dependence on the ClassifierModelWrapper, get_classification_trainer, ChatbotModelWrapper, get_causal_lm_trainer,. .. . ..
from exporch.utils.classification.pl_models import ClassifierModelWrapper
from exporch.utils.classification.pl_trainer import get_classification_trainer

from exporch.utils.causal_language_modeling.pl_models import ChatbotModelWrapper
from exporch.utils.causal_language_modeling.pl_trainer import get_causal_lm_trainer


class Experiment:
    """
    Class to run an experiment in a standardized way.

    The experiment comprises the following steps:
    1. Experiment initialization:
        - Starts the experiment defining the paths where to store the configuration, the model, the logs, and the
          checkpoints.
    2. Experiment execution:
        - Initializes the PyTorch Lightning model and trainer.
        - Validates the model before training.
        - Trains the model.
        - Validates the model after training.
        - Tests the model.
    3. Experiment storage:
        - Stores the model and the configuration.

    Args:
        task (str):
            The task to perform.
        model (torch.nn.Module):
            The model to use.
        dataset (pl.LightningDataModule):
            The dataset to use.
        config (Config):
            The configuration containing the information about the experiment.
        tokenizer (AutoTokenizer | PreTrainedTokenizer, optional):
            The tokenizer to use.
        store_model_function (callable, optional):
            The function to store the model.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        task (str):
            The task to perform.
        model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
            The model to use.
        dataset (pl.LightningDataModule):
            The dataset to use.
        config (Config):
            The configuration containing the information about the experiment.
        tokenizer (AutoTokenizer | PreTrainedTokenizer):
            The tokenizer to use.
        lightning_model (pl.LightningModule):
            The PyTorch Lightning model.
        lightning_trainer (pl.Trainer):
            The PyTorch Lightning trainer.
        store_model_function (callable):
            The function to store the model.
    """

    def __init__(
            self,
            task: str,
            model: [torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            dataset: pl.LightningDataModule,
            config: Config,
            tokenizer: AutoTokenizer | PreTrainedTokenizer = None,
            store_model_function: callable = None,
            **kwargs
    ) -> None:
        self.task = task
        self.model = model
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer

        self.lightning_model = None
        self.lightning_trainer = None

        self.store_model_function = store_model_function

        config.set("task", task)

    def start_experiment(
            self,
            **kwargs
    ) -> dict:
        """
        Initializes the experiment.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the paths where to store the configuration, the model, the logs, and the
                checkpoints.
        """

        if self.config.status is ExperimentStatus.NOT_STARTED:
            self.config.start_experiment()
            print("Experiment started")
        else:
            print("Running the experiment, it is already started")

        paths_dict = self.config.get_paths()

        return paths_dict

    def run_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Runs the experiment.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        self.start_experiment(**kwargs)

        self.dataset.setup()

        self.lightning_trainer = self._get_trainer(**kwargs)
        self.lightning_model = self._get_lightning_model(**kwargs)
        #validate_results = self._validate(**kwargs)
        #print(f"Validate results before training: {validate_results}")

        fit_results = self._fit(**kwargs)
        print(f"Fit results: {fit_results}")
        validate_results = self._validate(**kwargs)
        print(f"Validate results: {validate_results}")
        test_results = self._test(**kwargs)
        print(f"Test results: {test_results}")

        self.config.end_experiment()

        self._store_experiment()

        print("Experiment completed")

    def _get_lightning_model(
            self,
            **kwargs
    ) -> pl.LightningModule:
        """
        Returns the PyTorch Lightning model to train the model using the PyTorch Lightning framework.
        The model is defined based on the task to perform.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            pl.LightningModule:
                The PyTorch Lightning model.
        """
        pl_model = None

        # Defining training arguments
        training_keys = ["optimizers_settings"]
        training_args = {}
        for key in training_keys:
            if self.config.contains(key):
                training_args[key] = self.config.get(key)

        if self.config.contains("max_epochs"):
            training_args["max_steps"] = self.config.get("max_epochs") * len(self.dataset.train_dataloader())
        else:
            training_args["max_steps"] = len(self.dataset.train_dataloader())

        # Defining the data type of the model
        dtype = (self.config.get("dtype") if self.config.contains("dtype") else "float32")

        if self.task == "classification":
            pl_model = ClassifierModelWrapper(
                model=self.model,
                tokenizer=self.tokenizer,

                num_classes=self.config.get("num_classes"),
                id2label=self.config.get("id2label"),
                label2id=self.config.get("label2id"),

                **training_args,

                path_to_storage=self.config.get("path_to_experiment"),
                dtype=dtype
            )

        elif self.task == "question-answering":
            pass
        elif self.task == "chatbot":
            pl_model = ChatbotModelWrapper(
                model=self.model,
                tokenizer=self.tokenizer,

                **training_args,

                stop_tokens=self.config.get("stop_tokens"),

                path_to_storage=self.config.get("path_to_experiment"),
                dtype=dtype
            )

        else:
            raise ValueError(f"Task {self.task} not recognized")

        return pl_model

    def _get_trainer(
            self,
            **kwargs
    ) -> pl.Trainer:
        """
        Returns the PyTorch Lightning trainer to train the model using the PyTorch Lightning framework.
        The trainer is defined based on the task to perform.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            pl.Trainer:
                The PyTorch Lightning trainer.
        """

        if self.task == "classification":
            return get_classification_trainer(
                self.config,
                **kwargs
            )
        elif self.task == "question-answering":
            pass
        elif self.task == "chatbot":
            return get_causal_lm_trainer(
                self.config,
                **kwargs
            )
        else:
            raise ValueError(f"Task {self.task} not recognized")

    def _fit(
            self,
            **kwargs
    ) -> dict:
        """
        Trains the model using the PyTorch Lightning framework.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the results of the training.
        """

        return self.lightning_trainer.fit(
            self.lightning_model,
            self.dataset
        )

    def _validate(
            self,
            **kwargs
    ) -> dict:
        """
        Validates the model using the PyTorch Lightning framework.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the results of the validation.
        """

        return self.lightning_trainer.validate(
            self.lightning_model,
            self.dataset
        )

    def _test(
            self,
            **kwargs
    ) -> dict:
        """
        Tests the model using the PyTorch Lightning framework.

        Args:
            kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the results of the testing.
        """

        return self.lightning_trainer.test(
            self.lightning_model,
            self.dataset
        )

    def _store_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Stores the model and the configuration.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        store_model_and_info(
            self.model,
            self.config,
            tokenizer=self.tokenizer,
            store_model_function=self.store_model_function
        )

    def store_experiment(
            self,
            **kwargs
    ) -> None:
        """
        Stores the model and the configuration.

        Args:
            kwargs:
                Additional keyword arguments.
        """

        self._store_experiment()

    @classmethod
    def load_experiment(
            cls,
            path_to_experiment: str,
            load_model_function: callable = None,
            **kwargs
    ) -> Experiment:
        """
        Loads the model and the configuration.

        Args:
            path_to_experiment (str):
                The path to the experiment.
            load_model_function (callable, optional):
                The function to load the model. Defaults to None.
            kwargs:
                Additional keyword arguments.

        Returns:
            Experiment:
                The loaded experiment.
        """

        model, config, tokenizer = load_model_and_info(
            path_to_experiment,
            load_model_function=load_model_function
        )

        experiment = cls(
            task=config.get("task"),
            model=model,
            dataset=None,
            config=config,
            tokenizer=tokenizer
        )

        return experiment

    def get_model(
            self
    ) -> torch.nn.Module:
        """
        Returns the model.

        Returns:
            torch.nn.Module:
                The model.
        """

        return self.model

    def get_config(
            self
    ) -> Config:
        """
        Returns the configuration.

        Returns:
            Config:
                The configuration.
        """

        return self.config

    def get_tokenizer(
            self
    ) -> AutoTokenizer | PreTrainedTokenizer:
        """
        Returns the tokenizer.

        Returns:
            AutoTokenizer | PreTrainedTokenizer:
                The tokenizer.
        """

        return self.tokenizer


class GeneralPurposeExperimentFactory:
    """
    Class to create an experiment based on the experiment type.

    Class Attributes:
        mapping (dict):
            The mapping between the experiment type and the corresponding class.
    """

    mapping = {}

    @classmethod
    def register(
            cls,
            mapping: dict
    ) -> None:
        """
        Registers the mapping between the experiment type and the corresponding class.

        Args:
            mapping (dict):
                The mapping between the experiment type and the corresponding class.
        """

        cls.mapping = mapping

    @classmethod
    def create(
            cls,
            config_file_path: str
    ) -> GeneralPurposeExperiment:
        """
        Creates an experiment returning the correct experiment class based on the experiment type in the configuration
        file.

        Args:
            config_file_path (str):
                The path to the configuration file.

        Returns:
            GeneralPurposeExperiment:
                The experiment.
        """

        config = Config(config_file_path)
        if not config.contains("experiment_type"):
            raise ValueError("The configuration file does not contain the key 'experiment_type'.")

        for key in cls.mapping.keys():
            if key in config.get("experiment_type"):
                return cls.mapping[key](config_file_path)

        raise ValueError(f"The experiment type {config.get('experiment_type')} is not recognized.")


class GeneralPurposeExperiment(ABC):
    """
    Class to run an experiment in a standardized way.

    Args:
        config_file_path (str):
            The path to the configuration file.

    Attributes:
        config (Config):
            The configuration containing the information about the experiment.
        status (ExperimentStatus):
            The status of the experiment.
        running_times (list[dict]):
            The running times of the experiment.
        mandatory_keys (list):
            The mandatory keys in the configuration.
        data (Any):
            The data already computed in the experiment.
    """

    mandatory_keys = ["path_to_storage", "experiment_type", "model_id"]

    def __init__(
            self,
            config_file_path: str
    ) -> None:
        # Loading the configuration and checking the mandatory keys
        self.config = Config(config_file_path)
        self.config.check_mandatory_keys(self.get_mandatory_keys())

        self.status = ExperimentStatus.NOT_STARTED
        self.running_times = []

        # Checking the path to the storage
        file_available, directory_path, file_name = check_path_to_storage(
            self.config.get("path_to_storage"),
            self.config.get("experiment_type"),
            self.config.get("model_id").split("/")[-1],
            self.config.get("version") if self.config.contains("version") else None,
        )
        self.config.update(
            {
                "file_available": file_available,
                "file_path": os.path.join(directory_path, file_name),
                "directory_path": directory_path,
                "file_name": file_name,
                "log_path": os.path.join(directory_path, "logs.log")
            }
        )

        # Creating the logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(self.config.get("log_path"))]
        )
        self.logger = logging.getLogger()
        self.log(f"Configuration file from: '{config_file_path}'.")

        # Doing a consistency between the stored configuration and the loaded configuration
        self.check_stored_configuration_consistency(os.path.join(directory_path, "config.yaml"))
        self.log(f"Consistency check between the stored configuration and the loaded configuration passed.")

        # Loading the data if the file is available
        data = None
        if file_available:
            print(f"The file '{self.config.get('file_path')}' is available.")
            with open(self.config.get("file_path"), "rb") as f:
                data = pkl.load(f)

        self.data = data

    def get_config(
            self
    ) -> Config:
        """
        Returns the configuration.

        Returns:
            Config:
                The configuration.
        """

        return self.config

    def get_data(
            self
    ) -> Any:
        """
        Returns the data computed in the experiment.

        Returns:
            Any:
                The data computed in the experiment.
        """

        return self.data

    def set_data(
            self,
            data: Any
    ) -> None:
        """
        Sets the data computed in the experiment.

        Args:
            data (Any):
                The data computed in the experiment.
        """

        self.data = data

    def log(
            self,
            message: str,
            level: str = "info"
    ) -> None:
        """
        Logs a message in the log file.

        Args:
            message (str):
                The message to log.
            level (str, optional):
                The level of the log. Defaults to "info".
        """

        # TODO Allow different levels of logs
        if level == "info":
            self.logger.info(message)
        else:
            raise NotImplementedError("Only info level is implemented.")

    def check_stored_configuration_consistency(
            self,
            config_file_path: str
    ) -> None:
        """
        Checks if there are any differences between the current configuration and the one stored in config.yaml (if it
        exists) in the directory_path.

        Args:
            config_file_path (str):
                The path where the config.yaml is expected to be found.
        """

        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                stored_config = yaml.safe_load(f)

            # Comparing with the current configuration
            differences = self.compare_configs(self.config.to_dict(), stored_config)

            if differences:
                print("Configuration differences found:")
                self.log("Configuration differences found:")
                for key, (current_value, stored_value) in differences.items():
                    diff_message = f"- {key}: Current: {current_value}, Stored: {stored_value}"
                    print(diff_message)
                    self.log(diff_message + "\n")
                user_confirmation = input("There are differences in the configuration. Do you want to continue with the new configuration? (yes/no): ").strip().lower()
                if user_confirmation != "yes" and user_confirmation != "y":
                    print("Experiment aborted due to configuration inconsistency.")
                    self.status = ExperimentStatus.STOPPED
                    raise Exception("Experiment aborted due to configuration inconsistency.")
            else:
                self.log("No differences found between the current configuration and the stored configuration.")
        else:
            self.log(f"No stored configuration found at {config_file_path}, continuing with the current configuration.")

    @staticmethod
    def compare_configs(
            current_config: dict,
            stored_config: dict
    ) -> dict:
        """
        Compares two configuration dictionaries and returns the differences.

        Args:
            current_config (dict):
                The current configuration.
            stored_config (dict):
                The stored configuration from config.yaml.

        Returns:
            dict:
                A dictionary with keys as the differing config parameters and values as tuples of
                (current_value, stored_value).
        """

        not_evaluated_keys = ["version", "file_available", "just_plot", "device"]
        for key in not_evaluated_keys:
            if key in current_config:
                del current_config[key]
            if key in stored_config:
                del stored_config[key]

        differences = {}
        for key in current_config:
            if key in stored_config:
                if current_config[key] != stored_config[key]:
                    differences[key] = (current_config[key], stored_config[key])
            else:
                differences[key] = (current_config[key], None)

        # Finding any keys in the stored config that are not in the current config
        for key in stored_config:
            if key not in current_config:
                differences[key] = (None, stored_config[key])

        return differences

    def get_mandatory_keys(
            self
    ) -> list:
        """
        Returns all mandatory keys, including the keys defined in superclasses.

        Returns:
            list:
                A list of all mandatory keys from the current class and its superclasses.
        """
        # Initializing a list with mandatory keys from the current class
        mandatory_keys = list(self.mandatory_keys)

        # Traversing the superclass hierarchy and collect mandatory keys
        for cls in self.__class__.__mro__[1:]:
            if hasattr(cls, "mandatory_keys"):
                mandatory_keys.extend(cls.mandatory_keys)

        return list(set(mandatory_keys))

    def launch_experiment(
            self
    ) -> None:
        """
        Launches the experiment. Executing the operations that are defined in the specific subclass of
        GeneralPurposeExperiment.
        """

        self.running_times.append({"start_time": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), "end_time": None})
        self.store_configuration()
        try :
            self.status = ExperimentStatus.RUNNING
            self._run_experiment()
        except Exception as e:
            self.running_times[-1]["end_time"] = datetime.now()
            self.status = ExperimentStatus.STOPPED
            self.store_configuration()
            raise e
        self.running_times[-1]["end_time"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.status = ExperimentStatus.COMPLETED
        self.store_configuration()

    @abstractmethod
    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment. Performing the operations that are defined in the specific subclass of
        GeneralPurposeExperiment. This method is abstract and must be implemented in the specific subclass.
        """

        pass

    def store_configuration(
            self
    ) -> None:
        """
        Stores the configuration of the experiment.
        """

        self.config.store(self.config.get("directory_path"))

    def store_data(
            self
    ) -> None:
        """
        Stores the data computed in the experiment.
        """

        with open(self.config.get("file_path"), "wb") as f:
            pkl.dump(self.data, f)

    def exists_file(
            self,
            file_name: str
    ) -> bool:
        """
        Checks if a file exists in the directory path.

        Args:
            file_name (str):
                The name of the file to check.

        Returns:
            bool:
                True if the file exists, False otherwise.
        """

        return os.path.exists(os.path.join(self.config.get("directory_path"), file_name)) and os.path.isfile(os.path.join(self.config.get("directory_path"), file_name))

    def store(
            self,
            data: Any,
            file_name: str,
            extension: str = "pkl"
    ) -> None:
        """
        Stores the data in a file.

        Args:
            data (Any):
                The data to store.
            file_name (str):
                The name of the file where to store the data.
            extension (str, optional):
                The extension of the file. Defaults to "pkl".
        """

        if data is None:
            raise ValueError("The data to store is None.")

        self.log(f"Trying to store data in file '{file_name}' with extension '{extension}'.")
        if extension == "pkl":
            with open(os.path.join(self.config.get("directory_path"), file_name), "wb") as f:
                pkl.dump(data, f)
        elif extension == "pt":
            with open(os.path.join(self.config.get("directory_path"), file_name), "wb") as f:
                torch.save(data, f)
        else:
            raise NotImplementedError(f"Extension {extension} not implemented.")
        self.log(f"Successfully stored data in file '{os.path.join(self.config.get('directory_path'), file_name)}'.")

    def load(
            self,
            file_name: str,
            extension: str = "pkl"
    ) -> Any:
        """
        Loads the data from a file.

        Args:
            file_name (str):
                The name of the file where to load the data.
            extension (str, optional):
                The extension of the file. Defaults to "pkl".
        """

        self.log(f"Trying to load data from file '{file_name}' with extension '{extension}'.")
        try:
            if extension == "pkl":
                with open(os.path.join(self.config.get("directory_path"), file_name), "rb") as f:
                    data = pkl.load(f)
            elif extension == "pt":
                with open(os.path.join(self.config.get("directory_path"), file_name), "rb") as f:
                    data = torch.load(f)
            else:
                raise NotImplementedError(f"Extension {extension} not implemented.")
            self.log(f"Successfully loaded data from file '{os.path.join(self.config.get('directory_path'), file_name)}'.")
        except FileNotFoundError:
            self.log(f"File '{os.path.join(self.config.get('directory_path'), file_name)}' not found.")
            data = None

        return data


class NopGeneralPurposeExperiment(GeneralPurposeExperiment):
    """
    A no-operation experiment that does nothing.
    """

    mandatory_keys = ["nop"]

    def _run_experiment(
            self
    ) -> None:
        """
        Does nothing.
        """

        pass