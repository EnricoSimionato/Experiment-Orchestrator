from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import logging
import os
import pickle as pkl
from typing import Any
import yaml

import torch

from exporch.configuration.config import  Config, ExperimentStatus
from exporch.utils.storage_utils.storage_utils import check_path_to_storage


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
        data (list):
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
        file_available, experiment_root_path, file_name = check_path_to_storage(
            self.config.get("path_to_storage"),
            self.config.get("experiment_type"),
            self.config.get("model_id").split("/")[-1],
            self.config.get("version") if self.config.contains("version") else None,
        )
        self.config.update(
            {
                "file_available": file_available,
                "file_path": os.path.join(experiment_root_path, file_name),
                "experiment_root_path": experiment_root_path,
                "file_name": file_name,
                "log_path": os.path.join(experiment_root_path, "logs.log")
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
        self.check_stored_configuration_consistency(os.path.join(experiment_root_path, "config.yaml"))
        self.log(f"Consistency check between the stored configuration and the loaded configuration passed.")

        # Loading the data if the file is available
        data = None
        if file_available:
            self.log(f"The file '{self.config.get('file_path')}' is available.", print_message=True)
            self.log(f"Trying to load the data from the file '{self.config.get('file_path')}'.")
            with open(self.config.get("file_path"), "rb") as f:
                data = pkl.load(f)
            self.log(f"Successfully loaded the data from the file '{self.config.get('file_path')}'.")

        self.low_memory_mode = self.config.get("low_memory_mode") if self.config.contains("low_memory_mode") else True
        if self.low_memory_mode:
            self.log("Running in low memory mode. The data will be stored in the disk and loaded when needed.")
        else:
            self.log("Running in normal mode. The data will persist in main memory.")
        self.data = data

    def get_experiment_path(
            self
    ) -> str:
        """
        Returns the path to the experiment.

        Returns:
            str:
                The path to the experiment.
        """

        return self.config.get("experiment_root_path")

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

    def set_config(
            self,
            config: Config
    ) -> None:
        """
        Updates the configuration.

        Args:
            config (Config):
                The configuration to update.
        """

        self.config = config

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

    def is_low_memory_mode(
            self
    ) -> bool:
        """
        Returns whether the experiment is running in low memory mode.

        Returns:
            bool:
                True if the experiment is running in low memory mode, False otherwise.
        """

        return self.low_memory_mode

    def set_data(
            self,
            data: Any,
            position: int = None,
            store: bool = True
    ) -> None:
        """
        Sets and stores the data computed in the experiment.

        Args:
            data (Any):
                The data computed in the experiment.
            position (int, optional):
                The position where to store the data. Defaults to None.
            store (bool, optional):
                Whether to store the data. Defaults to True.
        """

        if data is None:
            self.log("The data to store is None. The data will not be stored.")
            self.data = None

        if not isinstance(data, tuple) and not isinstance(data, list) and position is None:
            raise ValueError("When the provided data is not a tuple, the position must be provided.")

        if position is not None:
            if self.data is None:
                formatted_data = [None] * (position + 1)
            else:
                if len(self.data) <= position:
                    formatted_data = list(self.data) + [None] * (position - len(self.data) + 1)
                else:
                    formatted_data = list(self.data)

            formatted_data[position] = data
        else:
            formatted_data = data

        self.data = tuple(formatted_data)

        if store:
            with open(self.config.get("file_path"), "wb") as f:
                pkl.dump(self.data, f)

    def _load_data(
            self,
    ) -> tuple:
        """
        Loads the data computed in the experiment.

        Returns:
            tuple:
                The data computed in the experiment.
        """

        if not os.path.exists(self.config.get("file_path")):
            data = None
        else:
            with open(self.config.get("file_path"), "rb") as f:
                data = pkl.load(f)

        self.data = data

        return data

    def log(
            self,
            message: str,
            print_message: bool = False,
            level: str = "info"
    ) -> None:
        """
        Logs a message in the log file.

        Args:
            message (str):
                The message to log.
            print_message (bool, optional):
                Whether to print the message. Defaults to False.
            level (str, optional):
                The level of the log. Defaults to "info".
        """

        # TODO Allow different levels of logs
        if level == "info":
            self.logger.info(message)
        else:
            raise NotImplementedError("Only info level is implemented.")

        if print_message:
            print(message)

    def check_stored_configuration_consistency(
            self,
            config_file_path: str
    ) -> None:
        """
        Checks if there are any differences between the current configuration and the one stored in config.yaml (if it
        exists) in the experiment_root_path.

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
                    self.log(diff_message + "\n", print_message=True)
                user_confirmation = input("There are differences in the configuration. Do you want to continue with the new configuration? (yes/no): ").strip().lower()
                if user_confirmation != "yes" and user_confirmation != "y":
                    print("Experiment aborted due to configuration inconsistency.")
                    self.status = ExperimentStatus.STOPPED
                    raise Exception("Experiment aborted due to configuration inconsistency.")
                else:
                    # Adding the previous values to the configuration with the key as key_previous_0, key_previous_1, ...
                    for key in differences:
                        counter = 0
                        new_key = f"_{key}_previous_{counter}"

                        # Incrementing counter if a previous key exists in the stored configuration
                        previous_elements = {}
                        while new_key in stored_config:
                            previous_elements[new_key] = stored_config[new_key]
                            counter += 1
                            new_key = f"_{key}_previous_{counter}"

                        previous_elements[new_key] = stored_config[key] if key in stored_config else None

                        # Adding the previous values to the configuration
                        self.config.update(previous_elements)
                    # Storing the new configuration
                    self.store_configuration()
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

        non_evaluated_keys = ["version", "file_available", "just_plot", "device"]

        differences = {}
        for key in current_config:
            if key not in non_evaluated_keys and not key.startswith("_"):
                if key in stored_config:
                    if current_config[key] != stored_config[key]:
                        differences[key] = (current_config[key], stored_config[key])
                else:
                    differences[key] = (current_config[key], None)

        # Finding any keys in the stored config that are not in the current config
        for key in stored_config:
            if key not in non_evaluated_keys and not key.startswith("_"):
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

    def create_experiment_directory(
            self,
            directory_name: str,
            key_in_configuration: str = None
    ) -> None:
        """
        Creates a directory for the experiment.

        Args:
            directory_name (str):
                The name of the directory to be created.
            key_in_configuration (str):
                The key in the configuration to store the path to the directory.
        """

        self.config.create_directory(self.config.get("experiment_root_path"), directory_name, key_in_configuration)

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
            if self.config.contains("just_plot") and self.config.get("just_plot") and self.data is None:
                raise ValueError("It is not possible to plot the results without performing the analysis. "
                                 "Set 'just_plot' to False.")
            if not self.config.contains("just_plot") or not self.config.get("just_plot"):
                self.log(f"Starting the experiment.")
                self._run_experiment()
                self._postprocess_results()
            self._plot_results(self.config, self.get_data())
        except (KeyboardInterrupt, SystemExit) as e:
            self.log(f"Analysis interrupted by the user.", print_message=True)
            self.running_times[-1]["end_time"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.status = ExperimentStatus.STOPPED
            self.store_configuration()
            raise e
        except Exception as e:
            self.log(f"Unexpected error: {e}", print_message=True)
            self.running_times[-1]["end_time"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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

    @abstractmethod
    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results obtained from the experiment.
        The performed operations will depend on the specific subclass of GeneralPurposeExperiment.
        """

        pass

    @abstractmethod
    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results obtained from the experiment.
        The performed operations will depend on the specific subclass of GeneralPurposeExperiment.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        pass

    def store_configuration(
            self
    ) -> None:
        """
        Stores the configuration of the experiment.
        """

        self.config.store(self.config.get("experiment_root_path"))

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

        return os.path.exists(os.path.join(self.config.get("experiment_root_path"), file_name)) and os.path.isfile(os.path.join(self.config.get("experiment_root_path"), file_name))

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
            with open(os.path.join(self.config.get("experiment_root_path"), file_name), "wb") as f:
                pkl.dump(data, f)
        elif extension == "pt":
            with open(os.path.join(self.config.get("experiment_root_path"), file_name), "wb") as f:
                torch.save(data, f)
        elif extension == "plt":
            data.savefig(os.path.join(self.config.get("experiment_root_path"), file_name))
        else:
            raise NotImplementedError(f"Extension {extension} not implemented.")
        self.log(f"Successfully stored data in file '{os.path.join(self.config.get('experiment_root_path'), file_name)}'.")

    def load(
            self,
            file_name: str,
            extension: str = "pkl"
    ) -> Any:
        """
        Loads the data from a file.
        Returns None if the file is not found.

        Args:
            file_name (str):
                The name of the file where to load the data.
            extension (str, optional):
                The extension of the file. Defaults to "pkl".
        """

        self.log(f"Trying to load data from file '{file_name}' with extension '{extension}'.")
        try:
            if extension == "pkl":
                with open(os.path.join(self.config.get("experiment_root_path"), file_name), "rb") as f:
                    data = pkl.load(f)
            elif extension == "pt":
                with open(os.path.join(self.config.get("experiment_root_path"), file_name), "rb") as f:
                    data = torch.load(f, weights_only=False)
            else:
                raise NotImplementedError(f"Extension {extension} not implemented.")
            self.log(f"Successfully loaded data from file '{os.path.join(self.config.get('experiment_root_path'), file_name)}'.")
        except FileNotFoundError:
            self.log(f"File '{os.path.join(self.config.get('experiment_root_path'), file_name)}' not found.")
            data = None
        except RuntimeError as e:
            self.log(f"RuntimeError while loading model: {e}")
            data = None
        except Exception as e:
            self.log(f"Unexpected error while loading '{file_name}': {e}")
            data = None

        return data

    def delete(
            self,
            file_name: str
    ) -> None:
        """
        Deletes the file.

        Args:
            file_name (str):
                The name of the file to delete.
        """

        if self.exists_file(file_name):
            os.remove(os.path.join(self.config.get("experiment_root_path"), file_name))
            self.log(f"File '{file_name}' successfully deleted.")
        else:
            self.log(f"File '{file_name}' not found. It was not deleted.")

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

    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results obtained from the experiment.
        The performed operations will depend on the specific subclass of GeneralPurposeExperiment.
        """

        pass

    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results obtained from the experiment.
        The performed operations will depend on the specific subclass of GeneralPurposeExperiment.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        pass
