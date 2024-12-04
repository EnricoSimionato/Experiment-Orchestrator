import copy
from enum import Enum
import yaml
import os
from typing import Any

from exporch.utils.print_utils.print_utils import Verbose


class ExperimentStatus(Enum):
    NOT_STARTED = "Not started"
    RUNNING = "Running"
    COMPLETED = "Completed"
    STOPPED = "Stopped"


class Config:
    """
    Config class to store all the configuration parameters of an experiment about training a deep model using Pytorch
    Lightning.

    Args:
        path_to_config (str):
            The path to the configuration file.

    Attributes:
        All the keys in the config file are added as attributes to the class.
        verbose (Verbose):
            The verbosity level of the configuration.
    """

    def __init__(
            self,
            path_to_config: str = None
    ) -> None:
        # Checking if the path to the configuration file is provided and exists
        if path_to_config is None:
            raise Exception("The path to the configuration file cannot be None.")
        if not os.path.exists(path_to_config):
            raise Exception(f"Path '{path_to_config}' does not exist.")

        # Loading the configuration file
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)

        # Checking if the path to storage is provided and exists
        if "path_to_storage" not in config.keys():
            raise Exception("The path to storage must be provided in the configuration.")
        if not os.path.exists(config["path_to_storage"]):
            raise Exception(f"Path '{config['path_to_storage']}' does not exist.")

        # Setting the verbosity level
        try:
            self.verbose = Verbose(config.pop("verbose"))
        except KeyError:
            self.verbose = Verbose(0)

        self.__dict__.update(config)

    @staticmethod
    def convert_to_config(
            configuration_dict: dict
    ) -> 'Config':
        """
        Converts a dictionary to a Config object.

        Args
            configuration_dict (dict):
                The dictionary to be converted.

        Returns:
            Config:
                The Config object.
        """

        if "path_to_storage" not in configuration_dict.keys():
            configuration_dict["path_to_storage"] = os.path.join(os.getcwd())

        with open("tmp.yaml", "w") as f:
            yaml.dump(configuration_dict, f)

        configuration = Config("tmp.yaml")
        os.remove("tmp.yaml")

        return configuration

    def contains(
            self,
            key: str
    ) -> bool:
        """
        Checks if the specified key is present in the configuration.

        Args:
            key (str):
                The key to be checked.
            **kwargs:
                Additional keyword arguments.

        Returns:
            bool:
                True if the key is present, False otherwise.
        """

        return key in self.__dict__.keys()

    def get(
            self,
            key: str,
            default: Any = None
    ) -> Any:
        """
        Returns the value of the specified key.

        Args:
            key (str):
                The key whose value is to be returned.
            default (Any):
                The default value to be returned if the key is not present.

        Returns:
            Any:
                The value of the specified key.
        """

        if default is not None:
            if not self.contains(key):
                return default
        return self.__dict__[key]

    def get_dict(
            self,
            keys: list
    ) -> dict:
        """
        Returns a dictionary containing the values of the specified keys.

        Args:
            keys (list):
                A list of keys whose values are to be returned.

        Returns:
            dict:
                A dictionary containing the values of the specified keys.
        """

        for key in keys:
            if not self.contains(key):
                print(f"The key '{key}' is not present in the configuration.")

        filtered_dict = {
            key: self.get(key) for key in keys if self.contains(key)
        }

        return filtered_dict

    def get_paths(
            self
    ) -> dict:
        """
        Returns the paths of the experiment.

        Returns:
            dict:
                A dictionary containing the paths of the experiment.
        """

        paths = {}

        for key in self.__dict__.keys():
            if "path" in key:
                paths[key] = self.__dict__[key]

        return paths

    def get_verbose(
            self,
    ) -> Verbose:
        """
        Returns the verbosity level of the configuration.

        Returns:
            Verbose:
                The verbosity level of the configuration.
        """

        return self.verbose

    def to_dict(
            self
    ) -> dict:
        """
        Returns a copy of the configuration as a dictionary.

        Returns:
            dict:
                The configuration as a dictionary.
        """

        config_dictionary_representation = copy.deepcopy(self.__dict__)
        config_dictionary_representation.pop("verbose")

        return config_dictionary_representation

    def check_mandatory_keys(
            self,
            mandatory_keys: list
    ) -> None:
        """
        Checks if the configuration file contains the mandatory keys.

        Args:
            mandatory_keys (list):
                A list of mandatory keys that must be present in the configuration file.
        """

        for key in mandatory_keys:
            if not self.contains(key):
                raise Exception(f"The configuration file must contain the key '{key}'.")

    def set(
            self,
            key: str,
            value: Any
    ):
        """
        Sets the value of the specified key.

        Args:
            key (str):
                The key whose value is to be set.
            value (Any):
                The value to be set for the specified key.
        """

        self.__dict__[key] = value

    def update(
            self,
            config: dict
    ) -> None:
        """
        Updates or inserts the configuration parameters according to the provided dictionary.

        Args:
            config (dict):
                A dictionary containing the configuration parameters to be updated or inserted.
        """

        self.__dict__.update(config)

    def create_directory(
            self,
            base_path: str,
            directory_name: str,
            key_in_configuration: str = None
    ) -> None:
        """
        Creates a directory for the experiment.

        Args:
            base_path (str):
                The base path for the directory, where the directory will be created.
            directory_name (str):
                The name of the directory to be created.
            key_in_configuration (str):
                The key in the configuration to store the path to the directory.
        """

        if not os.path.exists(base_path):
            raise Exception(f"The base path '{base_path}' does not exist.")

        if key_in_configuration is None:
            key_in_configuration = "path_to_" + directory_name

        path_to_directory = os.path.join(base_path, directory_name)

        if not os.path.exists(path_to_directory):
            os.makedirs(path_to_directory)

        if self.contains(key_in_configuration):
            raise Exception(f"The key '{key_in_configuration}' already exists in the configuration.")

        self.set(key_in_configuration, path_to_directory)

    def store(
            self,
            path: str = None
    ) -> None:
        """
        Stores the configuration file.

        Args:
            path (str):
                The path to store the configuration file.
        """

        if path is None:
            path = self.get("path_to_experiment")

        path = os.path.join(path, "config.yaml")
        with open(path, "w") as file:
            config_dictionary_representation = copy.deepcopy(self.__dict__)
            config_dictionary_representation.pop("verbose")
            yaml.dump(config_dictionary_representation, file, default_flow_style=False)

    def __str__(
            self
    ) -> str:
        """
        Returns the string representation of the configuration.

        Returns:
            str:
                The string representation of the configuration.
        """

        config_string = ""
        for key in self.__dict__.keys():
            config_string += f"{key}: {self.__dict__[key]}\n"

        return config_string
