import yaml
import os
import shutil

from exporch import GeneralPurposeExperiment, NopGeneralPurposeExperiment
from exporch import GeneralPurposeExperimentFactory


example_path_to_config = "../data/CONFIG.yaml"
example_path_to_storage = "../results"

example_config = {
    "path_to_storage": example_path_to_storage,
    "experiment_type": "general_purpose_experiment",
    "model_id": "model_id",
    "version": "0",
    "attribute": "attribute",
    "nop": "nop"
}

def setup(
        path_to_config: str,
        path_to_storage: str,
        config_dict: dict
) -> None:
    with open(path_to_config, "w") as file:
        yaml.dump(example_config, file)
    if os.path.exists(os.path.join(path_to_storage, config_dict["model_id"])):
        shutil.rmtree(os.path.join(path_to_storage, config_dict["model_id"]))

class TestGeneralPurposeExperiment:
    def setup_method(
            self
    ) -> None:
        self.path_to_config = example_path_to_config
        self.path_to_storage = example_path_to_storage
        self.config_dict = example_config.copy()
        setup(self.path_to_config, self.path_to_storage, self.config_dict)

    def test_init_base(
            self
    ) -> None:
        _ = NopGeneralPurposeExperiment(self.path_to_config)
        assert True

    def test_get_mandatory_keys(
            self
    ) -> None:
        experiment = NopGeneralPurposeExperiment(self.path_to_config)
        assert experiment.get_mandatory_keys().sort() == (GeneralPurposeExperiment.mandatory_keys + NopGeneralPurposeExperiment.mandatory_keys).sort()

    def test_check_mandatory_keys(
            self
    ) -> None:
        for key in ["path_to_storage", "experiment_type", "model_id"]:
            example_config_copy = self.config_dict.copy()
            example_config_copy.pop(key)
            with open(self.path_to_config, "w") as file:
                yaml.dump(example_config_copy, file)
            try:
                _ = NopGeneralPurposeExperiment(self.path_to_config)
                assert False
            except Exception as e:
                assert True

    def test_launch_experiment(
            self
    ) -> None:
        experiment = NopGeneralPurposeExperiment(self.path_to_config)
        experiment.launch_experiment()
        assert os.path.exists(os.path.join(
            self.path_to_storage, self.config_dict["model_id"]))
        assert os.path.exists(os.path.join(
            self.path_to_storage, self.config_dict["model_id"], self.config_dict["experiment_type"]))
        assert os.path.exists(os.path.join(
            self.path_to_storage, self.config_dict["model_id"], self.config_dict["experiment_type"],
            "version_" + self.config_dict["version"]))
        assert os.path.exists(os.path.join(
            self.path_to_storage, self.config_dict["model_id"], self.config_dict["experiment_type"],
            "version_" + self.config_dict["version"], "logs.log"))


class TestGeneralPurposeExperimentFactory:
    @classmethod
    def setup_method(
            cls
    ) -> None:
        setup(example_path_to_config, example_path_to_storage, example_config)

    @classmethod
    def test_register(
        cls
    ) -> None:
        assert len(GeneralPurposeExperimentFactory.mapping.keys()) == 0
        GeneralPurposeExperimentFactory.register({"nop_general_purpose_experiment": NopGeneralPurposeExperiment})
        assert len(GeneralPurposeExperimentFactory.mapping.keys()) == 1
        assert list(GeneralPurposeExperimentFactory.mapping.keys())[0] == "nop_general_purpose_experiment"
        assert GeneralPurposeExperimentFactory.mapping["nop_general_purpose_experiment"] == NopGeneralPurposeExperiment

    @classmethod
    def test_create(
            cls
    ) -> None:
        GeneralPurposeExperimentFactory.register({"general_purpose_experiment": NopGeneralPurposeExperiment})
        experiment = GeneralPurposeExperimentFactory.create(example_path_to_config)
        assert isinstance(experiment, NopGeneralPurposeExperiment)

