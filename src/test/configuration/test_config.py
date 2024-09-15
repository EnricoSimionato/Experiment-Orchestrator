import os
import yaml

from exporch import Config

class TestConfig:

    def __init__(self):
        self.configuration = Config("../data/CONFIG.yaml")

    def test_init(self):
        print(self.configuration)
        assert True

    def test_store(self):
        self.configuration.store(self.configuration.get("path_to_storage"))
        with open(os.path.join(self.configuration.get("path_to_storage"), "config.yaml"), "r") as f:
            stored_configuration = yaml.safe_load(f)
        for el in stored_configuration:
            assert self.configuration.contains(el)
            assert self.configuration.get(el) == stored_configuration[el]
        assert True

a =TestConfig()
a.test_init()
a.test_store()