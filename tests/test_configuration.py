import unittest
from src.models.configurationClasses import  deepLegisConfig
import os

class TestConfigurationCode(unittest.TestCase):

    def test_all_configs(self):

        print("Testing all configs in ./src/configs/.")
        models_to_test = os.listdir('src/configs/')
        for file in models_to_test:

            config = deepLegisConfig(file)


if __name__ == '__main__':

    unittest.main()