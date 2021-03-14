import unittest

# Base directory needs to be on $PYTHONPATH
from src.models.deeplegis import *
from src.models.data_loader import *
from src.models.configurationClasses import  deepLegisConfig

import pprint
import os
pp = pprint.PrettyPrinter() # for the config


class TestModels(unittest.TestCase):


    def test_models(self):
        """
        Test that every configuation successfully runs for at least 
        one epoch with 1/1000th of the dataset.
        """

        print("Testing all models!")
        models_to_test = os.listdir('src/configs/')
        for file in models_to_test[:1]:
            self.individual_model_test(file)

    def test_single_model(self):
        """
        Test that every configuation successfully runs for at least 
        one epoch with 1/1000th of the dataset.
        """

        self.individual_model_test('distilbert_128.json')

    def individual_model_test(self, json_file):
        """
        Run every model with 1000 times less data and for one epoch
        """

        datavol = os.environ['DATA_VOL']
        config = deepLegisConfig(json_file, output_root= datavol + "test_output/")
        config.epochs = 1

        # The class code is in the config, specified by the json
        deep_legis_model = config.model_class(config)  

        print("Import and process the dataset")
        deep_legis_model.load_data(reduce_by_factor=1000)

        pp.pprint(vars(config))

        print("Build the model and show the strucutre.")
        deep_legis_model.build()
        deep_legis_model.deep_legis_model.summary()

        print("Train the model!")
        deep_legis_model.train()

        print("Evaluation on the Test set:")
        deep_legis_model.evaluate()

        deep_legis_model.deep_legis_model.save(config.model_location)

        print("Cache predictions on all observations for later use.")
        deep_legis_model.full_dataset_prediction()



if __name__ == '__main__':

    unittest.main()