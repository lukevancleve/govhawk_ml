# Run script for no_text
import pprint
from src.models.deeplegis import *
from src.models.data_loader import *
from src.models.configurationClasses import  deepLegisConfig

pp = pprint.PrettyPrinter() # for the config

config = deepLegisConfig("no_text.json")
config.build_from_scratch = True
config.epochs = 1

# The class code is in the config, specified by the json
deep_legis_model = config.model_class(config)  

print("Import and process the dataset")
deep_legis_model.load_data()

pp.pprint(vars(config))

print("Build the model and show the strucutre.")
deep_legis_model.build()
deep_legis_model.deep_legis_model.summary()

print("Train the model!")
deep_legis_model.train()

print("Evaluation on the Test set:")
deep_legis_model.evaluate()

print("Cache predictions on all observations for later use.")
deep_legis_model.full_dataset_prediction()