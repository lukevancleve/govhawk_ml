# Run script for no_text
import pprint
from src.models.deeplegis import *
from src.models.data_loader import *
from src.models.configurationClasses import  deepLegisConfig
pp = pprint.PrettyPrinter() # for the config

models_to_run = [
#    "no_text.json",
#    "distilbert_128.json",
    "distilbert_128_rev_cat.json",
    "distilbert_128_metadata_weights.json",
    "bert_128.json",
    "distilbert_512.json",
    "bert_512.json"
]
models_to_run = ["distilbert_128.json"]

def run_model(json_file):
    """
    """

    config = deepLegisConfig(json_file)

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

    print("Save the model.")
    deep_legis_model.deep_legis_model.save(config.model_location)

    print("Evaluation on the Test set:")
    deep_legis_model.evaluate()

    print("Cache predictions on all observations for later use.")
    deep_legis_model.full_dataset_prediction()

for model in models_to_run:
    run_model(model)