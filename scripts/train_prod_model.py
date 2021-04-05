#from src.models.deeplegis import *
from src.models.data_loader import createDeepLegisDataFrame
from src.models.configurationClasses import  deepLegisConfig
from src.models.predict_model import DeepLegisCatboost


config = deepLegisConfig("distilbert_feature_extractor_128.json")
df, encoder = createDeepLegisDataFrame(config, read_cached=True)

print(df.head())
print(df.shape)
prod_model = DeepLegisCatboost()
prod_model.train_catboost(df)