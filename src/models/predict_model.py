
from src.models.configurationClasses import deepLegisConfig
from catboost import CatBoostClassifier
from transformers import BertTokenizer
from src.models.deeplegis import *
from src.models.data_loader import *
from src.data.data_downloader import data_downloader

import pandas as pd


def make_predictions_from_sqs(df):
    """
    Take the inputs from sqs that need to be predicted and do:
    1. Massage into form the transformer want
    2. Take the output from the transformer and concat with the other metadata
    3. Run this output through catboost
    4. Return the predictions.
    """

    assert df is not None, "Caller should provide a pd.DataFrame"
    assert 'plain_url' in df.columns 
    assert 'session_id' in df.columns 
    assert 'chamber_id' in df.columns 
    assert 'version_number' in df.columns 
    assert 'bill_id' in df.columns 
    assert 'bill_version_id' in df.columns 
    assert 'partisan_lean' in df.columns 

    dd = data_downloader("data/tests/")
    # Data wrangle to pass to the model
    df['raw']   = df.plain_url.apply(dd.download_plain)
    df['text'] = df.raw.apply(dd.clean_text)
    df['sc_id'] = df['session_id'].astype(str) + "-" + df['chamber_id'].astype(str)
    df['version_number'] = df['version_number'].astype(int)
    df['bill_id'] = df['bill_id'].astype(int)
    df['bill_version_id'] = df['bill_version_id'].astype(int)
    df['partisan_lean'] = df['partisan_lean'].astype(float)

    # Dummy for the dataset batching functions
    df['passed'] = 1

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenizer_wrapper(text):
        d = tokenizer(text, truncation=True, padding='max_length', max_length=128)
        return d['input_ids']

    df['tokens'] = df.text.apply( tokenizer_wrapper)

    label_encoder = pickle.load( open( "models/encoder_production.pkl", "rb" ) ) 
    df['sc_id_cat'] = label_encoder.transform(df['sc_id'])



    config = deepLegisConfig("distilbert_feature_extractor_128.json")
    deep_legis_model = config.model_class(config) 




    deep_legis_model.batch_df(df, n_sc_id_classes=len(label_encoder.classes_), only_full=True)

    deep_legis_model.deep_legis_model = tf.keras.models.load_model('models/transformer_production')



    hidden_states = deep_legis_model.deep_legis_model.predict(deep_legis_model.full_batches)

    metadata_df = deep_legis_model.df[['sc_id_cat', 'version_number', 'partisan_lean']]
    metadata_df.reset_index(drop=True, inplace=True)

    feature_extractor_df = pd.concat([metadata_df, pd.DataFrame(hidden_states)], axis=1)


    catboost_model = CatBoostClassifier()
    catboost_model.load_model('models/catboost_production')
    preds_cat = catboost_model.predict_proba(feature_extractor_df)[:,1]
