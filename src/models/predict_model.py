
from src.models.configurationClasses import deepLegisConfig
from catboost import CatBoostClassifier
from transformers import BertTokenizer
from src.models.deeplegis import *
from src.models.data_loader import *
from src.data.data_downloader import data_downloader
from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report, confusion_matrix
import pandas as pd


def make_predictions_from_sqs(df):
    """
    Take the inputs from the df that need to be predicted and do:
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

    
    dd = data_downloader("data")
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

    # Use the same encoder from training.
    label_encoder = pickle.load( open( "models/encoder_production.pkl", "rb" ) ) 
    df['sc_id_cat'] = label_encoder.transform(df['sc_id'])

    prod_model = DeepLegisCatboost()
    return prod_model.predict_from_df_prod(df)


class DeepLegisCatboost():

    def __init__(self):
        self.label_encoder = pickle.load( open( "models/encoder_production.pkl", "rb" ) ) 


    def create_hidden_states(self, df):

        config = deepLegisConfig("distilbert_feature_extractor_128.json")
        deep_legis_model = config.model_class(config) 

        # Batch the data
        deep_legis_model.batch_df(df, n_sc_id_classes=len(self.label_encoder.classes_), only_full=True)

        # Load the transformer
        #deep_legis_model.deep_legis_model = tf.keras.models.load_model('models/transformer_production')
        deep_legis_model.build()

        # Do prediction with the transformer on the full dataset.
        hidden_states = deep_legis_model.deep_legis_model.predict(deep_legis_model.full_batches)

        return pd.DataFrame(hidden_states)

    def train_catboost(self, df):


        hidden_states = self.create_hidden_states(df)
        print("Created hiddgen states.")

        # Combine the metadata with the transformer output
        metadata_df = df[['sc_id_cat', 'version_number', 'partisan_lean']]
        metadata_df.reset_index(drop=True, inplace=True)
        feature_extractor_df = pd.concat([metadata_df, hidden_states], axis=1)

        # Train the Classifier.
        model = CatBoostClassifier(
            custom_loss=['Accuracy'],
            random_seed=42,
            logging_level='Silent',
            depth=10
        )
        categorical_features_indices = [0]
        (X_train, X_test, Y_train, Y_test) = \
            train_test_split(feature_extractor_df, df[['passed']], test_size=0.1, random_state=0)

        model.fit(
            X_train, Y_train,
            cat_features=categorical_features_indices,
            eval_set=(X_test, Y_test),
            plot=False
        )

        
        model.save_model('models/catboost.production')

        pred = model.predict_proba(X_test)[:,1]
        truth = Y_test.values
        print(confusion_matrix(truth, pred>0.5))
        print(f"AUROC:{roc_auc_score(truth, pred)}")


    def predict_from_df_prod(self, df):
        """
        Production prediction code.
        """
    
        hidden_states = self.create_hidden_states(df)

        # Combine the metadata with the transformer output
        metadata_df = deep_legis_model.df[['sc_id_cat', 'version_number', 'partisan_lean']]
        metadata_df.reset_index(drop=True, inplace=True)
        feature_extractor_df = pd.concat([metadata_df, pd.DataFrame(hidden_states)], axis=1)

        # Run the Catboost Classifier.
        catboost_model = CatBoostClassifier()
        catboost_model.load_model('models/catboost_production')
        preds_cat = catboost_model.predict_proba(feature_extractor_df)[:,1]
        return preds_cat
