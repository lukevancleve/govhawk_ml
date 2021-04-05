from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import tensorflow as tf

from transformers import DistilBertTokenizer
from src.data.read_parallel import read_parallel_local
import sys
import os
import pandas as pd
import datetime
import time
from pandarallel import pandarallel
import pickle

def createDeepLegisDataFrame(config, read_cached=True, reduce_by_factor=None, random_state=1, project_root="./"):
        """
        Create the full dataset from the preprepared ml_data.csv
        """

        if read_cached:
            # Cached has the text already tokenized.

            tic = time.perf_counter()
            if config.max_length == 128:
                pickle_file = config.data_vol + "preprocessed_df_128.pkl"
                df = pd.read_pickle(pickle_file)
            elif config.max_length == 512:
                pickle_file = config.data_vol + "preprocessed_df_512.pkl"
                df = pd.read_pickle(pickle_file)
            elif config.max_length == 4096:
                pickle_file = config.data_vol + "preprocessed_df_longformer_4096.pkl"
                df = pd.read_pickle(pickle_file)
            else:
                raise "Invalid max_length for pickle file."
            toc = time.perf_counter()
            print(f"Loading pickle file ({pickle_file}) took {(toc-tic)/60.0} min -  {toc - tic:0.4f} seconds")

            print(f"Original number of examples: {len(df)}")
            if reduce_by_factor is not None:
                df = df.sample(n=int(len(df)/reduce_by_factor), random_state=random_state) 
                print(f"Reduced number of examples:  {len(df)}")

        else:

            # Pre-wrangled metadata
            df = pd.read_csv(config.project_root + "references/derived/ml_data.csv", encoding="latin1", parse_dates=True)
            df.id = df.id.astype(int)

            print(f"Original number of examples: {len(df)}")
            if reduce_by_factor is not None:
                df = df.sample(n=int(len(df)/reduce_by_factor), random_state=random_state) 
                print(f"Reduced number of examples:  {len(df)}")

            # Skip the text if we're not using it.
            if config.tokenizer is not None:
                clean_path = config.data_vol + "clean/"
                if not os.path.exists(clean_path):
                    raise 'No such directory: ' + clean_path
                print(f"Loading {len(df)} text files")
                df['text'] = read_parallel_local(df['id'], config.data_vol + "clean/")
                na_text_rows = df.text.isna()
                if sum(na_text_rows) > 0:
                    print(f"WARNING! Removing {sum(na_text_rows)} rows becase the text value is None.")
                    df = df[~na_text_rows]

                # Use all the CPU cores to tokenize the text.
                pandarallel.initialize()
                tic = time.perf_counter()

                # Used to pass back an element of the dict
                def tokenizer_wrapper(text):
                    d = config.tokenizer(text, truncation=True, padding='max_length', max_length=config.max_length)
                    return d['input_ids']

                df['tokens'] = df.text.parallel_apply( tokenizer_wrapper)
                toc = time.perf_counter()
                print(f"Tokenized in {(toc-tic)/60.0} min -  {toc - tic:0.4f} seconds")

                df = df.reset_index(drop=True)

        # Encode all the labels before (potentially) reducing the dataset.
        sc_id_encoder = pickle.load( open( project_root + "models/encoder_production.pkl", "rb" ) )
        #df['sc_id_cat'] = sc_id_encoder.fit_transform(df['sc_id'])    

        return df, sc_id_encoder

class legislationDataset(ABC):
    """Abstract Class for data loading for all the DeepLegis variations"""

    def __init__(self, config):

        self.config = config
        self.testing = False
        self.train_test_ratio = 0.91
        self.train_valid_ratio = 0.90

    @abstractmethod
    def to_feature(self):
        pass

    @abstractmethod
    def select_vars(self, df):
        pass

    @abstractmethod
    def to_feature_map(self):
        pass

    def create_batch_stream(self, df):
        
        print(df.head())

        if self.config.only_full:
            # For use when not expecting a pre-generated train/val/test split.
            full_data1  = self.select_vars(df)
            full_data = (full_data1.map(self.to_feature_map, \
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(self.config.batch_size)
             .prefetch(tf.data.experimental.AUTOTUNE) 
            )
            return None, None, None, full_data, None

        df_train_full, df_test = train_test_split(df, train_size = \
            self.train_test_ratio, random_state = 1, stratify = df.passed.values)
        df_train, df_valid = train_test_split(df_train_full, train_size = \
            self.train_valid_ratio, random_state = 1, stratify = df_train_full.passed.values)

        # Save which obersvaitons were in train/val/test:
        train = df_train[["id"]]
        train['split'] = "train"
        test = df_test[['id']]
        test['split'] = "test"
        val = df_valid[['id']]
        val['split'] = "val"
        split_data = pd.concat([train,val,test])

        print(f"Full size: {df.shape}")
        print(f"Training size: {df_train.shape}")
        print(f"Validation size: {df_valid.shape}")
        print(f"Test size: {df_test.shape}")

        full_data1  = self.select_vars(df)
        train_data1 = self.select_vars(df_train)
        val_data1   = self.select_vars(df_valid)
        test_data1  = self.select_vars(df_test)

        if self.testing:
            for elem in train_data1.take(1):
                print(elem)
    

        train_data = (train_data1.map(self.to_feature_map, \
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .shuffle(250)
             .batch(self.config.batch_size)
             .prefetch(tf.data.experimental.AUTOTUNE) 
             )

        val_data = (val_data1.map(self.to_feature_map, \
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(self.config.batch_size)
             .prefetch(tf.data.experimental.AUTOTUNE) 
            )

        test_data = (test_data1.map(self.to_feature_map, \
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(self.config.batch_size)
             .prefetch(tf.data.experimental.AUTOTUNE) 
            )

        full_data = (full_data1.map(self.to_feature_map, \
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(self.config.batch_size)
             .prefetch(tf.data.experimental.AUTOTUNE) 
            )

        return train_data, val_data, test_data, full_data, split_data

class legislationDatasetPartisanLean(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, tokens, label, partisan_lean):
  
        return (tf.cast(tokens, 'int32'), tf.cast(label, 'int32'), \
                tf.cast(partisan_lean, 'float32'))

    def to_feature_map(self, tokens, label, partisan_lean):
        input_ids, label_id, partisan_lean  \
           = tf.py_function(self.to_feature, [tokens, label, partisan_lean], \
                Tout = [tf.int32, tf.int32, tf.float32])
    
        input_ids.set_shape([self.config.max_length])
        label_id.set_shape([])
        partisan_lean.set_shape([])
    
        x = {
            'input_ids': input_ids,
            'partisan_lean': partisan_lean
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['tokens'].values, 
                                                   df['passed'].values, 
                                                   df['partisan_lean'].values))

class legislationDatasetText(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
    
    def to_feature(self, tokens, label):
  
        return (tf.cast(tokens, 'int32'), tf.cast(label, 'int32'))

    def to_feature_map(self, tokens, label):
        input_ids, label_id = tf.py_function(self.to_feature, [tokens, label], \
            Tout = [tf.int32, tf.int32])
    
        input_ids.set_shape([self.config.max_length])
        label_id.set_shape([])
    
        x = {
            'input_ids': input_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['tokens'].to_list(), df['passed'].values))

class legislationDatasetAll(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, tokens, label, partisan_lean, version_number):
  
        return (tf.cast(tokens, 'int32'), tf.cast(label, 'int32'), \
               tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, tokens, label, partisan_lean, version_number, sc_id):
        input_ids, label_id, partisan_lean, version_number  \
           = tf.py_function(self.to_feature, [tokens, label, partisan_lean, version_number], \
               Tout = [tf.int32, tf.int32, tf.float32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])

        input_ids.set_shape([self.config.max_length])
        label_id.set_shape([])
        partisan_lean.set_shape([])
        version_number.set_shape([])
        sc_ids = sc_ids[0]
    
        x = {
            'input_ids': input_ids,
            'partisan_lean': partisan_lean,
            'version_number': version_number,
            'sc_id': sc_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['tokens'].to_list(), 
                                                   df['passed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

class legislationDatasetRevCat(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, tokens, label,  version_number):
  
        return (tf.cast(tokens, 'int32'), tf.cast(label, 'int32'),  \
                tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, tokens, label,  version_number, sc_id):
        input_ids, label_id,  version_number  \
           = tf.py_function(self.to_feature, [tokens, label, version_number], \
               Tout = [tf.int32, tf.int32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])
        sc_ids = sc_ids[0]

        input_ids.set_shape([self.config.max_length])
        label_id.set_shape([])
        version_number.set_shape([])
    
        x = {
            'input_ids': input_ids,
            'version_number': version_number,
            'sc_id': sc_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['tokens'].values, 
                                                   df['passed'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

class legislationDatasetNoText(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, label, partisan_lean, version_number):

    
        return (tf.cast(label, 'int32'), tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self,  label, partisan_lean, version_number, sc_id):

        label_id, partisan_lean, version_number  \
           = tf.py_function(self.to_feature, [label, partisan_lean, version_number], Tout = [tf.int32, tf.float32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])
        sc_ids = sc_ids[0]
        
        label_id.set_shape([])
        partisan_lean.set_shape([])
        version_number.set_shape([])
    
        x = {
            'partisan_lean': partisan_lean,
            'version_number': version_number,
            'sc_id': sc_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['passed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

class legislationDatasetAllWithAttention(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, tokens, attention_mask, label, partisan_lean, version_number):
  
        return (tf.cast(tokens, 'int32'), tf.cast(attention_mask, 'int32'), tf.cast(label, 'int32'), \
               tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, tokens, attention_mask, label, partisan_lean, version_number, sc_id):
        input_ids, attention_mask, label_id, partisan_lean, version_number  \
           = tf.py_function(self.to_feature, [tokens, attention_mask, label, partisan_lean, version_number], \
               Tout = [tf.int32, tf.int32, tf.int32, tf.float32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])

        input_ids.set_shape([self.config.max_length])
        attention_mask.set_shape([self.config.max_length])
        label_id.set_shape([])
        partisan_lean.set_shape([])
        version_number.set_shape([])
        sc_ids = sc_ids[0]
    
        x = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'partisan_lean': partisan_lean,
            'version_number': version_number,
            'sc_id': sc_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['tokens'].to_list(),
                                                   df['attention_mask'].to_list(), 
                                                   df['passed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

class legislationDatasetAllSigned(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, tokens, label, partisan_lean, version_number):
  
        return (tf.cast(tokens, 'int32'), tf.cast(label, 'int32'), \
               tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, tokens, label, partisan_lean, version_number, sc_id):
        input_ids, label_id, partisan_lean, version_number  \
           = tf.py_function(self.to_feature, [tokens, label, partisan_lean, version_number], \
               Tout = [tf.int32, tf.int32, tf.float32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])

        input_ids.set_shape([self.config.max_length])
        label_id.set_shape([])
        partisan_lean.set_shape([])
        version_number.set_shape([])
        sc_ids = sc_ids[0]
    
        x = {
            'input_ids': input_ids,
            'partisan_lean': partisan_lean,
            'version_number': version_number,
            'sc_id': sc_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['tokens'].to_list(), 
                                                   df['signed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))
