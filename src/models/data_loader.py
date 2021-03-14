from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import tensorflow as tf

from transformers import DistilBertTokenizer
from src.data.read_parallel import read_parallel_local
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime


def createDeepLegisDataFrame(config, reduce_by_factor=None, random_state=1):
        """
        Create the full dataset from the preprepared ml_data.csv
        """

        # Pre-wrangled metadata
        df = pd.read_csv(config.project_root + "references/derived/ml_data.csv", encoding="latin1", parse_dates=True)
        df.id = df.id.astype(int)


        # Encode all the labels before (potentially) reducing the dataset.
        sc_id_encoder = LabelEncoder()
        df['sc_id_cat'] = sc_id_encoder.fit_transform(df['sc_id'])    
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
                
        df = df.reset_index(drop=True)
        return df, sc_id_encoder

class legislationDataset(ABC):
    """Abstract Class for data loading for all the DeepLegis variations"""

    def __init__(self, config, testing=False):

        self.config = config
        self.testing = testing
        self.train_test_ratio = 0.91
        self.train_valid_ratio = 0.90

    @abstractmethod
    def to_feature(self):
        pass

    @abstractmethod
    def select_vars(self):
        pass

    @abstractmethod
    def to_feature_map(self):
        pass

    def create_batch_stream(self, df):
        
        print(df.head())

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
            for text, label, pl in train_data1.take(1):
                print(text)
                print(label)
                print(pl)

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

    def to_feature(self, text, label, partisan_lean):
  
        output = self.config.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", \
            truncation=True, padding='max_length', max_length=self.config.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'), \
                tf.cast(partisan_lean, 'float32'))

    def to_feature_map(self, text, label, partisan_lean):
        input_ids, label_id, partisan_lean  \
           = tf.py_function(self.to_feature, [text, label, partisan_lean], \
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

        return tf.data.Dataset.from_tensor_slices((df['text'].values, 
                                                   df['passed'].values, 
                                                   df['partisan_lean'].values))

class legislationDatasetText(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
    
    def to_feature(self, text, label):
  
        output = self.config.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", \
            truncation=True, padding='max_length', max_length=self.config.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'))

    def to_feature_map(self, text, label):
        input_ids, label_id = tf.py_function(self.to_feature, [text, label], \
            Tout = [tf.int32, tf.int32])
    
        input_ids.set_shape([self.config.max_length])
        label_id.set_shape([])
    
        x = {
            'input_ids': input_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['text'].values, df['passed'].values))

class legislationDatasetAll(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, text, label, partisan_lean, version_number):
  
        output = self.config.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", \
             truncation=True, padding='max_length', max_length=self.config.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'), \
               tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, text, label, partisan_lean, version_number, sc_id):
        input_ids, label_id, partisan_lean, version_number  \
           = tf.py_function(self.to_feature, [text, label, partisan_lean, version_number], \
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

        return tf.data.Dataset.from_tensor_slices((df['text'].values, 
                                                   df['passed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

class legislationDatasetRevCat(legislationDataset):

    def __init__(self, config):
        super().__init__(config)

    def to_feature(self, text, label,  version_number):
  
        output = self.config.tokenizer(text.numpy().decode('ascii'), \
            return_tensors="tf", truncation=True, padding='max_length', \
            max_length=self.config.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'),  \
                tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.config.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, text, label,  version_number, sc_id):
        input_ids, label_id,  version_number  \
           = tf.py_function(self.to_feature, [text, label, version_number], \
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

        return tf.data.Dataset.from_tensor_slices((df['text'].values, 
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

