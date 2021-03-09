from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import tensorflow as tf

class legislationDataset(ABC):
    """Abstract Class for data loading for all the DeepLegis variations"""

    def __init__(self, config):

        self.max_length = config['max_length']
        self.train_batch_size = config['train_batch_size']
        self.testing = config['testing']
        self.train_test_ratio = config['train_test_ratio']
        self.train_valid_ratio = config['train_valid_ratio']

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

        df_train_full, df_test = train_test_split(df, train_size = self.train_test_ratio, random_state = 1, stratify = df.signed.values)
        df_train, df_valid = train_test_split(df_train_full, train_size = self.train_valid_ratio, random_state = 1, stratify = df_train_full.signed.values)
        print(f"Training size: {df_train.shape}")
        print(f"Validation size: {df_valid.shape}")
        print(f"Test size: {df_test.shape}")


        train_data1 = self.select_vars(df_train)
        val_data1   = self.select_vars(df_valid)
        test_data1  = self.select_vars(df_test)

        if self.testing:
            for text, label, pl in train_data1.take(1):
                print(text)
                print(label)
                print(pl)

        train_data = (train_data1.map(self.to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .shuffle(1000)
             .batch(self.train_batch_size, drop_remainder=True)
             .prefetch(tf.data.experimental.AUTOTUNE) 
             )

        val_data = (val_data1.map(self.to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(self.train_batch_size, drop_remainder=True)
             .prefetch(tf.data.experimental.AUTOTUNE) 
            )

        test_data = (test_data1.map(self.to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(self.train_batch_size, drop_remainder=True)
             .prefetch(tf.data.experimental.AUTOTUNE) 
            )

        return train_data, val_data, test_data

class legislationDatasetPartisanLean(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = config['tokenizer']

    def to_feature(self, text, label, partisan_lean):
  
        output = self.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", truncation=True, padding='max_length', max_length=self.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'), tf.cast(partisan_lean, 'float32'))

    def to_feature_map(self, text, label, partisan_lean):
        input_ids, label_id, partisan_lean  \
           = tf.py_function(self.to_feature, [text, label, partisan_lean], Tout = [tf.int32, tf.int32, tf.float32])
    
        input_ids.set_shape([self.max_length])
        label_id.set_shape([])
        partisan_lean.set_shape([])
    
        x = {
            'input_ids': input_ids,
            'partisan_lean': partisan_lean
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['text'].values, df['signed'].values, df['partisan_lean'].values))


class legislationDatasetText(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = config['tokenizer']
    
    def to_feature(self, text, label):
  
        output = self.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", truncation=True, padding='max_length', max_length=self.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'))

    def to_feature_map(self, text, label):
        input_ids, label_id = tf.py_function(self.to_feature, [text, label], Tout = [tf.int32, tf.int32])
    
        input_ids.set_shape([self.max_length])
        label_id.set_shape([])
    
        x = {
            'input_ids': input_ids
        }
    
        return (x, label_id)

    def select_vars(self, df):

        return tf.data.Dataset.from_tensor_slices((df['text'].values, df['signed'].values))

class legislationDatasetAll(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
        self.n_sc_id_classes = config['n_sc_id_classes']
        self.tokenizer = config['tokenizer']

    def to_feature(self, text, label, partisan_lean, version_number):
  
        output = self.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", truncation=True, padding='max_length', max_length=self.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'), tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, text, label, partisan_lean, version_number, sc_id):
        input_ids, label_id, partisan_lean, version_number  \
           = tf.py_function(self.to_feature, [text, label, partisan_lean, version_number], Tout = [tf.int32, tf.int32, tf.float32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])

        input_ids.set_shape([self.max_length])
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
                                                   df['signed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

class legislationDatasetRevCat(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
        self.n_sc_id_classes = config['n_sc_id_classes']
        self.tokenizer = config['tokenizer']

    def to_feature(self, text, label,  version_number):
  
        output = self.tokenizer(text.numpy().decode('ascii'), return_tensors="tf", truncation=True, padding='max_length', max_length=self.max_length)
    
        return (tf.squeeze(output['input_ids'],0), tf.cast(label, 'int32'),  tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.n_sc_id_classes, dtype='float32')

    def to_feature_map(self, text, label,  version_number, sc_id):
        input_ids, label_id,  version_number  \
           = tf.py_function(self.to_feature, [text, label, version_number], Tout = [tf.int32, tf.int32, tf.float32])
    
        sc_ids = tf.py_function(self.sc_one_hot, [sc_id], Tout=[tf.float32])
        sc_ids = sc_ids[0]

        input_ids.set_shape([self.max_length])
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
                                                   df['signed'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))


class legislationDatasetNoText(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
        self.n_sc_id_classes = config['n_sc_id_classes']

    def to_feature(self, label, partisan_lean, version_number):

    
        return (tf.cast(label, 'int32'), tf.cast(partisan_lean, 'float32'), tf.cast(version_number, 'float32'))

    def sc_one_hot(self, sc_id):

        return tf.one_hot(sc_id, self.n_sc_id_classes, dtype='float32')

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

        return tf.data.Dataset.from_tensor_slices((df['signed'].values, 
                                                   df['partisan_lean'].values, 
                                                   df['version_number'].values, 
                                                   df['sc_id_cat'].values))

