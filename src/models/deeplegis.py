import tensorflow as tf
from transformers import LongformerTokenizer
from sklearn.model_selection import train_test_split
from transformers import TFLongformerModel, TFLongformerForSequenceClassification, TFBertForSequenceClassification

class legislationDataset():

    def __init__(self, config):

        self.max_length = config['max_length']
        self.train_batch_size = config['train_batch_size']
        self.testing = config['testing']
        self.train_test_ratio = config['train_test_ratio']
        self.train_valid_ratio = config['train_valid_ratio']
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



class legislationDatasetText(legislationDataset):

    def __init__(self, config):
        super().__init__(config)
    
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





def deep_legis_text(config):

    model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
    ids = tf.keras.Input((config['max_length']), dtype=tf.int32, name='input_ids')
    x = model.longformer(ids) # Get the main Layer
    x = x['last_hidden_state'][:,0,:]
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(700, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dl_model = tf.keras.Model(inputs={"input_ids":ids}, outputs=[x])

    return dl_model


def deep_legis_pl(config):

    model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
    ids = tf.keras.Input((config['max_length']), dtype=tf.int32, name='input_ids')
    pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
    x = model.longformer(ids) # Get the main Layer
    x = x['last_hidden_state'][:,0,:]
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.concat([x, pl], axis=-1)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dl_model = tf.keras.Model(inputs={"input_ids":ids,"partisan_lean":pl}, outputs=[x])

    return dl_model


def deep_legis_all(config):

    model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
    ids = tf.keras.Input((config['max_length']), dtype=tf.int32, name='input_ids')
    pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
    vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
    cat = tf.keras.Input((config['n_sc_id_classes'] ), dtype=tf.float32, name='sc_id')
    print(cat.shape)
    #cat = tf.keras.Input(shape=None, dtype=tf.float32, name='sc_id')

    #cat = tf.cast(cat, 'float32')
    #cat = tf.squeeze(cat,1)

    x = model.longformer(ids) # Get the main Layer
    x = x['last_hidden_state'][:,0,:]
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.concat([x, pl, vn, cat], axis=-1)
    x = tf.keras.layers.Dense(700, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dl_model = tf.keras.Model(inputs={"input_ids":ids,"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])

    return dl_model

def deep_legis_vn_cat(config):

    model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
    ids = tf.keras.Input((config['max_length']), dtype=tf.int32, name='input_ids')
    vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
    cat = tf.keras.Input((config['n_sc_id_classes'], ), dtype=tf.float32, name='sc_id')

    x = model.longformer(ids) # Get the main Layer
    x = x['last_hidden_state'][:,0,:]
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.concat([x, vn, cat], axis=-1)
    x = tf.keras.layers.Dense(700, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn, "sc_id": cat}, outputs=[x])

    return dl_model


#########################################################

def deep_legis_no_text(config):

    pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
    vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
    cat = tf.keras.Input((config['n_sc_id_classes'] ), dtype=tf.float32, name='sc_id')

    print(cat.shape)
    x = tf.concat([pl, vn, cat], axis=-1)
 #   x = tf.concat([pl, vn], axis=-1)
    x = tf.keras.layers.Dense(700, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dl_model = tf.keras.Model(inputs={"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])

    return dl_model



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


def bert_all(config):

    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    ids = tf.keras.Input((config['max_length']), dtype=tf.int32, name='input_ids')
    pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
    vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
    cat = tf.keras.Input((config['n_sc_id_classes'] ), dtype=tf.float32, name='sc_id')
    # print(cat.shape)
    #cat = tf.keras.Input(shape=None, dtype=tf.float32, name='sc_id')

    #cat = tf.cast(cat, 'float32')
    #cat = tf.squeeze(cat,1)

    x = model.bert(ids) # Get the main Layer
    x = x['last_hidden_state'][:,0,:]
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.concat([x, pl, vn, cat], axis=-1)
    x = tf.keras.layers.Dense(700, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dl_model = tf.keras.Model(inputs={"input_ids":ids,"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])

    return dl_model