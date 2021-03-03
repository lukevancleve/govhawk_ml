import tensorflow as tf
from transformers import LongformerTokenizer
from sklearn.model_selection import train_test_split

class legislationDataset():

    def __init__(self, config):

        self.max_length = config['max_length']
        self.train_batch_size = config['train_batch_size']
        self.testing = config['testing']
        self.train_test_ratio = config['train_test_ratio']
        self.train_valid_ratio = config['train_valid_ratio']
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


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

    def create_batch_stream(self, df):

        df_train_full, df_test = train_test_split(df, train_size = self.train_test_ratio, random_state = 1, stratify = df.signed.values)
        df_train, df_valid = train_test_split(df_train_full, train_size = self.train_valid_ratio, random_state = 1, stratify = df_train_full.signed.values)
        print(f"Training size: {df_train.shape}")
        print(f"Validation size: {df_valid.shape}")
        print(f"Test size: {df_test.shape}")


        train_data1 = tf.data.Dataset.from_tensor_slices((df_train['text'].values, df_train['signed'].values, df_train['partisan_lean'].values))
        val_data1   = tf.data.Dataset.from_tensor_slices((df_valid['text'].values, df_valid['signed'].values, df_valid['partisan_lean'].values))
        test_data1  = tf.data.Dataset.from_tensor_slices((df_test['text'].values, df_test['signed'].values, df_test['partisan_lean'].values))

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