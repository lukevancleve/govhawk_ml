from abc import ABC, abstractmethod
from typing import Dict
import tensorflow as tf
from transformers import LongformerTokenizer
from transformers import TFLongformerModel, TFLongformerForSequenceClassification, TFBertForSequenceClassification
from src.models.data_loader import *
import os

class BaseLegisModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, config: Dict):
        self.config = config
        self.n_dense_layers = 700

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    def train(self):
        """
        Training function for all (nearly?) versions of DeepLegis.
        """

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                           log_dir=self.config['log_dir'], 
                                           histogram_freq=1, 
                                           profile_batch='10, 15'
        )
        
        checkpoint_path = self.config['checkpoint_path']
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        if 'learning_rate' in self.config:
            learning_rate = self.config['learning_rate']
        else:
            learning_rate = 1e-5

        self.deep_legis_model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall(),
                         tf.keras.metrics.AUC()]
        )

        model_history = self.deep_legis_model.fit(self.train_batches, 
                                       epochs=self.config['epochs'],
                                       validation_data=self.val_batches,
                                       callbacks = [cp_callback, tensorboard_callback, es_callback])
        return model_history

    def evaluate(self):

        return  self.deep_legis_model.evaluate(self.test_batches, return_dict=True)
        


class deepLegisAll(BaseLegisModel):
    """
    DeepLegis model with all metadata included. Base model is longformer.
    """
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        self.train_batches, self.val_batches, self.test_batches = \
            legislationDatasetAll(self.config).create_batch_stream(df)

    def build(self):

        # Metadata
        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config['n_sc_id_classes'] ), dtype=tf.float32, name='sc_id')
        meta = tf.concat([pl, vn, cat], axis=-1)

        # Load the initial weights with the ones trained from the DL model without text
        meta = tf.keras.layers.Dense(self.n_dense_layers, activation='relu', name="no_text_dense_layer",
                                     kernel_initializer= noTextKernelInitializer, bias_initializer= noTextBiasInitializer)(meta)

        # transformer
        self.base_transformer_model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
        ids = tf.keras.Input((self.config['max_length']), dtype=tf.int32, name='input_ids')
        x = self.base_transformer_model.longformer(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)

        # Join
        x = tf.concat([x, meta], axis=-1)
        x = tf.keras.layers.Dense(self.n_dense_layers, activation='relu', name="join_dense_layer")(x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids,"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])       




        self.deep_legis_model = dl_model


class noTextKernelInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):

        if 'DATA_VOL' not in os.environ:
            # Manually set:
            DATA_VOL = '/datavol/'
            #raise("Use Docker")
        else:
            DATA_VOL = os.environ['DATA_VOL']
          
        model_location = DATA_VOL + "models/no_text/full_model.h5"
        no_text_model = tf.keras.models.load_model(model_location)
        return tf.convert_to_tensor(no_text_model.get_weights()[0])

    def get_config(self):  # To support serialization
        return {}

class noTextBiasInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):

        if 'DATA_VOL' not in os.environ:
            # Manually set:
            DATA_VOL = '/datavol/'
            #raise("Use Docker")
        else:
            DATA_VOL = os.environ['DATA_VOL']
          
        model_location = DATA_VOL + "models/no_text/full_model.h5"
        no_text_model = tf.keras.models.load_model(model_location)
        return tf.convert_to_tensor(no_text_model.get_weights()[1])
    def get_config(self):  # To support serialization
        return {}
class deepLegisText(BaseLegisModel):
    """
    DeepLegis model with only text included. Base model is longformer.
    """
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        self.train_batches, self.val_batches, self.test_batches = \
            legislationDatasetText(self.config).create_batch_stream(df)

    def build(self):

        self.base_transformer_model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

        ids = tf.keras.Input((self.config['max_length']), dtype=tf.int32, name='input_ids')
        x = model.longformer(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(self.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids}, outputs=[x])

        self.deep_legis_model = dl_model


class deepLegisNoText(BaseLegisModel):
    """
    DeepLegis model with no text included. Reference model for information gain from
    adding the text.
    """
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        self.train_batches, self.val_batches, self.test_batches = \
            legislationDatasetNoText(self.config).create_batch_stream(df)

    def build(self):

        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config['n_sc_id_classes'] ), dtype=tf.float32, name='sc_id')

        x = tf.concat([pl, vn, cat], axis=-1)
        x = tf.keras.layers.Dense(self.n_dense_layers, activation='relu', name="no_text_dense_layer")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])

        self.deep_legis_model = dl_model



class deepLegisPartisanLean(BaseLegisModel):
    """
    DeepLegis model with text and partisan lean included. 
    """
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        self.train_batches, self.val_batches, self.test_batches = \
            legislationDatasetPartisanLean(self.config).create_batch_stream(df)

    def build(self):

        self.base_transformer_model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

        ids = tf.keras.Input((self.config['max_length']), dtype=tf.int32, name='input_ids')
        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        x = model.longformer(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.concat([x, pl], axis=-1)
        x = tf.keras.layers.Dense(self.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids,"partisan_lean":pl}, outputs=[x])
       

        self.deep_legis_model = dl_model


class deepLegisRevCat(BaseLegisModel):
    """
    DeepLegis model with text, version number, partisan lean included. 
    """
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        self.train_batches, self.val_batches, self.test_batches = \
            legislationDatasetRevCat(self.config).create_batch_stream(df)

    def build(self):

        self.base_transformer_model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

        ids = tf.keras.Input((self.config['max_length']), dtype=tf.int32, name='input_ids')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config['n_sc_id_classes'], ), dtype=tf.float32, name='sc_id')

        x = model.longformer(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.concat([x, vn, cat], axis=-1)
        x = tf.keras.layers.Dense(self.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn, "sc_id": cat}, outputs=[x])


        self.deep_legis_model = dl_model


class deepLegisBertAll(BaseLegisModel):
    """
    DeepLegis model with BERT as the transformer, ALL metadata included.
    """
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        self.train_batches, self.val_batches, self.test_batches = \
            legislationDatasetAll(self.config).create_batch_stream(df)

    def build(self):

        self.base_transformer_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        ids = tf.keras.Input((self.config['max_length']), dtype=tf.int32, name='input_ids')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config['n_sc_id_classes'], ), dtype=tf.float32, name='sc_id')

        x = model.longformer(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.concat([x, vn, cat], axis=-1)
        x = tf.keras.layers.Dense(self.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn, "sc_id": cat}, outputs=[x])


        self.deep_legis_model = dl_model