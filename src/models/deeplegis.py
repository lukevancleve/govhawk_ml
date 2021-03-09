from abc import ABC, abstractmethod
from typing import Dict
import tensorflow as tf
from transformers import LongformerTokenizer
from transformers import TFLongformerModel, TFLongformerForSequenceClassification, TFBertForSequenceClassification
from src.models.data_loader import *

class BaseLegisModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, config: Dict):
        self.config = config

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

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, profile_batch='10, 15')
        checkpoint_path = self.checkpoint_path

        self.deep_legis_model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall(),
                         tf.keras.metrics.AUC()]
        )

        model_history = self.deep_legis_model.fit(self.train, epochs=self.epochs,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_steps=self.validation_steps,
                                       validation_data=self.val,
                                       callbacks = [checkpoint_path, tensorboard_callback])
        return model_history

    def evaluate(self):
        predictions = []
        for x, label in self.test.take(1):
            predictions.append( (self.deep_legis_model.predict(x), label) )
        return predictions

class deepLegisAll(BaseLegisModel):
    def __init__(self, config):
        super().__init__(config)
        

    def load_data(self, df):
        
        train, val, test = legislationDatasetAll(self.config).create_batch_stream(df)
        self.train = train
        self.val = val
        self.test = test

    def build(self):

        self.base_transformer_model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

        ids = tf.keras.Input((self.config['max_length']), dtype=tf.int32, name='input_ids')
        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config['n_sc_id_classes'] ), dtype=tf.float32, name='sc_id')
        
        x = self.base_transformer_model.longformer(ids) # Get the main Layer

        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.concat([x, pl, vn, cat], axis=-1)
        x = tf.keras.layers.Dense(700, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids,"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])       

        self.deep_legis_model = dl_model





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