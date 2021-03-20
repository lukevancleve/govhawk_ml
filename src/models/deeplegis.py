from abc import ABC, abstractmethod
from typing import Dict
import tensorflow as tf
from transformers import LongformerTokenizer
from transformers import TFLongformerModel, TFLongformerForSequenceClassification, TFBertForSequenceClassification
from transformers import TFDistilBertForSequenceClassification
from src.models.data_loader import *
import os

class BaseLegisModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, config):
        self.config = config


    def load_data(self, project_root = None, reduce_by_factor=None, only_full=False):

        if only_full:
            self.config.only_full=True
            
        df, sc_id_encoder = createDeepLegisDataFrame(self.config, reduce_by_factor=reduce_by_factor)
        self.df = df
        self.config.n_sc_id_classes = len(sc_id_encoder.classes_)
        self.config.label_endocer = sc_id_encoder

        # Create the specific full,train,val,test streams for this model
        self.process_specific_data()

    @abstractmethod
    def process_specific_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    def load_saved_model(self):
        self.deep_legis_model = tf.keras.models.load_model(self.config.model_location)

    def train(self):
        """
        Training function for all (nearly?) versions of DeepLegis.
        """

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                           log_dir=self.config.log_dir, 
                                           histogram_freq=1, 
                                           profile_batch='10, 15'
        )
        
        checkpoint_path = self.config.checkpoint_path
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        self.deep_legis_model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall(),
                         tf.keras.metrics.AUC()]
        )

        model_history = self.deep_legis_model.fit(self.train_batches, 
                                       epochs=self.config.epochs,
                                       validation_data=self.val_batches,
                                       callbacks = [cp_callback, tensorboard_callback, es_callback])
        return model_history

    def evaluate(self):
        return  self.deep_legis_model.evaluate(self.test_batches, return_dict=True)
    
    def full_dataset_prediction(self):

        preds = self.deep_legis_model.predict(self.full_batches)
        df_preds = self.df[['id', 'passed']]
        df_preds['preds'] = preds

        prediction_file = self.config.data_vol + "models/" + self.config.model_name + "/predicions.csv"
        print("Savinging predictions to: " + prediction_file)

        df_preds = df_preds.merge(self.split_data, on = 'id')

        df_preds.to_csv(prediction_file, index=False)

        return preds

class deepLegisNoText(BaseLegisModel):
    """
    DeepLegis model with no text included. Reference model for information gain from
    adding the text.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self, testing=False):
      
        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetNoText(self.config).create_batch_stream(self.df, testing=testing)

    def build(self):

        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config.n_sc_id_classes,), dtype=tf.float32, name='sc_id')

        x = tf.concat([pl, vn, cat], axis=-1)
        print(x.shape)
        print(self.config.n_sc_id_classes)
        x = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"partisan_lean":pl, "version_number": vn, "sc_id": cat}, outputs=[x])

        self.deep_legis_model = dl_model

class deepLegisText(BaseLegisModel):
    """
    DeepLegis model with only text included. Base model is longformer.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):

        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAll(self.config).create_batch_stream(self.df)

    def build(self):
        self.base_transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')

        x = self.base_transformer_model.distilbert(ids) # Get the main Layer
        x = x[0][:,0,:]
        # x = tf.keras.layers.Dropout(0.2)(x)
        # x = tf.concat([x], axis=-1)
        # x = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu')(x)
        # x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids}, outputs=[x])

        self.deep_legis_model = dl_model

class deepLegisPartisanLean(BaseLegisModel):
    """
    DeepLegis model with text and partisan lean included. 
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):

        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetPartisanLean(self.config).create_batch_stream(self.df)

    def build(self):

        self.base_transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        ids = tf.keras.Input((self.config.max_length), dtype=tf.int32, name='input_ids')
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
        
    def process_specific_data(self):

        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetRevCat(self.config).create_batch_stream(self.df)

    def build(self):
        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        cat = tf.keras.Input((self.config.n_sc_id_classes, ), dtype=tf.float32, name='sc_id')
        meta = tf.concat([vn, cat], axis=-1)
        ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer")
        meta = ntdl(meta)

        # Handle the Transformer
        self.base_transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        x = self.base_transformer_model.distilbert(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)

        # Combine the two and run through another dense layer.
        x = tf.concat([x, meta], axis=-1)
        x = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn,  "sc_id": cat}, outputs=[x])

        self.deep_legis_model = dl_model


        self.deep_legis_model = dl_model

class deepLegisBert(BaseLegisModel):
    """
    DeepLegis model with BERT as the transformer, ALL metadata included.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):

        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAll(self.config).create_batch_stream(self.df)

    def build(self):

        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        cat = tf.keras.Input((self.config.n_sc_id_classes, ), dtype=tf.float32, name='sc_id')
        meta = tf.concat([vn, pl, cat], axis=-1)

        # Load the initial weights with the ones trained from the DL model without text
        if self.config.load_weights_from_no_text:
            #if 'no_text_dense_layer_initialization_path' in self.config:
            print("Usinging pretrained weights from the no_text model! --------------------")
            model_location = self.config.data_vol + "models/no_text/full_model.h5"
            ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer",
                                     kernel_initializer= noTextKernelInitializer(model_location=model_location), bias_initializer= noTextBiasInitializer(model_location=model_location))
        else:
            ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer")
        meta = ntdl(meta)

        # Handle the Transformer
        self.base_transformer_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        x = self.base_transformer_model.bert(ids) # Get the main Layer
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)

        # Combine the two and run through another dense layer.
        x = tf.concat([x, meta], axis=-1)
        x = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn, "partisan_lean": pl, "sc_id": cat}, outputs=[x])

        self.deep_legis_model = dl_model


class deepLegisDistilBert(BaseLegisModel):
    """
    DeepLegis model with DistillBERT as the transformer, ALL metadata included.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):
       
        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAll(self.config).create_batch_stream(self.df)

    def build(self):

        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        cat = tf.keras.Input((self.config.n_sc_id_classes, ), dtype=tf.float32, name='sc_id')
        meta = tf.concat([vn, pl, cat], axis=-1)

        # Load the initial weights with the ones trained from the DL model without text
        if self.config.load_weights_from_no_text:
            #if 'no_text_dense_layer_initialization_path' in self.config:
            print("Usinging pretrained weights from the no_text model! --------------------")
            model_location = self.config.data_vol + "models/no_text/full_model.h5"
            ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer",
                                     kernel_initializer= noTextKernelInitializer(model_location=model_location), bias_initializer= noTextBiasInitializer(model_location=model_location))
        else:
            ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer")
        meta = ntdl(meta)

        # Handle the Transformer
        self.base_transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        x = self.base_transformer_model.distilbert(ids) # Get the main Layer
        hidden_state = x[0]
        
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)

        # Combine the two and run through another dense layer.
        x = tf.concat([x, meta], axis=-1)
        x = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn, "partisan_lean": pl, "sc_id": cat}, outputs=[x])

        self.deep_legis_model = dl_model

class deepLegisDistilBertFeatureExtractor(BaseLegisModel):
    """
    DeepLegis model with DistillBERT as the transformer, output is the hidden state.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):
       
        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAll(self.config).create_batch_stream(self.df)

    def build(self):

        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        # Handle the Transformer
        self.base_transformer_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        x = self.base_transformer_model.bert(ids) # Get the main Layer
        hidden_state = x[0][:,0,:]

        dl_model = tf.keras.Model(inputs={"input_ids":ids,}, outputs=[hidden_state])

        self.deep_legis_model = dl_model


class deepLegisLongformerFeatureExtractor(BaseLegisModel):
    """
    DeepLegis model with DistillBERT as the transformer, output is the hidden state.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):
       
        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAll(self.config).create_batch_stream(self.df)

    def build(self):

        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        # Handle the Transformer
        self.base_transformer_model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
        x = self.base_transformer_model.longformer(ids) # Get the main Layer
        hidden_state = x[0][:,0,:]

        dl_model = tf.keras.Model(inputs={"input_ids":ids}, outputs=[hidden_state])

        self.deep_legis_model = dl_model


class deepLegisDistilBertTextFeatureExtractor(BaseLegisModel):
    """
    DeepLegis model with DistillBERT as the transformer, output is the hidden state.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):
       
        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAll(self.config).create_batch_stream(self.df)

    def build(self):

        text_only_model_location = self.config.output_root + "models/distilbert_128_text/full_model.h5"
        cached_model = tf.keras.models.load_model(text_only_model_location)
           
        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        # Handle the Transformer
        #self.base_transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        x = cached_model.layers[1](ids) # Get the main Layer
        hidden_state = x[0][:,0,:]

        dl_model = tf.keras.Model(inputs={"input_ids":ids,}, outputs=[hidden_state])

        self.deep_legis_model = dl_model


class deepLegisDistilBertSigned(BaseLegisModel):
    """
    DeepLegis model with DistillBERT as the transformer, ALL metadata included.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def process_specific_data(self):
       
        self.train_batches, self.val_batches, self.test_batches, self.full_batches, self.split_data = \
            legislationDatasetAllSigned(self.config).create_batch_stream(self.df)

    def build(self):

        # Handle the Meta Data
        ids = tf.keras.Input((self.config.max_length, ), dtype=tf.int32, name='input_ids')
        vn = tf.keras.Input((1, ), dtype=tf.float32, name='version_number')
        pl = tf.keras.Input((1, ), dtype=tf.float32, name='partisan_lean')
        cat = tf.keras.Input((self.config.n_sc_id_classes, ), dtype=tf.float32, name='sc_id')
        meta = tf.concat([vn, pl, cat], axis=-1)

        # Load the initial weights with the ones trained from the DL model without text
        if self.config.load_weights_from_no_text:
            #if 'no_text_dense_layer_initialization_path' in self.config:
            print("Usinging pretrained weights from the no_text model! --------------------")
            model_location = self.config.data_vol + "models/no_text/full_model.h5"
            ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer",
                                     kernel_initializer= noTextKernelInitializer(model_location=model_location), bias_initializer= noTextBiasInitializer(model_location=model_location))
        else:
            ntdl = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu', name="no_text_dense_layer")
        meta = ntdl(meta)

        # Handle the Transformer
        self.base_transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        x = self.base_transformer_model.distilbert(ids) # Get the main Layer
        hidden_state = x[0]
        
        x = x['last_hidden_state'][:,0,:]
        x = tf.keras.layers.Dropout(0.2)(x)

        # Combine the two and run through another dense layer.
        x = tf.concat([x, meta], axis=-1)
        x = tf.keras.layers.Dense(self.config.n_dense_layers, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        dl_model = tf.keras.Model(inputs={"input_ids":ids, "version_number": vn, "partisan_lean": pl, "sc_id": cat}, outputs=[x])

        self.deep_legis_model = dl_model

    def full_dataset_prediction(self):
        """Overloaded to use a different label"""

        preds = self.deep_legis_model.predict(self.full_batches)
        df_preds = self.df[['id', 'signed']]
        df_preds['preds'] = preds

        prediction_file = self.config.data_vol + "models/" + self.config.model_name + "/predicions.csv"
        print("Savinging predictions to: " + prediction_file)

        df_preds = df_preds.merge(self.split_data, on = 'id')

        df_preds.to_csv(prediction_file, index=False)

        return preds

class noTextKernelInitializer(tf.keras.initializers.Initializer):

    def __init__(self, model_location=None):
        self.model_location = model_location

    def __call__(self, shape, dtype=None):
          
        no_text_model = tf.keras.models.load_model(self.model_location)
        return tf.convert_to_tensor(no_text_model.get_weights()[0])

    def get_config(self):  # To support serialization
        return {}

class noTextBiasInitializer(tf.keras.initializers.Initializer):

    def __init__(self, model_location=None):
        self.model_location = model_location

    def __call__(self, shape, dtype=None):
        no_text_model = tf.keras.models.load_model(self.model_location)
        return tf.convert_to_tensor(no_text_model.get_weights()[1])
    def get_config(self):  # To support serialization
        return {}