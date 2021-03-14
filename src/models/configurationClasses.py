import sys
import os
import json
import datetime
from transformers import BertTokenizer, DistilBertTokenizer
from src.models.deeplegis import *

class deepLegisConfig():
    """
    Load and process a json configuration file for a model.
    """

    def __init__(self, json_file, project_root=None, output_root=None):
        self.model_name = None
        self.build_from_scratch = False
        self.tokenizer = None
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None
        self.n_sc_id_classes = None
        self.project_root = "./" # Notebooks will pass "../"
        self.data_vol = None
        self.max_length = None
        self.n_dense_layers = 1024 # metadata dense layer.
        self.model_location = None
        self.log_dir = None
        self.checkpoint_path = None
        self.model_class = None
        self.load_weights_from_no_text = False
        self.output_root = None

        json_file = 'src/configs/' + json_file

        if project_root is not None:
            self.project_root = project_root

        self.config_location = self.project_root + json_file

        with open(self.config_location) as f:
            d = json.load(f)

        def _assign_required_element(ele_name):
            if ele_name not in d:
                raise "'"+ele_name+"' is required in the config"
            setattr(self, ele_name, d[ele_name])

        _assign_required_element('model_name')
        _assign_required_element('batch_size')
        _assign_required_element('epochs')
        _assign_required_element('learning_rate')

        # Tokenizer
        if 'tokenizer' in d:
            self.tokenizer = self._make_tokenizer(d['tokenizer'])
            self.max_length = d['max_length']

        # Weights experiment:
        if 'load_weights_from_no_text' in d:
            self.load_weights_from_no_text = d['load_weights_from_no_text']       

        # Model
        self.model_class = self._select_model_class(self.model_name)

        self.build_from_scratch = False

        
        if 'DATA_VOL' not in os.environ:
            raise "Please set the DATA_VOL as an environmental variable."
        else:
            self.data_vol = os.environ['DATA_VOL']

        # Write the output to the same directory the clean data is in
        # optionally do it somewhere else. E.g. for tests.
        if output_root is not None:
            self.output_root = output_root
        else:
            self.output_root = self.data_vol

        self.model_location  = self.output_root + "models/" + self.model_name + "/full_model.h5"
        self.checkpoint_path = self.output_root + "models/" + self.model_name +"/" + self.model_name +".ckpt"
        self.log_dir = self.output_root + "logs/fit/" + self.model_name+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    _valid_tokenizers_strings = [ "distilbert-base-uncased", "bert-base-uncased"] 

    def _make_tokenizer(self, text):
        """
        Instantiate a tokenizer by name and return to the config
        """

        if text not in self._valid_tokenizers_strings:
            raise "Invalid tokenizer input:" + text

        if text == "distilbert-base-uncased":
            return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if text == "bert-base-uncased":
            return BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise "Previous test should never let you get here."

    def _select_model_class(self, text):
        """
        Match the model_name to that model's class code.
        """

    # No transformer baseline
        if text == "no_text":
            return deepLegisNoText 

    # Ablation analysis models
        elif text == "distilbert_128":
            return deepLegisDistillBert  
        elif text == "distilbert_128_rev_cat":
            return deepLegisRevCat
        elif text == "distilbert_128_text":
            return deepLegisText
        elif text == "distilbert_128_pl":
            return deepLegisPartisanLean

    # Different Transformer Architectures:
        elif text == "distilbert_512":
            return deepLegisDistillBert 
        elif text == "bert_128":
            return deepLegisBert  
        elif text == "bert_512":
            return deepLegisBert 

        else:
            raise "Invalid model:" + text
