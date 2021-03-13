import sys
import os
import json
import datetime
from src.models.deeplegis import *

class deepLegisConfig():
    """
    Base class. All used classes should implement their
    own __init__, call this init and then set the config_location
    and call self.load_config()
    """

    def __init__(self, json_file, project_root=None):
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
        self.n_dense_layers = 1024
        self.model_location = None
        self.log_dir = None
        self.checkpoint_path = None
        self.model_class = None
        self.load_weights_from_no_text = False

        json_file = 'src/configs/' + json_file

        if project_root is not None:
            self.project_root = project_root

        self.config_location = self.project_root + json_file
        self.load_config()

    def load_config(self):

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
            self.tokenizer = _make_tokenizer(d['tokenizer'])

        # Weights experiment:
        if 'load_weights_from_no_text' in d:
            self.load_weights_from_no_text = d['load_weights_from_no_text']       

        # Model
        self.model_class = _select_model_class(self.model_name)

        self.build_from_scratch = False

        if 'DATA_VOL' not in os.environ:
            # Setting for Docker
            #self.data_vol = '/data/'
            self.data_vol = '/home/luke/tmp_vol/'
        else:
            self.data_vol = os.environ['DATA_VOL']

        self.model_location  = self.data_vol + "models/" + self.model_name + "/full_model.h5"
        self.checkpoint_path = self.data_vol + "models/" + self.model_name +"/" + self.model_name +".ckpt"
        self.log_dir = self.data_vol + "logs/fit/" + self.model_name+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _make_tokenizer(text):

    if text == "distilbert-base-uncased":
        return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if text == "bert-base-uncased":
        return BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise "Invalid tokenizer input:" + text

def _select_model_class(text):

# No transformer baseline
    if text == "no_text":
        return deepLegisNoText  # Class, not and object

# Ablation analysis models
    elif text == "distilbert_128":
        return deepLegisDistillBert  # Class, not and object
    elif text == "distilbert_128_rev_cat":
        return deepLegisRevCat
    elif text == "distilbert_128_text":
        return deepLegisText
    elif text == "distilbert_128_pl":
        return deepLegisPartisanLean

# Different Transformer Architectures:
    elif text == "distilbert_512":
        return deepLegisDistillBert  # Class, not and object
    elif text == "bert_128":
        return deepLegisDistillBert  # Class, not and object
    elif text == "bert_512":
        return deepLegisDistillBert  # Class, not and object

    else:
        raise "Invalid model:" + text
