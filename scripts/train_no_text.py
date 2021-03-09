# Training script

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
from sklearn.preprocessing import LabelEncoder
import datetime
from src.data.read_parallel import read_parallel_local
import matplotlib.pyplot as plt
from src.models.deeplegis import *
from src.models.data_loader import *
from transformers import LongformerTokenizer


REDUCE_BY_FACTOR = 1 # Make the dataset smaller for development purposes
train_test_ratio = 0.91
train_valid_ratio = 0.90

if 'DATA_VOL' not in os.environ:
    raise("Use docker. This is set in the docker-compose file")
else:
    DATA_VOL = os.environ['DATA_VOL']
    
# Pre-wrangled metadata
df = pd.read_csv("eferences/derived/ml_data.csv", encoding="latin1", parse_dates=True)
df.id = df.id.astype(int)    
print(f"Original number of examples: {len(df)}")
df = df.sample(n=int(len(df)/REDUCE_BY_FACTOR)) #
print(f"Reduced number of examples:  {len(df)}")

df['text'] = read_parallel_local(df['id'], DATA_VOL + "/clean/"

df = df.reset_index(drop=True)

sc_id_encoder = LabelEncoder()
df['sc_id_cat'] = sc_id_encoder.fit_transform(df['sc_id'])


config = {}
config['max_length'] = 128
config['train_batch_size'] = 4
config['testing'] = False
config['train_test_ratio'] = 0.91
config['train_valid_ratio'] = 0.90 
config['tokenizer'] = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
config['n_sc_id_classes'] = len(sc_id_encoder.classes_)
config['checkpoint_path'] = "/data/models/no_text.ckpt"
config['log_dir'] = "/data/logs/"
config['epochs'] = 2
                                
#a = legislationDatasetPartisanLean(config)


b = deepLegisAll(config)
b.load_data(df)
b.build()
print(b.deep_legis_model.summary()
b.train()