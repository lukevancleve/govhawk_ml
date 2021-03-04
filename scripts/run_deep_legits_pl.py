import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
from sklearn.preprocessing import LabelEncoder
import transformers
print(sys.version)
print(transformers.__version__)
print(tf.__version__)

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data.read_parallel import read_parallel_local
from src.models.deeplegis import legislationDataset, legislationDatasetAll, legislationDatasetText, legislationDatasetRevCat
from src.models.deeplegis import deep_legis_pl, deep_legis_text, deep_legis_all, deep_legis_vn_cat



####################
REDUCE_BY_FACTOR = 100 # Make the dataset smaller for development purposes
train_test_ratio = 0.91
train_valid_ratio = 0.90

if 'DATA_VOL' not in os.environ:
    # Manually set:
    DATA_VOL = '/datavol'
else:
    DATA_VOL = os.environ['DATA_VOL']
    
# Pre-wrangled metadata
df = pd.read_csv("references/derived/ml_data.csv", encoding="latin1", parse_dates=True)
df.id = df.id.astype(int)    
print(f"Original number of examples: {len(df)}")
df = df.sample(n=int(len(df)/REDUCE_BY_FACTOR)) #
print(f"Reduced number of examples:  {len(df)}")

tmp = read_parallel_local(df['id'], DATA_VOL + "/clean/")
df['text'] = tmp

df = df.reset_index(drop=True)
sc_id_encoder = LabelEncoder()
df['sc_id_cat'] = sc_id_encoder.fit_transform(df['sc_id'])
print(df.head())
assert df['text'][0] is not None
###########################3


config = {}
config['max_length'] = 128
config['train_batch_size'] = 4
config['testing'] = False
config['train_test_ratio'] = 0.91
config['train_valid_ratio'] = 0.90
config['n_sc_id_classes'] = len(sc_id_encoder.classes_)
config['epochs'] = 5

print(config)
legis_builder = legislationDataset(config)
train_data, val_data, test_data = legis_builder.create_batch_stream(df)
from src.models.deeplegis import deep_legis_pl 
dl_pl = deep_legis_pl(config)
dl_pl.summary()

dl_pl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy()])

history = dl_pl.fit(train_data,
                   validation_data=val_data,
                   epochs=config['epochs'],
                   verbose=2, steps_per_epoch=10)

dl_pl.save('models/dl_pl')
print(history)