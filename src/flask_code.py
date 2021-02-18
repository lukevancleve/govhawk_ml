# Serve model as a flask application

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request
import joblib
from src.models.lda_feature_builder import addLDA 
import pandas as pd

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    model = joblib.load('notebooks/prod_model.joblib')
    print(f"Model:{model}")

@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        #print(f"request type is POST:{request.method}")
        data = request.get_json(force=True)  # Get data posted as a json
        #print(data)
        df = pd.json_normalize(data)
        #print(df)
        df['sc_id'] = df['session_id'].astype(str) + "-" + df['chamber_id'].astype(str)
        df = df[['partisan_lean', 'version_number', 'sc_id', 'text']]
        #print(f"XXXXXXXXXXXXXXXXX{model}")
        prediction = model.predict_proba(df)


        print(prediction)
    return str(prediction[:,1])


if __name__ == '__main__':
    import sys
    #print (sys.path)
    load_model()  # load model at the beginning once only
    
    app.run(host='0.0.0.0', port=5000)