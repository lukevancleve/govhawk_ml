DeepLegis
==============================

Machine learning system using NLP on state-level legislative text. Work was done jointly for the startup [Govhawk](https://govhawk.com/) and as a Capstone 
product for the Machine Learning Engineer trach with [Springboard](https://www.springboard.com/)

![Architecture](https://github.com/lukevancleve/govhawk_ml/blob/master/reports/figures/network_architecture.png)

Project Organization
------------


    ├── README.md          <- The top-level README for developers using this project.
    │    │
    ├── models             <- Trained and serialized models, other artifacts
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- External data, data derived from external data.
    |   |--- external      <- Reference csv files
    |   |--- derived       <- Tables derived from the external csv files.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    |
    |-- sripts             <- Standalone python scripts that use modules in src


Instructions for use:
---------

## Production

The production version of this model is a docker image that listens to an AWS SQS feed. Two artifacts are cached from the training process, `models/encoder_production.pkl` and `models/catboost.production`, these respectively encode which legislative session the bill belongs to and the actual GBDT Catboost
model. For a real production process one would edit the files in `deployed\` to listen to a production SQS queue instead of the test one and write the output to
relational tables.

```
python deployed/test_recieve
```


## Training

#### Step 1 - Take training text files off of S3 and store locally

From within the docker image, run the below command. This will open up the csv file in `references/external/` and download all of the text file into
the directory specified by `$DATA_VOL`. This will create two dirctories in `$DATA_VOL`: `raw` and `clean`. The `raw` directory has the original text
files from each state, the `clean` directory has these same files but stripped of a large amount of superfluous text.

```
python scripts/make_local_clean_text.py
```

#### Step 2

Create cleaned ML datasets from the provided metadata.

```
python scripts/run_partisan_lean.py
python scripts/make_ml_data.py
```

#### Step 3

Tokenize the clean text data and combine it with the pre-prepared ML dataset. The tokenization of a batch takes a substantive amount of time relative to
the amount of time a GPU takes to process the batch. Thus, the strategy is to pretokenized and cache the inputs. 

```
python scripts/pretokenize.py
```

#### Step 4

Train the model. Run all the cached observations through BERT and then use the outputs as features for Catboost. Save the model.

```
python scripts/train_prod_model.py
```