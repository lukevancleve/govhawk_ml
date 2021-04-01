govhawk_ml
==============================

ML NLP on state level legislative text. Currently a __work in progress__. Plan is to be deliverable end of March 2021.

![Architecture](https://github.com/lukevancleve/govhawk_ml/blob/master/reports/figures/network_architecture.jpg)

Project Organization
------------


    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── clean          <- Cleaned text file
    │   └── raw            <- The original, raw text files.
    │    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
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


Instructions for use:
---------

## Production


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
