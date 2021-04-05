import logging
from pathlib import Path
import pandas as pd
import os
import sys
from src.data.data_downloader import data_downloader

def main(vol_path):
    """ 
    Download and clean all of the associated text files from S3 used in the training of this model. The files
    need to be stored locally as downloading them as a stream when training would be a prohibitive bottleneck.
    The files will later be tokenized and stored in a pickled dataset for easy reuse. These files are 
    around 10 GB and should not be in a docker image (as specified by the .dockerignore file).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if not os.path.isdir(vol_path):
        raise AssertionError("vol_path not mounted! Create a dir for local work or mount a volume in Docker.")

    # Bill_version.csv is the provided reference for every unique text in the corpus, the actual files
    # are stored in S3 with the URI as a function of original URL to the document.
    bv = pd.read_csv("./references/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)

    D = data_downloader(vol_path)

    D.download_clean_save_from_df(bv)

    logger.info('Finished final data set from raw data')

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    try:
        data_vol = os.environ['DATA_VOL']
    except Exception:
        print("No env variable for 'DATA_VOL' specified.")
        sys.exit(1)

    print("data volume set to:" + data_vol)    

    main(data_vol)
