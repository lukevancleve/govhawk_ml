# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pandas as pd
import os
import sys
import data_downloader

def main(vol_path):
    """ 
    
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if not os.path.isdir(vol_path):
        raise AssertionError("vol_path not mounted! Create a dir for local work or mount a volume in Docker.")

    # Bill_version.csv is the provided reference for every unique text in the corpus, the actual files
    # are stored in S3 with the URI as a function of original URL to the document.
    bv = pd.read_csv("./references/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)

    D = data_downloader.data_downloader(vol_path)

    D.download_clean_save_from_df(bv)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    try:
        data_vol = os.environ['DATA_VOL']
    except Exception:
        print("No env variable for 'DATA_VOL' specified.")
        sys.exit(1)

    print("data volume set to:" + data_vol)    

    main(data_vol)
