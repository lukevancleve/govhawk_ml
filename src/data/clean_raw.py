import raw_downloader
import pandas as pd
from os.path import exists
import os
from multiprocessing.pool import ThreadPool
import numpy as np

print("here")

bv = pd.read_csv("./data/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)
import time
start_time = time.time()

os.mkdir('data/raw')


def copy_local(i) -> int:
    """
    Go to the (i)th index of the bill version csv file and download the file for local work.
    """

    try:
       file_name =  "./data/raw/"+str(bv['id'][i])+".txt" 
    except Exception:
        print("Index {i} is longer than the data (len { len(bv) }). ")


    if exists(file_name):
        return 0 # Correctly skipped

    if i % 10_000 == 0:
        print(" Iteration: {}".format(i))
        print(" --- %s seconds ---" % (time.time() - start_time))

    url = bv['url'][i]

    try:
        br = raw_downloader.raw_downloader(url)

        text_file = open(file_name, "w")
        rv = text_file.write(br.plain_text)
        text_file.close()
    except Exception:
        print(f"Issue open/closing file from {url}.")

    return 1  # Correctly downloaded


tp = ThreadPool(processes=30)

n_dl = tp.map(copy_local, range(len(bv['url'])))

print(f"Number of files downloaded: {np.sum(n_dl)}")
print(f"Number of files already downloaded: {len(n_dl) - np.sum(n_dl)}")

tp.terminate()
tp.close()