import raw_downloader
import pandas as pd
from os.path import exists
import os
from multiprocessing.pool import ThreadPool
import numpy as np
from hashlib import md5

bv = pd.read_csv("./data/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)
import time
start_time = time.time()

if not os.path.exists('data/raw'):
    os.mkdir('data/raw')
if not os.path.exists('data/clean'):
    os.mkdir('data/clean')


def copy_local_and_clean(i):
    """
    Go to the (i)th index of the bill version csv file and download the file for local work.
    """


    try:
       file_name =  "./data/raw/"+str(bv['id'][i])+".txt" 
    except Exception:
        print("Index {i} is longer than the data (len { len(bv) }). ")

    if i % 10_000 == 0:
        print(" Iteration: {}".format(i))
        print(" --- %s seconds ---" % (time.time() - start_time))

    if not exists(file_name):

        url = bv['url'][i]
        br = raw_downloader.raw_downloader(url)
        plain_text = br.plain_text
        text_file = open(file_name, "w")
        rv = text_file.write(plain_text)
        text_file.close()
    else:
        with open(file_name) as f:
            plain_text = f.read()



    bp = raw_downloader.bill_processing()

    ct = bp.clean_text(plain_text)
    
    ac = 0
    if ct is None:
        #print(str.encode(bv['url'][i]))

        hash = md5()
        hash.update(str.encode(bv['url'][i]))

        url = "https://s3.amazonaws.com/statesavvy/" + hash.hexdigest()
        #print( f"Access Denied for bill version {bv['id'][i]} at {url}" )
        ac = 1
    else:

        file_name =  "./data/clean/"+str(bv['id'][i])+".txt"
        with open(file_name, "w") as f:
            rv = f.write(ct)
        
       
    return tuple([1, ac])  


tp = ThreadPool(processes=30)

n_dl = tp.map(copy_local_and_clean, range(len(bv['url'])))

print(f"Number of files downloaded:  {np.sum([x[0] for x in n_dl])}")
print(f"Number of files inaccesible: {np.sum([x[1] for x in n_dl])}")

tp.terminate()
tp.close()