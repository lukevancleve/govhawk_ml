import version_reader
import pandas as pd
from os.path import exists
from multiprocessing.pool import ThreadPool

bv = pd.read_csv("./data/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)
import time
start_time = time.time()


def copy_local(i):

    print(i)
    file_name =  "./data/raw/"+str(bv['id'][i])+".txt" 
    if exists(file_name):
        print(f"skip {i}")
        return None

    if i % 10_000 == 0:
        print(" Iteration: {}".format(i))
        print(" --- %s seconds ---" % (time.time() - start_time))

    url = bv['url'][i]

    br = version_reader.bill_version(url)
    #print(br.plain_text)

    #print(i)
    text_file = open(file_name, "w")
    rv = text_file.write(br.plain_text)
    text_file.close()


tp = ThreadPool(processes=30)
#versions = ThreadPool(10).imap_unordered(copy_local, range(len(bv['url'])))
#or i in versions:
#    copy_local(i)

tp.map(copy_local, range(len(bv['url'])))

#for i in range(10):
#for i in range(len(bv['url'])):
#    copy_local(i)

import time
time.sleep(10)

tp.terminate()
tp.close()