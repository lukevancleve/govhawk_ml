from typing import List
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import time


def read_parallel_local(ids: List[int], dir: str, processes=20) -> List[str]:
    """
    Paralleize the reading of documents on the local volume.
    ids: List of ids of the texts to read.
    dir: Directory of the local volume that contains the text.
    processs: number of parallel processes to use.

    """
    def read_legislative_file(id):
        try:
           with open(dir + "/" + str(id) + ".txt") as f:
                doc = f.read()
                return doc
        except:
            return None

    start_time = time.time()
    tp = ThreadPool(processes=processes)
    docs = tp.map(read_legislative_file, ids)
    tp.terminate()
    tp.close()

    print(f"Took {(time.time()-start_time)/60.0} min to open {len(ids)} files with {processes} processes." )

    return docs