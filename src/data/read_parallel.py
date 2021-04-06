from typing import List
import time

from multiprocessing.pool import ThreadPool


def read_parallel_local(ids: List[int], directory: str, processes=20) -> List[str]:
    """
    Paralleize the reading of documents on the local volume.
    ids: List of ids of the texts to read.
    dir: Directory of the local volume that contains the text.
    processs: number of parallel processes to use.

    """
    def read_legislative_file(rev_id:int):
        try:
            with open(directory + "/" + str(rev_id) + ".txt") as f:
                doc = f.read()
                return doc
        except (OSError, IOError):
            return None

    start_time = time.time()
    tp = ThreadPool(processes=processes)
    docs = tp.map(read_legislative_file, ids)
    tp.terminate()
    tp.close()

    print(f"Took {(time.time()-start_time)/60.0} min ({time.time()-start_time} sec)to open {len(ids)} files with {processes} processes." )

    return docs