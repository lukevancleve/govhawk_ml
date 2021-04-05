import requests
from typing import List, Tuple
import re
from hashlib import md5
import pandas as pd
import os
from multiprocessing.pool import ThreadPool
import numpy as np
#from nltk.tokenize import word_tokenize

class data_downloader():
    """
    Govhawk's plain text bills are stored on S3 with an address as a hash of the original documents url. This
    class takes the original version and retrieves the plain text, cleans it, and tokenizes it.
    """

    volume: str
    raw_dir: str
    clean_dir: str

    def __init__(self, volume):

        """
        Make new directories to store text files.
        """

        self.volume = volume
        self.raw_dir = self.volume + "/raw/"
        self.clean_dir = self.volume + '/clean/'
    
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)
        if not os.path.exists(self.clean_dir):
            os.mkdir(self.clean_dir)


    def download_clean_save_from_df(self, df):
        """
        Original reference data came from a csv file.
        """

        bills = zip(df['id'], df['url'])
        ids = list(range(len(df)))

        tp = ThreadPool(processes=30)
        n_dl = tp.map(self.download_clean_save_individual, bills)
        tp.terminate()
        tp.close()

        print(f"Number of files downloaded:  {np.sum([x[0] for x in n_dl])}")
        print(f"Number of files cleaned: {np.sum([x[1] for x in n_dl])}")


    def download_clean_save_individual(self, id_url: Tuple[int, str]):
        """
        Given an (id, url) list, save the raw and clean copy of a bill.
        """
        id, url = id_url
        try:
           raw_text = self.download_plain(url)
        except Exception:
            print(f"Could not open: {url}")
            return (0,0)

        try:
            raw_file =  self.raw_dir + str(id) + ".txt"
            with open(raw_file, "w") as f:
                f.write(raw_text)
        except Exception:
            return (0, 0)

        try:
            clean_text = self.clean_text(raw_text)
            clean_file = self.clean_dir + str(id) + ".txt"
            with open(clean_file, "w") as f:
                f.write(clean_text)
        except:
            return (1,0)

        return (1,1)


    def download_plain(self, version_url: str) -> str:

        plain_text_url = "https://s3.amazonaws.com/statesavvy/" + md5(str.encode(version_url)).hexdigest() + ".plain"
        r =  requests.get(plain_text_url, timeout=20)

        return r.text

    def clean_text(self, text: str) -> str:

        # 0. Remove all the non-ascii characters
        clean = text.encode("ascii", "ignore").decode()


        issue = clean.find('Access Denied')
        if issue != -1:
            return(None)

        start_anchors = ["- FullText", "DELAWARE STATE SENATE", "STATE OF NEW HAMPSHIRE",
        "Be it enacted by the General Assembly of the state of Missouri, as follows:",
        "(Text matches printed bills. Document has been reformatted to meet World Wide Web specifications."]
        for start_anchor in start_anchors:
            ind = clean.find(start_anchor)
            if ind != -1:
                clean = clean[(ind+len(start_anchor)):]


        end_anchors = ["----XX----"]
        for end_anchor in end_anchors:
            ind = clean.find(end_anchor)
            if ind != -1:
                clean = clean[:ind]

        #return(clean)
        # 1. Remove all the line numbers from bill text
        clean = re.sub('\n\s*\d*\s', '\n', clean)

        # Remove ic 3-2-3-233
        clean = re.sub('ic\s+[\d-]+', '', clean, flags=re.IGNORECASE )

        # Remove p.l.333-3333
        clean = re.sub('p\.l\.\s+[\d-]+', '', clean, flags=re.IGNORECASE )

        clean = re.sub('SB\d+ Enrolled.*?\\n', '', clean)

        # remove line feed looking thing:
        clean = re.sub('\s*=+\\n\s*LC\d*\s*\\n\s*=+\\n', '', clean)

        # 2. fix between line dashes
        clean = re.sub('[-]\s*\n\s*', '', clean)

        # 3. Remove (\d), (\d\d), or (\a)
        clean = re.sub('\(\d+\)', '', clean)
        clean = re.sub('\(\w+\)', '', clean)

        # 4. Remove markdownlinks:
        clean = re.sub('\[.*?\]\(.*?\)', '', clean)

        # 5. remove newlines
        clean = re.sub('\\n', ' ', clean)

        # 5. remove dashes following a space or preceding a space
        clean = re.sub('[-_]+', '-', clean)
        clean = re.sub(' [-_]', ' ', clean)
        clean = re.sub('[-_] ', ' ', clean)

        # collpse whitespce into one space.
        clean = re.sub('\s+', ' ', clean)
  
        clean = re.sub('\\nSec\. \d.*?.\\n', "\n", clean)

        clean = re.sub('\(.*?\)', '', clean)
        clean = re.sub('\[.*?\]', '', clean)

        clean = re.sub('\(Source.*?\)', '', clean)

        clean = re.sub('[Ss]ection \d+', '', clean)
        clean = re.sub('[hH]\.? [rR]\.? \d+', '', clean)

        clean = re.sub(' [-.,\d]+ ', ' ', clean)

        # lower case all
        clean = clean.lower()

        return(clean)
        # 4. Tokenize it
        #tokens = word_tokenize(clean)

        # save back to the class
        #self.tokens = [word for word in tokens if word.isalpha()]
        #self.clean_string = " ".join(self.tokens)



