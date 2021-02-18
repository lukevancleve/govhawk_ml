import requests
from typing import List
import re
from hashlib import md5
#from nltk.tokenize import word_tokenize

class raw_downloader():
    """
    Govhawk's plain text bills are stored on S3 with an address as a hash of the original documents url. This
    class takes the original version and retrieves the plain text, cleans it, and tokenizes it.
    """

    version_url: str
    plain_text_url: str
    plain_text: str
    tokens: List[str]
    clean_string: str

    def __init__(self, url):
        self.download_plain(url)
        #self.clean_text()

    def download_plain(self, version_url: str):

        self.verison_url = version_url
        self.plain_text_url = "https://s3.amazonaws.com/statesavvy/" + md5(str.encode(version_url)).hexdigest() + ".plain"
        r = requests.get(self.plain_text_url)
        self.plain_text = r.text

class bill_processing():

    def clean_text(self, text: str) -> str:



        # 0. Remove all the non-ascii characters
        clean = text.encode("ascii", "ignore").decode()


        issue = clean.find('Access Denied')
        if issue != -1:
            return(None)

        #
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



