import requests
from typing import List
import re
from hashlib import md5
from nltk.tokenize import word_tokenize

class bill_version():
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

    def clean_text(self):

        # 0. Remove all the non-ascii characters
        clean = self.plain_text.encode("ascii", "ignore").decode()

        # 1. Remove all the line numbers from bill text
        clean = re.sub('\n\s*\d*\s', '\n', clean)

        # 2. fix between line dashes
        clean = re.sub('[-]\s*\n\s*', '', clean)

        # 3. Remove (\d), (\d\d), or (\a)
        clean = re.sub('\(\d+\)', '', clean)
        clean = re.sub('\(\w+\)', '', clean)

        # 4. Tokenize it
        tokens = word_tokenize(clean)

        # save back to the class
        self.tokens = [word for word in tokens if word.isalpha()]
        self.clean_string = " ".join(self.tokens)



