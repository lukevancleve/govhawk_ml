import unittest
import os
import sys
import pandas as pd

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data.read_parallel import read_parallel_local

raw_folder = "data/vol/raw/"

class TestSum(unittest.TestCase):

    def test_one_known_file(self):
        """
        Test opening a known local file and returning a string
        """

        doc = read_parallel_local([1528347], raw_folder)
        #print(doc)
        self.assertIs(type(doc[0]), str)

    def test_10_000_known_file(self):
        """
        Test opening a known local file and returning a string
        """

        df = pd.read_csv("data/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)
        
        # open many files:
        read_parallel_local(df.sample(1000)['id'], raw_folder)

if __name__ == '__main__':
    unittest.main()