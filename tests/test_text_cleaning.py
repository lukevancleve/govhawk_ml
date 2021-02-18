import unittest
import os
import sys, getopt
import pandas as pd

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


from src.data.raw_downloader import bill_processing
raw_folder = "data/raw"
externals = 'data/external/'


class TestSum(unittest.TestCase):

    def test_clearn_file(self):

        bill_id = 2312354
        with open(raw_folder + "/" + str(bill_id) + ".txt") as f:
            text = f.read()
        #print(text)
        bp = bill_processing()
        print(text)
        print("---------------------------")
        print(bp.clean_text(text))
        #print(f"bill_id: {bill_id}")


if __name__ == '__main__':

    unittest.main()