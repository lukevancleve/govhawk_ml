# import unittest
# import os
# import sys, getopt
# import pandas as pd
# from transformers import LongformerTokenizer

# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# from src.data.data_downloader import data_downloader

# if 'DATA_VOL' not in os.environ:
#     # Manually set:
#     raise("DATA_VOL should be set in the Docker image.")
#     #DATA_VOL = '/datavol/'
# else:
#     DATA_VOL = os.environ['DATA_VOL']



# raw_folder = DATA_VOL + "raw/"
# externals = 'references/external/'


# class TestSum(unittest.TestCase):

#     def test_clearn_file(self):

#         bill_id = 2123319
#         with open(raw_folder + "/" + str(bill_id) + ".txt") as f:
#             text = f.read()
#         #print(text)
#         bp = data_downloader(DATA_VOL)
#         print("----Original---------------")
#         print(text)
#         print("----Clean------------------")

#         clean = bp.clean_text(text)
#         print(clean)
#         tokenized = tokenizer(clean)

#         print(f"Number of tokens:{len(tokenized['input_ids'])}")
#         #print(f"bill_id: {bill_id}")


# if __name__ == '__main__':

#     unittest.main()