import pandas as pd
#import spacy
#from spacy.tokens import Doc
#from spacy.vocab import Vocab
import time
from multiprocessing.pool import ThreadPool


#nlp = spacy.load("en_core_web_sm")

bv = pd.read_csv("./data/external/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)

#i = 1
file_names =  ["./data/raw/"+str(id)+".txt" for id in bv['id']]

def make_features(file_name):
    with open(file_name, "r") as myfile:
        text = myfile.read()

    n_char = len(text)
    
    return n_char




tp = ThreadPool(processes=30)
start_time = time.time()
text_lengths = tp.map(make_features, file_names)
print(" --- Processing files took: %s seconds ---" % (time.time() - start_time))

features = pd.DataFrame(text_lengths, columns = ['text_length'], index = bv.id)
features.to_csv('./data/derived/basic_version_features.csv')
#start_time = time.time()
#docs = [make_doc(file) for file in file_names]
#texts = [make_texts(fn) for fn in file_names[10000:20000]]
#print(" --- Reading files took: %s seconds ---" % (time.time() - start_time))


#start_time = time.time()
#docs = []
#for doc in nlp.pipe(texts, disable = ['tagger', "parser", "ner"]):
#    docs.append(doc)
#docs = list(nlp.pipe(texts, disable = ['tagger', "parser", "ner"]))
#print(" --- Tokenizing took: %s seconds ---" % (time.time() - start_time))



#start_time = time.time()
#for i in range(len(docs)):
#    fn = "data/tokenized/" + str(bv["id"][i])
#    docs[i].to_disk(fn)
#print(" --- Saving tokens took: %s seconds ---" % (time.time() - start_time))



#start_time = time.time()
#docs2 = []
#for i in range(len(docs)):
#    fn = "data/tokenized/" + str(bv["id"][i])
#    docs2.append(Doc(Vocab()).from_disk(fn))
#print(" --- Loading tokens took: %s seconds ---" % (time.time() - start_time))
