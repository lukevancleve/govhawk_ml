from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess

class addLDA(BaseEstimator, TransformerMixin):

    
    def __init__(self, lda_model, id2word, n_topics):
        
        self.lda_model = lda_model
        self.id2word = id2word
        self.n_topics = n_topics
    
    def dense_vector(self, ldav):
        v = [0]*self.n_topics
        for (i, p) in ldav:
            v[i] = p
        return tuple(v)

    def sent_to_words(self, sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(simple_preprocess(str(sentence), deacc=True))


    def fit(self, target, Y=None):
        #print("Fitting")
        return self
    
    def transform(self, target, Y=None):
        #print("Transforming")
        #print(f" Passed in data shape{target.shape}")
        
        data_words = list(self.sent_to_words(target.text.values.tolist()))
        corpus = [self.id2word.doc2bow(text) for text in data_words]
        doc_topics = [self.dense_vector(self.lda_model[x]) for x in corpus]
        features = np.asarray(doc_topics)
        #print(f"Features shape:{features.shape}")
     
        return pd.DataFrame(features)