# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:39:43 2019

@author: maste
"""
# LSA

#import modules
from gensim import models,corpora
from gensim.similarities import MatrixSimilarity
import pandas as pd
from os import listdir
import numpy as np

def train_model_get_cosine_matrix(statements, num):
    
    statements = [statement.split() for statement in statements]
    dictionary = corpora.Dictionary(statements)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in statements]
    
    ###tfidf model
    # https://stackoverflow.com/questions/50521304/why-i-get-different-length-of-vectors-using-gensim-lsi-model
    tfidf = models.TfidfModel(doc_term_matrix, normalize = True)
    corpus_tfidf = tfidf[doc_term_matrix]
    
    lsi = models.LsiModel(corpus_tfidf, num_topics=num, id2word=dictionary)
    
    #turn dictionary into doc2vec
    words = [dictionary.doc2bow([word])for word in sorted(list(dictionary.token2id.keys()))]
    
    vectorized_corpus = lsi[words]
    
    index = MatrixSimilarity(vectorized_corpus)
    index[vectorized_corpus]
    
    out = pd.DataFrame(index[vectorized_corpus])
    out.columns = sorted(list(dictionary.token2id.keys()))
    out.index = sorted(list(dictionary.token2id.keys()))
    return out

if __name__ == "__main__":
    dims = np.arange(5,16)    
    for path in listdir('../distributions/distance/'):
        print(path)
        path_out = '../distributions/distance/' + path
        with open(path_out,'r') as file:
            statements = file.readlines()
            
        for i in dims:
            out = train_model_get_cosine_matrix(statements,i)
            out.to_csv('../cosines/distance/lsa'+str(i)+'_'+ path.split('.')[0] + '.csv', index = False)