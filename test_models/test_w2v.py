# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 07:38:58 2019

@author: maste
"""

# Word2Vec

#import modules
import pickle
from gensim.models import Word2Vec
import pandas as pd
from os import listdir
import random

def train_model_get_cosine_matrix(statements):
    statements = [statement.split() for statement in statements]
    words = sorted(set([word for statement in statements for word in statement]))
    
    w2v = Word2Vec(statements, size=300, window=3, min_count=1, workers=4, iter=100)
    
    #turn dictionary into doc2vec
    sim = [[w2v.wv.n_similarity([worda],[wordb])
        for wordb in words]
        for worda in words]
    
    out = pd.DataFrame(sim)
    out.columns = words
    out.index = words
    return out

if __name__ == "__main__":    
    for path in listdir('../distributions/distance/'):
        print(path)
        path_out = '../distributions/distance/' + path
        with open(path_out,'r') as file:
            statements = file.readlines()
        random.shuffle(statements)
        out = train_model_get_cosine_matrix(statements)
        out.to_csv('../cosines/distance/w2v_'+ path.split('.')[0] + '.csv', index = False)