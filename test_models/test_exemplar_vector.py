# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:48:53 2020

@author: maste
"""

from collections import Counter
from os import listdir

import numpy as np
from scipy import sparse
import pandas as pd


#Generate Memory, Dictionary
def get_sparse_wd(corpus):
    '''
    Input list of strings, convert to sparse word by document matrix
    '''
#    path = '5_points.pk'
#    corpus = pickle.load(open(path,'rb'))
#    corpus = [line.rstrip('\n') for line in corpus]
#
    corpus = Counter(corpus)
    corpus, counts = list(zip(*corpus.items()))
    corpus = [line.rstrip('\n').split(' ') for line in corpus]

    #parse to get dictionary
    corpus_ = []
    for corp in corpus:
        corpus_.extend(corp)

    mydict = sorted(set(corpus_))
    mydict = dict(zip(mydict, range(0, len(mydict))))

    #map to get key/values from dictionary -col
    col = list(map(mydict.get, corpus_))

    #map to get row values
    row = []
    for index, corp in enumerate(corpus):
        row.extend([index] * len(corp))
    data = []
    for val in counts:
        data.extend([val]*3)
#    data = [1] * len(row)
    mem = sparse.csr_matrix((data, (row, col)),
                            shape=(len(corpus),
                                   len(mydict)),
                            dtype='float32')
    return mem, mydict

def get_set_of_contexts(wd, mydict, word, drop):
    '''
    Retrieve echoes for all contexts in which word occurs
    '''
    vects = np.diag(np.ones(len(mydict)))
    vects[:, word] = 1
    vects = np.delete(vects, [word, drop], 0)

    echoes = np.zeros((len(mydict)-1, len(mydict)))

    for ind, vect in enumerate(vects):
        activation = np.ones(wd.shape[0])
        for i in np.where(vect != 0)[0]:
            probe = np.zeros(len(vect))
            probe[i] = 1
            activation = activation * \
                np.power(wd.dot(probe) /
                         (np.array(np.sqrt(wd.multiply(wd).sum(1)).transpose())[0] \
                          * np.sqrt(1)), 3)
        echo = wd.T.multiply(activation).sum(1).T
        echoes[ind] = np.multiply(echo, (1 - vect))

    return echoes

def cosineTable(vects):
    '''
    cosine similarity matrix
    '''
    return vects.dot(vects.transpose()) / \
            np.outer(np.sqrt(vects.power(2).sum(1)),
                     np.sqrt(vects.power(2).sum(1)))

def get_sim(AC, BC):
    '''
    cosine similarity matrix between different vectors
    '''
    return np.divide(np.multiply(AC, BC).sum(1),
                     np.multiply(np.sqrt(np.power(AC, 2).sum(1)), \
                                 np.sqrt(np.power(BC, 2).sum(1))) + .0001).sum()

def train_model_get_cosine_matrix(statements):
    '''
    train model
    '''
    wd, mydict = get_sparse_wd(statements)

    sims = np.zeros((len(mydict), len(mydict)))
    for i in np.arange(len(mydict)):
        for j in np.arange(i, len(mydict)):
            AC = get_set_of_contexts(wd, mydict, i, j)
            BC = get_set_of_contexts(wd, mydict, j, i)
            sim = get_sim(AC, BC)
            sims[i, j] = sim
            sims[j, i] = sim
    sims = sims / sims.max(0)
    sims = pd.DataFrame(sims)
    sims.columns = list(mydict.keys())
    sims.index = list(mydict.keys())
    return sims

if __name__ == "__main__":
    for path in listdir('../distributions/distance/'):
#        path = listdir('../distributions/distance/')[10]
        print(path)
        path_out = '../distributions/distance/' + path
        with open(path_out, 'r') as file:
            in_statements = file.readlines()
        out = train_model_get_cosine_matrix(in_statements)
        out.to_csv('../cosines/distance/exemplarVector2_'+ path.split('.')[0] + \
                   '.csv', index=False)

    for path in listdir('../distributions/uniform/'):
#        path = listdir('../distributions/distance/')[10]
        print(path)
        path_out = '../distributions/uniform/' + path
        with open(path_out, 'r') as file:
            in_statements = file.readlines()
        out = train_model_get_cosine_matrix(in_statements)
        out.to_csv('../cosines/uniform/exemplarVector2_'+ path.split('.')[0] + \
                   '.csv', index=False)
