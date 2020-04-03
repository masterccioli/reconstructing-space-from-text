# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:48:53 2020

@author: maste
"""

import numpy as np
from scipy import sparse
from os import listdir
import pandas as pd

#Generate Memory, Dictionary
def get_sparse_wd(corpus):
#    path = '5_points.pk'
#    corpus = pickle.load(open(path,'rb'))
    corpus = [line.rstrip('\n').split(' ') for line in corpus]

    #parse to get dictionary
    c = []
    for corp in corpus:
        c.extend(corp)

    mydict = sorted(set(c))
    mydict = dict(zip(mydict, range(0, len(mydict))))

    #map to get key/values from dictionary -col
    col = list(map(mydict.get, c))

    #map to get row values
    row = []
    for index, corp in enumerate(corpus):
        row.extend([index] * len(corp))

    data = [1] * len(row)

    Mem = sparse.csr_matrix((data, (row, col)),
                            shape=(len(corpus),
                                   len(mydict)),
                                   dtype='float32')
    return Mem, mydict

def get_set_of_contexts(wd, mydict, word):
#    C = list(mydict.values())
#    C.remove(word_a)
#    C.remove(word_b)
    # insert ones in word col,row
    cols = list(np.repeat(word, len(mydict) - 1))
    rows = list(np.delete(np.arange(len(mydict)),word))
    #insert diagonal
    cols.extend(list(np.arange(len(mydict))))
    rows.extend(np.arange(len(mydict)))
#    cols = list(zip(np.repeat(word,len(mydict)-1), C)) #
#    cols = [item for sublist in cols for item in sublist]
#    rows = np.repeat(np.arange(len(C)),2)
    data = [1] * len(rows)
    AC = sparse.csr_matrix((data, (rows,cols)), shape = (len(mydict), len(mydict)))
    
    P = AC.dot(wd.transpose())
#    P = P.tolil()
#    P[np.where(P.A < 2)] = 0
#    P[np.where(P.A > 0)] = 1
#    P = P.tocsr()
    P = sparse.csr_matrix(np.round(P.multiply(1/wd.sum(1).transpose()).tocsr().A))
    
    echos = P.dot(wd)
    context = echos - AC*echos.max()
    context[np.where(context.A < 0)] = 0
    return context

def cosineTable(vects):
    return vects.dot(vects.transpose()) / \
            np.outer(np.sqrt(vects.power(2).sum(1)),
                     np.sqrt(vects.power(2).sum(1)))



def get_sim(AC,BC):
    return np.divide(AC.multiply(BC).sum(1), np.multiply(np.sqrt(AC.power(2).sum(1)), np.sqrt(BC.power(2).sum(1))) + .0001).sum() / BC.shape[0]

def train_model_get_cosine_matrix(statements):
    wd, mydict = get_sparse_wd(statements)
    
    # reduce wd so traces are unique
    
    # get cossim for all rows in matrix
    trace_sims = wd.dot(wd.transpose()).multiply(1/3) # denom is 3 since all statements len == 3
    # for each row in trace_sims
    
    new_wd = []
    skip = []
    # while wd still has traces
    while (len(skip) < 10000):
        test_rows = np.arange(10000)
        test_rows = np.delete(test_rows, skip)
        row = trace_sims[test_rows[0]]
        
        # find which rows are identical
        rows = np.where(np.round(row.A,1) == np.ones(10000))[1]
        # put sum in new matrix
        new_wd.append(sparse.csr_matrix(wd[rows].sum(0)))
        # remove row and all identical rows from trace_sims
        skip.extend(rows)
    wd = sparse.vstack(new_wd)
    
    sims = np.zeros((len(mydict), len(mydict)))
    for i in np.arange(len(mydict)):
        for j in np.arange(i,len(mydict)):
            # skip in i == j
            if i == j:
                sims[i, j] = 1
                sims[j, i] = 1
                continue
            AC = get_set_of_contexts(wd, mydict, i)
            BC = get_set_of_contexts(wd, mydict, j)
            sim = get_sim(AC,BC)
            sims[i, j] = sim
            sims[j, i] = sim
    
    out = pd.DataFrame(sims)
    out.columns = list(mydict.keys())
    out.index = list(mydict.keys())
    return out

if __name__ == "__main__":
    for path in listdir('../distributions/distance/'):
#        path = listdir('../distributions/distance/')[10]
        print(path)
        path_out = '../distributions/distance/' + path
        with open(path_out,'r') as file:
            statements = file.readlines()
        out = train_model_get_cosine_matrix(statements)
        out.to_csv('../cosines/distance/exemplar_'+ path.split('.')[0] + '.csv', index = False)