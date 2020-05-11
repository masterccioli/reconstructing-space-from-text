# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:12:45 2019

@author: maste
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:39:43 2019

@author: maste
"""
# LSA

#import modules
import pickle
from gensim import models,corpora
from gensim.similarities import MatrixSimilarity
import pandas as pd
from os import listdir
import numpy as np

def train_model_get_cosine_matrix(statements):
    # make exemplar memory
    mem = [set(statement.split()) for statement in statements]    
    # make dictionary
    dictionary = sorted(set([word for statement in mem for word in statement]))
    
#    a = set([dictionary[0], dictionary[1]])
#    b = set([dictionary[2], dictionary[1]])
#    
#    a_ = retrieve(a,mem)
#    b_ = retrieve(b,mem)
    # the similarity between two items is the sum of the similarities between 
    # the relationships between the first item and an intermediary
    sims = np.zeros((len(dictionary), len(dictionary)))
    for i in np.arange(len(dictionary)):
        for j in np.arange(i+1,len(dictionary)):
            # skip in i == j
            
            for t in sorted(list(set(dictionary) - \
                                 set([dictionary[i], dictionary[j]]))):
                ret_i_t = retrieve(set([dictionary[i], t]), mem)
                ret_j_t = retrieve(set([dictionary[j], t]), mem)
                sim_out = sim(ret_i_t, ret_j_t)
                sims[i, j] += sim_out
                sims[j, i] += sim_out
    np.fill_diagonal(sims,1)
    sims = sims/sims.max(1)
    np.fill_diagonal(sims,1)
    
    out = pd.DataFrame(sims)
    out.columns = dictionary
    out.index = dictionary
    return out

# make function to return echo of item
# where return = echo(item) / item
# return val is a set
def retrieve(items, mem):

#    out = set().union(*[statement for statement in mem if items.issubset(statement)])
    out = set().union(*[statement - items for statement in mem if items.issubset(statement)])
    return out

# define similarity function as intersection over sum of two sets
def sim(a,b):
    if (len(a) + len(b) - len(a.intersection(b))) == 0:
        return 0
    return len(a.intersection(b)) / (len(a) + len(b) - len(a.intersection(b)))
    


for path in listdir('../distributions/uniform/')[:]:
#    out = ''
    print(path)
    path = listdir('../distributions/uniform/')[4]
    path_out = '../distributions/uniform/' + path
    with open(path_out,'r') as file:
        statements = file.readlines()
    out = train_model_get_cosine_matrix(statements)
    out.to_csv('../cosines/uniform/exemplarSet_'+ path.split('.')[0] + '.csv', index = False)
    
for path in listdir('../distributions/distance/'):
    print(path)
    path_out = '../distributions/distance/' + path
    with open(path_out,'r') as file:
        statements = file.readlines()
    out = train_model_get_cosine_matrix(statements)
    out.to_csv('../cosines/distance/exemplarSet_'+ path.split('.')[0] + '.csv', index = False)