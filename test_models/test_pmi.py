# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:53:36 2019

@author: maste
"""

#import modules
from gensim.models.phrases import npmi_scorer
import pickle
from nltk import FreqDist, ConditionalFreqDist
from os import listdir
import numpy as np
import pandas as pd

def train_model_get_cosine_matrix(statements):
    statements = [statement.split() for statement in statements]

    frequencies = FreqDist(w for word in statements for w in word)

    conditionalFrequencies = ConditionalFreqDist(
                                (key,word)
                                for key in sorted(frequencies.keys())
                                for statement in statements
                                for word in statement 
                                if key in statement)
        
    pmi = [[npmi_scorer(frequencies[worda], 
                  frequencies[wordb], 
                  conditionalFrequencies[worda][wordb], 
                  len(frequencies.keys()),
                  2,
                  sum(frequencies[key] for key in frequencies.keys()))
        for wordb in sorted(frequencies.keys())]
        for worda in sorted(frequencies.keys())]
        
        
    pmi = np.array(pmi)
    pmi[np.isinf(pmi)] = -1
    pmi[np.where(pmi < 0)] = 0
        
    pmi = pd.DataFrame(pmi)
    pmi.columns = sorted(frequencies.keys())
    pmi.index = sorted(frequencies.keys())

    return pmi
if __name__ == "__main__":
    for path in listdir('../distributions/distance/'):
        print(path)
        path_out = '../distributions/distance/' + path
        with open(path_out,'r') as file:
            statements = file.readlines()
        out = train_model_get_cosine_matrix(statements)
        out.to_csv('../cosines/distance/pmi_'+ path.split('.')[0] + '.csv', index = False)