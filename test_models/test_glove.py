# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:39:43 2019

@author: maste
"""
# LSA

#import modules
from os import listdir
import pandas as pd
import numpy as np

def cosine(compare):
    return np.dot(compare,compare.transpose()) / np.outer(np.sqrt(np.sum(compare*compare,1)),np.sqrt(np.sum(compare*compare,1)))

if __name__ == "__main__":
    for path in listdir('GloVe/vectors/distance/'):
        print(path)
        if path.split('.')[-1] == 'txt':
        #    path = listdir('glove_vectors/distance/')[4]
            vects = pd.read_csv('GloVe/vectors/distance/' + path, header = None, sep = ' ')
            vects = vects[:-1]
            vects = vects.sort_values(by=[0])
            words = sorted(list(vects.loc[:,vects.columns == 0][0]))
            out = pd.DataFrame(cosine(vects.loc[:, vects.columns != 0]))
            out.columns = list(vects.loc[:,vects.columns == 0][0])
            out.to_csv('../cosines/distance/glove_'+ path.split('.')[0] + '.csv', index = False)
