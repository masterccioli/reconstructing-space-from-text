# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:51:07 2019

@author: maste

Given corpus - parsed for words - create a WxD sparse matrix with dictionary
option: stemming
"""

from re import sub
import pickle
from scipy import sparse
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize
stemmer = SnowballStemmer("english")
import sys
import numpy as np

# Generate Sparse WxD, Dictionary
def loadCorpus(path, stem = False, is_parsed = False, window_length = False):
    if is_parsed:
        with open(path,'r') as file:
            corpus = pickle.load(file)
    else:
        ofile = open(path, 'r')
        corpus = []
        for line in ofile:
            line = sub('\n', '', line)
            line = sub('\r', '', line)
            line = sent_tokenize(line)
            line = [word_tokenize(i) for i in line]
            corpus.extend(line)
        ofile.close()
        
    if window_length:
        # make a scrolling window - width=4 words
        
        # need to flatten corpus
        corpus = [word for line in corpus for word in line]
        
        windowed_text = []
        for i in np.arange(len(corpus) - window_length):
            windowed_text.append(corpus[i:i + window_length])
        corpus = windowed_text
        
    # parse to get dictionary 
    c = []
    for corp in corpus:
        c.extend(corp)
    if stem:
        c = [stemmer.stem(i) for i in c]

    mydict = sorted(set(c))
    mydict = dict(zip(mydict, range(0, len(mydict))))

    #map to get key/values from dictionary -col
    col = list(map(mydict.get, c))

    #map to get row values
    row = []
    for index,corp in enumerate(corpus):
        row.extend([index] * len(corp))

    data = [1] * len(row)

    wd = sparse.csr_matrix((data, (row, col)), shape=(len(corpus),len(mydict)),dtype='float32')
    return wd,mydict

# Generate Sparse WxD, Dictionary
def loadCorpus_from_list(corpus, window_length = False):
    if window_length:
        # make a scrolling window - width=4 words
        
        # need to flatten corpus
        corpus = [word for line in corpus for word in line]
        
        windowed_text = []
        for i in np.arange(len(corpus) - window_length):
            windowed_text.append(corpus[i:i + window_length])
        corpus = windowed_text
        
    # parse to get dictionary 
    corpus = [statement.split(' ') for statement in corpus]
    c = []
    for corp in corpus:
        c.extend(corp)

    mydict = sorted(set(c))
    mydict = dict(zip(mydict, range(0, len(mydict))))

    #map to get key/values from dictionary -col
    col = list(map(mydict.get, c))

    #map to get row values
    row = []
    for index,corp in enumerate(corpus):
        row.extend([index] * len(corp))

    data = [1] * len(row)

    wd = sparse.csr_matrix((data, (row, col)), shape=(len(corpus),len(mydict)),dtype='float32')
    return wd,mydict

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: python get_wd_dict.py <text corpus> <processed_text_file_name>')
        sys.exit(1)
    in_f = sys.argv[1]
    
    stem = 'stem' in sys.argv
    is_parsed = 'parsed' in sys.argv
    
    wd,mydict = loadCorpus(in_f, stem, is_parsed)
    
    out_f = sys.argv[2]
    with open(sys.argv[2], 'wb') as out_f:
        pickle.dump([wd,mydict], out_f)