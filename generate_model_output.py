# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:10:42 2020

@author: maste
"""

from subprocess import call, run
import os

# Generate Corpus
print('Making Corpus')
os.chdir('distributions/corpus_generation/')
call('python corpus_gen.py')

# test models
print('Testing Models')
os.chdir('../../test_models/')

# LSA
print('LSA')
call('python test_lsa.py')

# GloVe
print('GloVe')
# get vectors
os.chdir('GloVe/')
call('bash -c ./run_GloVe.sh')
# get cosines
os.chdir('../')
call('python test_glove.py')

# PPMI
print('PPMI')
call('python test_pmi.py')

# test w2v
print('W2V')
call('python test_w2v.py')

# test exemplar
print('Exemplar')
call('python test_exemplar.py')

print('Done.')