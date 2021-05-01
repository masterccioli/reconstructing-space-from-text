# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:08:31 2021

@author: maste
"""

import os

def remove_files():
    for file in os.listdir():
        print(file)
        os.remove(file)


os.chdir('distributions/distance/')
remove_files()

os.chdir('../uniform/')
remove_files()

os.chdir('../points/')
remove_files()

os.chdir('../../cosines/distance/')
remove_files()

os.chdir('../uniform/')
remove_files()

os.chdir('../../r_evaluation/bidim_output/')
remove_files()

os.chdir('../plots/distance/')
remove_files()

os.chdir('../uniform/')
remove_files()

os.chdir('../original/')
remove_files()

os.chdir('../original/')
remove_files()

os.chdir('../../../test_models/GloVe/corpus/distance/')
remove_files()

os.chdir('../uniform/')
remove_files()

os.chdir('../../vectors/distance/')
remove_files()

os.chdir('../uniform/')
remove_files()

os.chdir('../../../../')
