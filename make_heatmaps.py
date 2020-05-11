# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:31:28 2020

@author: maste
"""
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir


from jaccard import generalized_jaccard as jac
from jaccard.generate_heatmaps import annotated_heat_maps as ahm
import get_wd


#################### 20 items
# get second order jaccard
wd,mydict = get_wd.loadCorpus('distributions/uniform/direction.txt')
second = jac.get_jaccard_matrix(wd,2,second_order=True)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
second.to_csv('distributions/points/second_order_20.csv',index=False)

first = jac.get_jaccard_matrix(wd,2,second_order=False)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
first.to_csv('distributions/points/first_order_20.csv',index=False)

# plot first order
img_size = 20
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(first, list(mydict.keys()), list(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/first_order_20.png', dpi=100, transparent = True)

# plot second order
img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(second, list(mydict.keys()), list(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/second_order_20.png', dpi=100, transparent = True)

############################## 8 items
# get second order jaccard
wd,mydict = get_wd.loadCorpus('distributions/uniform/direction_8.txt')
second = jac.get_jaccard_matrix(wd,2,second_order=True)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
second.to_csv('distributions/points/second_order_8.csv',index=False)

first = jac.get_jaccard_matrix(wd,2,second_order=False)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
first.to_csv('distributions/points/first_order_8.csv',index=False)

# get second order hausdorf
wd,mydict = get_wd.loadCorpus('distributions/uniform/direction_8.txt')
items = [jac.get_item_ngram_matrix(ngram_matrix= wd, 
                      item_ind = i, second_order = True, ignore_frequency = False)
            for i in range(0,wd.shape[1])]
second = jac.get_jaccard_matrix_simp(items,jac.hausdorff_euclid)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
second.to_csv('distributions/points/second_order_8.csv',index=False)

first = jac.get_jaccard_matrix(wd,2,second_order=False)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
first.to_csv('distributions/points/first_order_8.csv',index=False)


# make heatmaps for first and second order information
img_size = 20
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(first, list(mydict.keys()), list(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/first_order_8.png', dpi=100, transparent = True)

# plot second order
img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(second, list(mydict.keys()), list(mydict.keys()), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/second_order_8.png', dpi=100, transparent = True)


for path in listdir('cosines/uniform/'):
    print(path)
    matrix = pd.read_csv('cosines/uniform/'+path)
    img_size = 15
    fig, ax = plt.subplots()
    im, cbar = ahm.heatmap(matrix, list(matrix.columns), list(matrix.columns), ax=ax,
                       cmap="gist_heat", cbarlabel='Jaccard Index')
    ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
    fig.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(img_size, img_size)
    #plt.show()
    fig.savefig('heatmaps/'+path+'.png', dpi=100, transparent = True)


# plot glove
glove = pd.read_csv('cosines/uniform/glove_direction.csv')
img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(glove, list(glove.columns), list(glove.columns), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/glove.png', dpi=100, transparent = True)

# plot w2v
w2v = pd.read_csv('cosines/uniform/w2v_direction.csv')
img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(w2v, list(glove.columns), list(glove.columns), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/w2v.png', dpi=100, transparent = True)

# plot w2v
w2v = pd.read_csv('cosines/uniform/w2v_direction.csv')
img_size = 15
fig, ax = plt.subplots()
im, cbar = ahm.heatmap(w2v, list(glove.columns), list(glove.columns), ax=ax,
                   cmap="gist_heat", cbarlabel='Jaccard Index')
ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(img_size, img_size)
#plt.show()
fig.savefig('heatmaps/w2v.png', dpi=100, transparent = True)

########################
# get second order hausdorf
wd,mydict = get_wd.loadCorpus('distributions/uniform/shape.txt')
items = [jac.get_item_ngram_matrix(ngram_matrix= wd, 
                      item_ind = i, second_order = True, ignore_frequency = False)
            for i in range(0,wd.shape[1])]
second = jac.get_jaccard_matrix_simp(items,jac.get_jaccard_3)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
#second.to_csv('distributions/points/second_order_8.csv',index=False)

# get second order jaccard
wd,mydict = get_wd.loadCorpus('distributions/uniform/shape.txt')
second = jac.get_jaccard_matrix(wd,3,second_order=True)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
#second.to_csv('distributions/points/second_order_20.csv',index=False)


first = jac.get_jaccard_matrix(wd,2,second_order=False)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
first.to_csv('distributions/points/first_order_8.csv',index=False)

import pickle
path_out = '../code/distributions/uniform/shape_nsew.pk'
    square, angles, boundaries, statements = pickle.load(open(path_out,'rb'))
statements = sorted(set(statements))
with open('distributions/uniform/shape.txt','w') as file:
    file.write('\n'.join(statements))
    
########################
# get second order hausdorf
wd,mydict = get_wd.loadCorpus('../../first_second_order/corpus/artificialGrammar.txt')
items = [jac.get_item_ngram_matrix(ngram_matrix= wd, 
                      item_ind = i, second_order = True, ignore_frequency = False)
            for i in range(0,wd.shape[1])]
second = jac.get_jaccard_matrix_simp(items,jac.get_jaccard_3)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
#second.to_csv('distributions/points/second_order_8.csv',index=False)

# get second order jaccard
wd,mydict = get_wd.loadCorpus('distributions/uniform/shape.txt')
second = jac.get_jaccard_matrix(wd,3,second_order=True)
second = pd.DataFrame(second)
second.columns = mydict.keys()
second.index = mydict.keys()
#second.to_csv('distributions/points/second_order_20.csv',index=False)


first = jac.get_jaccard_matrix(wd,2,second_order=False)
first = pd.DataFrame(first)
first.columns = mydict.keys()
first.index = mydict.keys()
first.to_csv('distributions/points/first_order_8.csv',index=False)

first = jac.get_jaccard_matrix(wd,2,second_order=False)

first_ = first.copy()
np.fill_diagonal(first_,0)
first_ = cosine_table_self(first_)

first__a = first.copy()
first__b = first_.copy()
np.fill_diagonal(first__a,0)
np.fill_diagonal(first__b,0)
first__ = cosine_table(first__a,first__b)

first___ = first_.copy()
np.fill_diagonal(first___,0)
first___ = cosine_table_self(first___)

first____ = first___.copy()
np.fill_diagonal(first____,0)
first____ = cosine_table_self(first____)

def cosine_table_self(vects): # get cosine table, input one matrix
    return vects.dot(vects.transpose()) / \
            np.outer(np.sqrt(np.power(vects, 2).sum(1)),
                     np.sqrt(np.power(vects,2).sum(1)))
            
        
def cosine_table(vects_a,vects_b): # get cosine sims between two matrices
    return vects_a.dot(vects_b.transpose()) / \
            np.outer(np.sqrt(np.power(vects_a,2).sum(1)),
                     np.sqrt(np.power(vects_b,2).sum(1)))


import pickle
path_out = '../code/distributions/uniform/shape_nsew.pk'
    square, angles, boundaries, statements = pickle.load(open(path_out,'rb'))
statements = sorted(set(statements))
with open('distributions/uniform/shape.txt','w') as file:
    file.write('\n'.join(statements))