# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:04:18 2019

@author: maste
"""

import string
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd


def make_shape(sides, offset): # proceeds counter clockwise
    theta = np.arange(0, 360, 360/sides) + offset
    y = np.sin(theta * np.pi / 180)
    x = np.cos(theta * np.pi / 180)
    return np.vstack([x, y]).transpose()

def make_cluster(num_items, x_center, y_center):
    x = np.random.normal(loc=x_center, scale=.5, size=num_items)
    y = np.random.normal(loc=y_center, scale=.5, size=num_items)
    return np.vstack([x, y]).transpose()

def get_angles(shape):
    # get pairwise slopes
    x = np.subtract.outer(shape[:, 0], shape[:, 0])
    y = np.subtract.outer(shape[:, 1], shape[:, 1])
    angles = np.round(np.arctan2(y, x) / np.pi * 180, 2)
    # angles above diagonal need to be added to 180 degrees
    angles = angles + 180
    angles[np.where(angles == 360)] = 0
    return angles

def get_probabilities_from_distances(shape, corpus_size):
    x = np.subtract.outer(shape[:, 0], shape[:, 0])
    x = np.power(x, 2)
    y = np.subtract.outer(shape[:, 1], shape[:, 1])
    y = np.power(y, 2)
    out = np.power(x + y, 1/2)

    out = (np.max(out, 1) + 1 - out)
    np.fill_diagonal(out, 0)
    out = np.ndarray.flatten(out)
    out = out[np.where(out != 0)]
    out = out / np.sum(out)
    return out

# make boundaries
def get_boundaries(num_partitions, rotation, relationships):
    # rotation must be less than 360/divisor
    out = np.arange(0, 360+1, 360/num_partitions) + rotation
    out = out[np.where(out <= 360)]
    if not 0 in out:
        out = np.hstack([[0], out])
    if not 360 in out:
        out = np.hstack([out, [360]])

    return [[out[i], out[i+1], relationships[i]] for i in \
            np.arange(len(out) - 1)]

def get_statements_angles(angles, boundaries, multiplier=1):
    vals = []
    for i in np.arange(angles.shape[0]):
        for j in np.arange(angles.shape[0]):
            if i == j:
                continue
            for boundary in boundaries:
                if (boundary[0] < angles[i, j]) and (angles[i, j] < boundary[1]):
                    vals.extend([letters[j] + ' ' + boundary[2] + ' ' + \
                                 letters[i]] * int(multiplier))
                    continue
    return vals

def get_statements_distances(points, multiplier=1):
    dists = euclidean_distances(points)
    mean_dist = np.mean(dists)
    vals = []
    for i in np.arange(dists.shape[0]):
        for j in np.arange(dists.shape[0]):
            if i == j:
                continue
            if (mean_dist < dists[i, j]):
                vals.extend([letters[j] + ' far_from ' + letters[i]] * int(multiplier))
            else:
                vals.extend([letters[j] + ' near_to ' + letters[i]] * int(multiplier))
    return vals

def get_statements_absolute(points, multiplier=1):
    vals = []
    for ind, i in enumerate(points):
        if (i[0] > 0) & (i[1] > 0):
            vals.extend([letters[ind] + ' is_north'] * int(multiplier))
        elif (i[0] > 0) & (i[1] < 0):
            vals.extend([letters[ind] + ' is_east'] * int(multiplier))
        elif (i[0] < 0) & (i[1] < 0):
            vals.extend([letters[ind] + ' is_south'] * int(multiplier))
        elif (i[0] < 0) & (i[1] > 0):
            vals.extend([letters[ind] + ' is_west'] * int(multiplier))
    return vals

def get_statements_north_south(points, multiplier=1):
    vals = []
    for ind, i in enumerate(points):
        if i[1] > 0:
            vals.extend([letters[ind] + ' is_north'] * int(multiplier))
        elif i[1] < 0:
            vals.extend([letters[ind] + ' is_south'] * int(multiplier))
    return vals

def sample_statements(statements, corpus_size, probabilities=False):
    if isinstance(probabilities, bool):
        # sample each statement with equal probability
        corpus = [str(i) for i in np.random.choice(statements, corpus_size)]
    else:
        # sample statements based on probabilities
        corpus = [str(i) for i in np.random.choice(statements, corpus_size, p=probabilities)]
    return corpus

def save_statements_as_txt(path, statements):
    with open(path, 'w') as filehandle:
        for listitem in statements:
            filehandle.write('%s\n' % listitem)

def make_output(statements,
                distribution_name,
                relationship_name,
                corpus_size,
                sampling_distribution):
    #distance
    # sample statements according to distribution
    sample = sample_statements(statements, corpus_size, sampling_distribution)

    # save statements
    save_statements_as_txt('../../test_models/GloVe/corpus/distance/' +
                            distribution_name + '_' +
                            relationship_name +
                            '.txt', sample)
    save_statements_as_txt('../distance/'  +
                           distribution_name + '_' +
                           relationship_name +
                           '.txt', sample)
    #uniform
    # sample statements according to distribution
    sample = sample_statements(statements, corpus_size)

    # save statements
    save_statements_as_txt('../../test_models/GloVe/corpus/uniform/' +
                            distribution_name + '_' +
                            relationship_name +
                            '.txt', sample)
    save_statements_as_txt('../uniform/'  +
                           distribution_name + '_' +
                           relationship_name +
                           '.txt', sample)

if __name__ == "__main__":
    corpus_size = 10000

    # make boundary sets for angular relationships
    nsew = ['east_of', 'north_of', 'west_of', 'south_of', 'east_of']
    nsew = get_boundaries(4, 45, nsew)

    # make point distributions
    num_points = 20
    letters = string.ascii_uppercase[:num_points]
    shape_20 = make_shape(num_points, 1)
    cluster_1 = make_cluster(num_points, 0, 0)
    cluster_2 = np.vstack([make_cluster(int(num_points/2), 1, 1), \
                           make_cluster(int(num_points/2), -1, -1)])

    # save point distributions
    out = pd.DataFrame(shape_20)
    out.columns = ['indepV1', 'indepV2']
    out.to_csv('../points/shape_'+str(num_points)+'.csv', index=None)
    out = pd.DataFrame(cluster_1)
    out.columns = ['indepV1', 'indepV2']
    out.to_csv('../points/cluster1_'+str(num_points)+'.csv', index=None)
    out = pd.DataFrame(cluster_2)
    out.columns = ['indepV1', 'indepV2']
    out.to_csv('../points/cluster2_'+str(num_points)+'.csv', index=None)

    # get angles for each distribution
    angles_shape = get_angles(shape_20)
    angles_cluster_1 = get_angles(cluster_1)
    angles_cluster_2 = get_angles(cluster_2)

    # get distances for each distribution
    dists_shape = get_probabilities_from_distances(shape_20, corpus_size)
    dists_cluster_1 = get_probabilities_from_distances(cluster_1, corpus_size)
    dists_cluster_2 = get_probabilities_from_distances(cluster_2, corpus_size)

    # for north/south/east/west
    statements_shape_nsew = get_statements_angles(angles_shape, nsew)
    statements_cluster_1_nsew = get_statements_angles(angles_cluster_1, nsew)
    statements_cluster_2_nsew = get_statements_angles(angles_cluster_2, nsew)

    # for near/far
    statements_shape_nf = get_statements_distances(shape_20)
    statements_cluster_1_nf = get_statements_distances(cluster_1)
    statements_cluster_2_nf = get_statements_distances(cluster_2)

    ##############
    # Sample normally
    # for north/south/east/west
    make_output(statements_shape_nsew,
                'shape',
                'nsew',
                corpus_size,
                dists_shape)
    make_output(statements_cluster_1_nsew,
                'cluster1',
                'nsew',
                corpus_size,
                dists_cluster_1)
    make_output(statements_cluster_2_nsew,
                'cluster2',
                'nsew',
                corpus_size,
                dists_cluster_2)
    # for near/far
    make_output(statements_shape_nf,
                'shape',
                'nf',
                corpus_size,
                dists_shape)
    make_output(statements_cluster_1_nf,
                'cluster1',
                'nf',
                corpus_size,
                dists_cluster_1)
    make_output(statements_cluster_2_nf,
                'cluster2',
                'nf',
                corpus_size,
                dists_cluster_2)

    # both combined
    make_output(statements_shape_nsew+statements_shape_nf,
                'shape',
                'nsew_nf',
                corpus_size,
                np.hstack([dists_shape,dists_shape])/2)
    make_output(statements_cluster_1_nsew+statements_cluster_1_nf,
                'cluster1',
                'nsew_nf',
                corpus_size,
                np.hstack([dists_cluster_1,dists_cluster_1])/2)
    make_output(statements_cluster_2_nsew+statements_cluster_2_nf,
                'cluster2',
                'nsew_nf',
                corpus_size,
                np.hstack([dists_cluster_2,dists_cluster_2])/2)
