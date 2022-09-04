#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:17:05 2022

@author: momo
"""

import pickle
import pandas as pd
import numpy as np

from math import log2

from sklearn.cluster import SpectralClustering

#hyper-params
K = 50 #clusters


# load
with open('movie_index.pickle', 'rb') as f:
    movie_index = pickle.load(f)
    
with open('tmp.pickle', 'rb') as f:
    tmp = pickle.load(f)

names_col = pd.DataFrame(movie_index['movie_name'])
prob_col = pd.DataFrame(tmp)

vectors = pd.concat([names_col, prob_col], axis=1)

names = names_col['movie_name'].unique()

avg_vectors = pd.DataFrame(columns=vectors.columns)
counter = 0
for i in names:
    v_for_a_movie = vectors[vectors['movie_name']==i]
    row = []
    row.append(v_for_a_movie['movie_name'].iloc[0])
    for j in range(50):
        row.append(v_for_a_movie[j].mean())
    row = pd.DataFrame(row).transpose()
    row.columns = avg_vectors.columns
    avg_vectors = pd.concat([avg_vectors, row], axis=0)
    counter += 1
    print(counter)
    
avg_vectors = avg_vectors.reset_index(drop=True)
aff = np.zeros((avg_vectors.shape[0],avg_vectors.shape[0]))

values = avg_vectors.iloc[:,1:]

def KL(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def JS(p, q):
    m = 0.5 * (p + q)
    return 0.5 * KL(p, m) + 0.5 * KL(q, m)



for i in range(aff.shape[0]):
    for j in range(aff.shape[0]):
        if i <= j:
            aff[i][j] = JS(values.iloc[i] , values.iloc[j])
    print(i/aff.shape[0])
    

aff = aff + aff.transpose()

sc = SpectralClustering(K, affinity='precomputed', n_init=100, assign_labels='discretize')
sc.fit_predict(aff)  

results = sc.labels_

outputs= pd.DataFrame(columns=['movie_name','cluster_label'])
outputs['movie_name'] = avg_vectors['movie_name']
outputs['cluster_label'] = pd.DataFrame(results, columns=['cluster_label'])

with open('clusters_labels.pickle', 'wb') as f:
    pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)