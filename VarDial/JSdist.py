#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import tensorflow as tf
import pickle as pkl
import mpu
from scipy.stats import spearmanr,entropy


text = []

for l in open('cdial_stripped.csv','r'):
    text.append(l.strip().split('\t'))


for i in range(len(text)):
    for j in range(1,3):
        text[i][j] = text[i][j].split()


etym_counts = defaultdict(int)
for l in text:
    etym_counts[tuple(l[2])]+=1



alignments = []
for l in open('alignments.txt','r'):
    alignments.append([int (i) for i in l.split()])


langs_raw = []
inputs_raw = []
outputs_raw = []
for i,l in enumerate(text):
  if l[0] != 'Pa.' and l[0] != 'Pk.':
    inputs = []
    outputs = []
    x,y=text[i][2],text[i][1]
    A = alignments[i]
    lang = l[0]
    etymon = ''.join(l[2])
    for j in range(1,len(A)-2):
        inputs.append(''.join(x[j-1:j+2]))
        outputs.append(''.join(y[A[j]:A[j+1]]))
    langs_raw.append(lang)
    inputs_raw.append(inputs)
    outputs_raw.append(outputs)


reflex_list = defaultdict(list)
for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        reflex_list[s].append(outputs_raw[i][j])


for k in reflex_list.keys():
    reflex_list[k] = sorted(set(reflex_list[k]))


outcomes = {}
for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        if s not in outcomes.keys():
            outcomes[s] = defaultdict(list)
        if langs_raw[i] not in outcomes[s].keys():
            outcomes[s][langs_raw[i]] = np.zeros(len(reflex_list[s]))
        outcomes[s][langs_raw[i]][reflex_list[s].index(outputs_raw[i][j])] += 1



langs_list = sorted(set([l for k in outcomes.keys() for l in outcomes[k].keys()]))
L = len(langs_list)


dists = defaultdict(list)
for i in range(L):
    for j in range(i+1,L):
        for k in outcomes.keys():
            if langs_list[i] in outcomes[k].keys() and langs_list[j] in outcomes[k].keys():
                dist1 = outcomes[k][langs_list[i]] + 1e-10
                dist2 = outcomes[k][langs_list[j]] + 1e-10
                M = .5*(dist1+dist2)
                dists[(i,j)].append(.5*(entropy(dist1,M)+entropy(dist2,M)))



avgdists = {}
for k in dists.keys():
    avgdists[k] = np.mean(dists[k])



coph_dists_full = {}
for l in open('patristic.txt','r'):
    line = l.strip().split()
    coph_dists_full[(langs_list.index(line[0]),langs_list.index(line[1]))] = float(line[2])



coph_dists = {}
for k in avgdists.keys():
    coph_dists[k] = coph_dists_full[k]



locations = []
for l in open('glottolog_IE.csv','r'):
    locations.append(l.strip().split('\t'))


lon = {}
lat = {}
for l in locations:
    if len(l) > 7:
        lon[l[0]] = float(l[7])
        lat[l[0]] = float(l[6])



geo_dists = {}
for i,j in avgdists.keys():
    geo_dists[(i,j)]=mpu.haversine_distance((lat[langs_list[i]],lon[langs_list[i]]),(lat[langs_list[j]],lon[langs_list[j]]))



cor_geo = spearmanr(list(geo_dists.values()),list(avgdists.values()))
cor_gen = spearmanr(list(coph_dists.values()),list(avgdists.values()))
print('geographic correlation: {}, {}'.format(cor_geo[0],cor_geo[1]))
print('genetic correlation: {}, {}'.format(cor_gen[0],cor_gen[1]))