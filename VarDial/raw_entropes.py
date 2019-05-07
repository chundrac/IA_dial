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
        if langs_raw[i] not in outcomes.keys():
            outcomes[langs_raw[i]] = defaultdict(list)
        if s not in outcomes[langs_raw[i]].keys():
            outcomes[langs_raw[i]][s] = np.zeros(len(reflex_list[s]))
        outcomes[langs_raw[i]][s][reflex_list[s].index(outputs_raw[i][j])] += 1



outcomes = {}
for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        if s not in outcomes.keys():
            outcomes[s] = defaultdict(list)
        if langs_raw[i] not in outcomes[s].keys():
            outcomes[s][langs_raw[i]] = np.zeros(len(reflex_list[s]))
        outcomes[s][langs_raw[i]][reflex_list[s].index(outputs_raw[i][j])] += 1



entropies = defaultdict(list)
for k in outcomes.keys():
    for l in outcomes[k].keys():
        if len(outcomes[k][l]) > 1:
            H = entropy(outcomes[k][l]/sum(outcomes[k][l]))/entropy(np.ones(len(outcomes[k][l]))/len(outcomes[k][l]))
        else:
            H = 0
        entropies[l].append(H)


np.mean([v for k in entropies.keys() for v in entropies[k]])
np.mean([np.mean(entropies[k]) for k in entropies.keys()])

