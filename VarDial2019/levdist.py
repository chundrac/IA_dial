import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import time
import mpu
from scipy.stats import spearmanr


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        s2, s1 = s1, s2
    N = len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]/N


text = []

for l in open('cdial_stripped.csv','r'):
    text.append(l.strip().split('\t'))


cogs = {}
for l in text:
    if l[0] != 'Pa.' and l[0] != 'Pk.':
        lang = l[0]
        etym = tuple(l[2].split())
        reflex = tuple(l[1].split())
        if etym not in cogs.keys():
            cogs[etym] = {}
        cogs[etym][lang] = reflex


langs_list = sorted(set([l for k in cogs.keys() for l in cogs[k].keys()]))
L = len(langs_list)

dists = defaultdict(list)
for i in range(L):
    for j in range(i+1,L):
        for k in cogs.keys():
            if langs_list[i] in cogs[k].keys() and langs_list[j] in cogs[k].keys():
                dists[(i,j)].append(levenshtein(cogs[k][langs_list[i]],cogs[k][langs_list[j]]))



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