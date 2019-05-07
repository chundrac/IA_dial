import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import pickle as pkl
import mpu
from sklearn.utils.extmath import softmax
from scipy.stats import spearmanr,entropy


f = open('data_and_variables.pkl','rb')
langs_,lens_,inputs_,outputs_,T,N,X,Y,L,lang_list = pkl.load(f)
f.close()


posterior_RNN = {}
posterior_flat = {}


for c in range(3):
    f = open('posterior_discrete_RNN_{}.pkl'.format(c),'rb')
    posterior_RNN[c] = pkl.load(f)
    f.close()


for c in range(3):
    f = open('posterior_discrete_flat_{}.pkl'.format(c),'rb')
    posterior_flat[c] = pkl.load(f)
    f.close()


for c in range(3):
    plt.plot(posterior_RNN[c][0])


plt.savefig('MAP_trace_RNN.pdf')
plt.clf()


for c in range(3):
    plt.plot(posterior_flat[c][0])


plt.savefig('MAP_trace_flat.pdf')
plt.clf()



locations = []
for l in open('glottolog_IE.csv','r'):
    locations.append(l.strip().split('\t'))


lon = {}
lat = {}
for l in locations:
    if len(l) > 7:
        lon[l[0]] = float(l[7])
        lat[l[0]] = float(l[6])
        


#make component map data frames

f_U = softmax(posterior_RNN[2][1][0])
for i,l in enumerate(lang_list):
    print(' '.join([l]+[str(lon[l])]+[str(lat[l])]+list([str(s) for s in f_U[i]])))






JSdivs_flat = defaultdict(list)
for c in range(3):
    f_U = softmax(posterior_flat[c][1][0])
    for i in range(L):
        for j in range(i+1,L):
            f_U_i = f_U[i]
            f_U_j = f_U[j]
            M = .5*(f_U_i+f_U_j)
            JSdivs_flat[(i,j)].append(.5*(entropy(f_U_i,M)+entropy(f_U_j,M)))



avgdists_flat = {}
for k in JSdivs_flat.keys():
    avgdists_flat[k] = np.mean(JSdivs_flat[k])




JSdivs_RNN = defaultdict(list)
for c in range(3):
    f_U = softmax(posterior_RNN[c][1][0])
    for i in range(L):
        for j in range(i+1,L):
            f_U_i = f_U[i]
            f_U_j = f_U[j]
            M = .5*(f_U_i+f_U_j)
            JSdivs_RNN[(i,j)].append(.5*(entropy(f_U_i,M)+entropy(f_U_j,M)))



avgdists_RNN = {}
for k in JSdivs_RNN.keys():
    avgdists_RNN[k] = np.mean(JSdivs_RNN[k])



JS_mat_RNN = np.zeros([L,L])
for k in avgdists_RNN.keys():
    JS_mat_RNN[k[0],k[1]] = avgdists_RNN[k]
    JS_mat_RNN[k[1],k[0]] = avgdists_RNN[k]



coph_dists_full = {}
for l in open('patristic.txt','r'):
    line = l.strip().split()
    if line[0] in lang_list and line[1] in lang_list:
        coph_dists_full[(lang_list.index(line[0]),lang_list.index(line[1]))] = float(line[2])



coph_dists = {}
for k in avgdists_RNN.keys():
    coph_dists[k] = coph_dists_full[k]



geo_dists = {}
for i,j in avgdists_RNN.keys():
    geo_dists[(i,j)]=mpu.haversine_distance((lat[lang_list[i]],lon[lang_list[i]]),(lat[lang_list[j]],lon[lang_list[j]]))



flat_geo = spearmanr(list(avgdists_flat.values()),list(geo_dists.values()))
flat_coph = spearmanr(list(avgdists_flat.values()),list(coph_dists.values()))
RNN_geo = spearmanr(list(avgdists_RNN.values()),list(geo_dists.values()))
RNN_coph = spearmanr(list(avgdists_RNN.values()),list(coph_dists.values()))


print('flat_geo: {} {}'.format(flat_geo[0],flat_geo[1]))
print('flat_coph: {} {}'.format(flat_coph[0],flat_coph[1]))
print('RNN_geo: {} {}'.format(RNN_geo[0],RNN_geo[1]))
print('RNN_coph: {} {}'.format(RNN_coph[0],RNN_coph[1]))