import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import pickle as pkl
import time
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax
from scipy.stats import entropy
from matplotlib2tikz import save as tikz_save

f = open('sound_changes.pkl','rb')
change_list,S,X,R,N,L,partition,lang_ind,sound_ind = pkl.load(f)
f.close()

K = 10

f = open('feat_matrix.csv','r')
feat_text = f.read()
f.close()

feat_text = [l.split('\t') for l in feat_text.split('\n')]
F = len(feat_text[0][1:])

feats = {l[0]:l[1:] for l in feat_text}


def gen_grad_dist_mats_env():
    delta_change = np.zeros([F,S,S])
    delta_env = np.zeros([F,S,S])
    for d in range(F):
        for i in range(len(change_list)):
            for j in range(i):
                if [feats[s][d] for s in [change_list[i][0][1]]+list(change_list[i][1])] != [feats[s][d] for s in [change_list[j][0][1]]+list(change_list[j][1])]:
                    delta_change[d,i,j] = delta_change[d,j,i] = 1
                if feats[change_list[i][0][0]][d] != feats[change_list[j][0][0]][d] or feats[change_list[i][0][2]][d] != feats[change_list[j][0][2]][d]:
                    delta_env[d,i,j] = delta_env[d,j,i] = 1
    delta = np.concatenate([delta_change,delta_env])
    return(delta)



def gen_bin_dists_mats_env():
    delta_change = np.zeros([S,S])
    delta_env = np.zeros([S,S])
    for i in range(len(change_list)):
        for j in range(i):
            if change_list[i][0][1] != change_list[j][0][1] or change_list[i][1] != change_list[j][1]:
                delta_change[i,j] = delta_change[j,i] = 1
            if change_list[i][0][0] != change_list[j][0][0] or change_list[i][0][2] != change_list[j][0][2]:
                delta_env[i,j] = delta_env[j,i] = 1
    return(np.stack([delta_change,delta_env]))



delta_grad = gen_grad_dist_mats_env()
D_grad = delta_grad.shape[0]
delta_bin = gen_bin_dists_mats_env()
D_bin = delta_bin.shape[0]


f=open('GP_bin_env_var_params.pkl','rb')
post_bin=pkl.load(f)
f.close()


f=open('diag_var_params.pkl','rb')
post_diag=pkl.load(f)
f.close()


f=open('GP_grad_env_var_params.pkl','rb')
post_grad=pkl.load(f)
f.close()




N = 100
curr = {k:[np.random.normal(loc=post_bin[c][k][0],scale=post_bin[c][k][1]) for c in range(len(post_bin)) for i in range(N)] for k in post_bin[0].keys()}
for d in range(D_bin):
    plt.hist([np.power(curr['inv_rhos'][i][d][0][0],2) for i in range(N*len(post_bin))],alpha=.4)


tikz_save('bin_inv_rho.tex')
        



N = 100
curr = {k:[np.random.normal(loc=post_grad[c][k][0],scale=post_grad[c][k][1]) for c in range(len(post_grad)) for i in range(N)] for k in post_grad[0].keys()}
for d in range(D_grad):
    plt.hist([np.power(curr['inv_rhos'][i][d][0][0],2) for i in range(N*len(post_bin))],alpha=.4)


tikz_save('grad_inv_rho.tex')

rhos = []
for d in range(D_grad):
    rhos.append([np.power(curr['inv_rhos'][i][d][0][0],2) for i in range(N*len(post_bin))])
