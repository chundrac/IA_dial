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



def softmax2d(x):
    x = x-x.max(axis=1,keepdims=True)
    return(np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True))



def relabel_diag():
    f = open('diag_var_params.pkl','rb')
    post = pkl.load(f)
    f.close()
    theta_hat = {}
    for c in post.keys():
        theta_hat[c] = softmax2d(post[c]['eta'][0])
    perms = list(itertools.chain(itertools.permutations(range(10))))
    for i in range(1,2):
        perm_KL = []
        for p in perms:
            perm_KL.append(sum(entropy(theta_hat[i].T[p,:],theta_hat[0].T)))
        perm = perms[perm_KL.index(min(perm_KL))]
        post[i]['eta'] = (post[i]['eta'][0][:,perm],post[i]['eta'][1][:,perm])
        post[i]['psi'] = (post[i]['psi'][0][perm,:],post[i]['psi'][1][perm,:])
    f = open('diag_var_params_relabeled.pkl','wb')
    pkl.dump(post,f)



def relabel_BGP():
    f = open('BGP_var_params.pkl','rb')
    post = pkl.load(f)
    f.close()
    theta_hat = {}
    for c in post.keys():
        theta_hat[c] = softmax2d(post[c]['eta'][0])
    perms = list(itertools.chain(itertools.permutations(range(10))))
    for i in range(1,2):
        perm_KL = []
        for p in perms:
            perm_KL.append(sum(entropy(theta_hat[i].T[p,:],theta_hat[0].T)))
        perm = perms[perm_KL.index(min(perm_KL))]
        post[i]['eta'] = (post[i]['eta'][0][:,perm],post[i]['eta'][1][:,perm])
        post[i]['psi'] = (post[i]['psi'][0][perm,:],post[i]['psi'][1][perm,:])
    f = open('BGP_var_params_relabeled.pkl','wb')
    pkl.dump(post,f)



def relabel_GGP():
    f = open('GGP_var_params.pkl','rb')
    post = pkl.load(f)
    f.close()
    theta_hat = {}
    for c in post.keys():
        theta_hat[c] = softmax2d(post[c]['eta'][0])
    perms = list(itertools.chain(itertools.permutations(range(10))))
    for i in range(1,2):
        perm_KL = []
        for p in perms:
            perm_KL.append(sum(entropy(theta_hat[i].T[p,:],theta_hat[0].T)))
        perm = perms[perm_KL.index(min(perm_KL))]
        post[i]['eta'] = (post[i]['eta'][0][:,perm],post[i]['eta'][1][:,perm])
        post[i]['psi'] = (post[i]['psi'][0][perm,:],post[i]['psi'][1][perm,:])
    f = open('GGP_var_params_relabeled.pkl','wb')
    pkl.dump(post,f)



relabel_diag()
relabel_BGP()
relabel_GGP()