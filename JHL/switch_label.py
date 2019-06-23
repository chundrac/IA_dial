from __future__ import division
import os
import codecs
import numpy as np
import random
import itertools
from collections import defaultdict
from numpy.random import multinomial,normal,uniform,multivariate_normal
from numpy import log,exp,mean
from scipy.special import gamma
import re
import pymc3 as pm
from pymc3.math import logsumexp
from sklearn.utils.extmath import softmax
from functools import reduce
import theano.tensor as tt
import theano
theano.config.gcc.cxxflags = "-fbracket-depth=6000"
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from scipy.stats import entropy
import pickle as pkl


f = open('data_variables.pkl','rb')
segs_to_keep,changes_pruned,Cutoff,change_list,reflex,Sigma,K,S,X,R,N,langs,L,lang_ind,sound_ind,s_breaks,nchains = pkl.load(f)
f.close()

assert(nchains==4)
nchains = 5

posterior_ln = defaultdict()
posterior_dir = defaultdict()
for c in range(nchains):
    posterior_ln[c] = defaultdict(list)
    f = open('posterior_ln_{}.pkl'.format(c),'rb')
    posterior_ln[c] = pkl.load(f)
    f.close()
    posterior_dir[c] = defaultdict(list)
    f = open('posterior_dir_{}.pkl'.format(c),'rb')
    posterior_dir[c] = pkl.load(f)
    f.close()


def JS_divergence(P,Q):
    M = .5*(P+Q)
    return(0.5 * (entropy(P, M) + entropy(Q, M)))



switch_dir = {}
switch_ln = {}


for c in range(1,nchains):
    JS_flip_ln = np.mean([JS_divergence(np.flip(np.mean(posterior_ln[c][k],0),0),np.mean(posterior_ln[0][k],0)) for k in posterior_ln[c].keys() if k.startswith('theta_')])
    JS_orig_ln = np.mean([JS_divergence(np.mean(posterior_ln[c][k],0),np.mean(posterior_ln[0][k],0)) for k in posterior_ln[c].keys() if k.startswith('theta_')])
    JS_flip_dir = np.mean([JS_divergence(np.flip(np.mean(posterior_dir[c][k],0),0),np.mean(posterior_dir[0][k],0)) for k in posterior_dir[c].keys() if k.startswith('theta_')])
    JS_orig_dir = np.mean([JS_divergence(np.mean(posterior_dir[c][k],0),np.mean(posterior_dir[0][k],0)) for k in posterior_dir[c].keys() if k.startswith('theta_')])
    if JS_flip_ln < JS_orig_ln:
        switch_ln[c] = 'yes'
    else:
        switch_ln[c] = 'no'
    if JS_flip_dir < JS_orig_dir:
        switch_dir[c] = 'yes'
    else:
        switch_dir[c] = 'no'
        


[JS_divergence(np.flip(np.mean(posterior_ln[c][k],0),0),np.mean(posterior_ln[0][k],0)) for k in posterior_ln[c].keys() if k.startswith('theta')]


new_posterior_ln = {c:defaultdict(list) for c in range(nchains)}
new_posterior_dir = {c:defaultdict(list) for c in range(nchains)}


for k in posterior_dir[0].keys():
    if k=='beta' or k=='ELBO' or k.startswith('phi') or k.startswith('theta'):
        new_posterior_dir[0][k] = posterior_dir[0][k]


for c in range(1,nchains):
    if switch_dir[c] == 'yes':
        new_posterior_dir[c]['beta'] = posterior_dir[c]['beta']
        new_posterior_dir[c]['ELBO'] = posterior_dir[c]['ELBO']
        for k in posterior_dir[c].keys():
            if k.startswith('theta'):
                new_posterior_dir[c][k] = np.flip(posterior_dir[c][k],1)
            if k.startswith('phi_0'):
                new_posterior_dir[c][k.replace('phi_0','phi_1')] = posterior_dir[c][k]
            if k.startswith('phi_1'):
                new_posterior_dir[c][k.replace('phi_1','phi_0')] = posterior_dir[c][k]
    else:
        for k in posterior_dir[c].keys():
            if k=='beta' or k=='ELBO' or k.startswith('phi') or k.startswith('theta'):
                new_posterior_dir[c][k] = posterior_dir[c][k]


for k in posterior_ln[0].keys():
    if k=='beta' or k=='ELBO' or k.startswith('phi') or k.startswith('theta'):
        new_posterior_ln[0][k] = posterior_ln[0][k]


for c in range(1,nchains):
    if switch_ln[c] == 'yes':
        new_posterior_ln[c]['beta'] = posterior_ln[c]['beta']
        new_posterior_ln[c]['ELBO'] = posterior_ln[c]['ELBO']
        for k in posterior_dir[c].keys():
            if k.startswith('theta'):
                new_posterior_ln[c][k] = np.flip(posterior_ln[c][k],1)
            if k.startswith('phi_0'):
                new_posterior_ln[c][k.replace('phi_0','phi_1')] = posterior_ln[c][k]
            if k.startswith('phi_1'):
                new_posterior_ln[c][k.replace('phi_1','phi_0')] = posterior_ln[c][k]
    else:
        for k in posterior_ln[c].keys():
            if k=='beta' or k=='ELBO' or k.startswith('phi') or k.startswith('theta'):
                new_posterior_ln[c][k] = posterior_ln[c][k]


f = open('posterior_dir_full.pkl','wb')
pkl.dump(new_posterior_dir,f)
f.close()


f = open('posterior_ln_full.pkl','wb')
pkl.dump(new_posterior_ln,f)
f.close()
