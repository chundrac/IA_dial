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
from scipy.stats import entropy
from functools import reduce
import pickle as pkl
import theano.tensor as tt
import theano
theano.config.gcc.cxxflags = "-fbracket-depth=6000"
#THEANO_FLAGS = "-fbracket-depth=6000"
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save



f = open('data_variables.pkl','rb')
segs_to_keep,changes_pruned,Cutoff,change_list,reflex,Sigma,K,S,X,R,N,langs,L,lang_ind,sound_ind,s_breaks,nchains = pkl.load(f)
f.close()

#assert(nchains==4)

f = open('posterior_ln_full.pkl','rb')
posterior_ln = pkl.load(f)
f.close()


f = open('posterior_dir_full.pkl','rb')
posterior_dir = pkl.load(f)
f.close()


gold_standard = np.array([1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,1,0])

def JSdiv(P,Q):
    M = .5*(P+Q)
    return(0.5 * (entropy(P, M) + entropy(Q, M)))



gold_bin = np.zeros([L,K])
gold_bin[np.arange(L),gold_standard] = 1


theta_ln = np.array([[posterior_ln[0]['theta_{}'.format(i)][t,:] for i in range(L)] for c in range(nchains) for t in range(500)])

theta_dir = np.array([[posterior_dir[0]['theta_{}'.format(i)][t,:] for i in range(L)] for c in range(nchains) for t in range(500)])



KLs_ln = [min([sum([entropy(gold_bin[i,:],theta_ln[t][i,:]) for i in range(L)]),sum([entropy(np.flip(gold_bin[i,:]),theta_ln[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]


#[min([sum([entropy(gold_bin[i,:],np.mean(theta_ln,0)[t][i,:]) for i in range(L)]),sum([entropy(np.flip(gold_bin[i,:]),np.mean(theta_ln,0)[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]


KLs_dir = [min([sum([entropy(gold_bin[i,:],theta_dir[t][i,:]) for i in range(L)]),sum([entropy(np.flip(gold_bin[i,:]),theta_dir[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]


Kls_ln_dir = [min([sum([entropy(theta_ln[t][i,:],theta_dir[t][i,:]) for i in range(L)]),sum([entropy(np.flip(theta_ln[t][i,:]),theta_dir[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]



JSs_ln = [min([sum([JSdiv(gold_bin[i,:],theta_ln[t][i,:]) for i in range(L)]),sum([JSdiv(np.flip(gold_bin[i,:]),theta_ln[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]


JSs_dir = [min([sum([JSdiv(gold_bin[i,:],theta_dir[t][i,:]) for i in range(L)]),sum([JSdiv(np.flip(gold_bin[i,:]),theta_dir[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]


JSs_ln_dir = [min([sum([JSdiv(theta_ln[t][i,:],theta_dir[t][i,:]) for i in range(L)]),sum([JSdiv(np.flip(theta_ln[t][i,:]),theta_dir[t][i,:]) for i in range(L)])]) for t in range(nchains*500)]


fig,axes=plt.subplots(1, 2, sharey='row')

axes[0].hist(JSs_ln,alpha=.4,color='#d62728',histtype='stepfilled',edgecolor='black')
axes[0].hist(JSs_dir,alpha=.4,color='#1f77b4',histtype='stepfilled',edgecolor='black')
axes[1].hist(JSs_ln_dir,alpha=.4,color='#17becf',histtype='stepfilled',edgecolor='black')


plt.savefig('output/JSD.pdf')
plt.clf()
