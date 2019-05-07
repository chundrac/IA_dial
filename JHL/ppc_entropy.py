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

assert(nchains==4)

f = open('posterior_ln_full.pkl','rb')
posterior_ln = pkl.load(f)
f.close()


f = open('posterior_dir_full.pkl','rb')
posterior_dir = pkl.load(f)
f.close()



np.random.seed(0)
T = 1000


z_list_dir = []
for t in range(T):
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_dir_hat = np.array([posterior_dir[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_dir_hat = np.array([np.concatenate([posterior_dir[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    Z_dir_unnorm = exp(np.dot(lang_ind,log(theta_dir_hat)) + np.dot(sound_ind,log(phi_dir_hat.T)))
    Z = Z_dir_unnorm/np.sum(Z_dir_unnorm,axis=1)[:,np.newaxis]
    z_list_dir.append(Z)



    
z_list_dir = np.array(z_list_dir)
H_dir = [entropy(l) for l in np.mean(z_list_dir,axis=0)]




z_list_ln = []
for t in range(T):
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_ln_hat = np.array([posterior_ln[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_ln_hat = np.array([np.concatenate([posterior_ln[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    Z_ln_unnorm = exp(np.dot(lang_ind,log(theta_ln_hat)) + np.dot(sound_ind,log(phi_ln_hat.T)))
    Z = Z_ln_unnorm/np.sum(Z_ln_unnorm,axis=1)[:,np.newaxis]
    z_list_ln.append(Z)




z_list_ln = np.array(z_list_ln)
H_ln = [entropy(l) for l in np.mean(z_list_ln,axis=0)]


labels = (z_list_dir,z_list_ln)

f = open('simulated_labels.pkl','wb')
pkl.dump(labels,f)
f.close()


plt.hist(H_dir,bins=25,alpha=.8,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
#tikz_save('output/entropy_dir_1.tex')
plt.savefig('output/entropy_dir_1.pdf')
plt.clf()


plt.hist(H_ln,bins=25,alpha=.8,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
#tikz_save('output/entropy_ln_1.tex')
plt.savefig('output/entropy_ln_1.pdf')
plt.clf()


H_dir_2=[[entropy(z_list_dir[t,i,:]) for i in range(N)] for t in range(T)]
H_dir_2 = np.array(H_dir_2)
plt.hist(np.mean(H_dir_2,axis=0),bins=25,alpha=.8,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
plt.savefig('output/entropy_dir_2.pdf')
#tikz_save('output/entropy_dir_2.tex')
plt.clf()


H_ln_2=[[entropy(z_list_ln[t,i,:]) for i in range(N)] for t in range(T)]
H_ln_2 = np.array(H_dir_2)
plt.hist(np.mean(H_ln_2,axis=0),bins=25,alpha=.8,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
plt.savefig('output/entropy_ln_2.pdf')
#tikz_save('output/entropy_ln_2.tex')
plt.clf()


f = open('output/ppc.tex','w')
print("%<*T>\n"+str(T)+"%</T>\n", file=f)
print("%<*HLn1>\n"+'%.4f' % np.mean(H_ln)+"%</HLn1>\n",file=f)
print("%<*HDir1>\n"+'%.4f' % np.mean(H_dir)+"%</HDir1>\n",file=f)
print("%<*HLn2>\n"+'%.4f' % np.mean(np.mean(H_ln_2,axis=0))+"%</HLn2>\n",file=f)
print("%<*HDir2>\n"+'%.4f' % np.mean(np.mean(H_dir_2,axis=0))+"%</HDir2>\n",file=f)
f.close()