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



f = open('ppc_accuracies.pkl','rb')
full_prior_ln,sparse_prior_ln,full_posterior_ln,z_posterior_ln,full_prior_ln_T,sparse_prior_ln_T,full_posterior_ln_T,z_posterior_ln_T,full_prior_dir,sparse_prior_dir,full_posterior_dir,z_posterior_dir,full_prior_dir_T,sparse_prior_dir_T,full_posterior_dir_T,z_posterior_dir_T=pkl.load(f)
f.close()


plt.hist([np.mean(full_prior_ln,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_ln,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_ln,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_ln,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_ln_word.tex')
plt.clf()


plt.hist([np.mean(full_prior_ln_T,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_ln_T,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_ln_T,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_ln_T,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_ln_sound.tex')
plt.clf()


plt.hist([np.mean(full_prior_dir,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_dir,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_dir,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_dir,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_dir_word.tex')
plt.clf()


plt.hist([np.mean(full_prior_dir_T,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_dir_T,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_dir_T,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_dir_T,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_dir_sound.tex')
plt.clf()


#panel plots

fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
axes[1,0].hist([np.mean(full_prior_ln,axis=1)],alpha=.4,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
axes[1,0].hist([np.mean(sparse_prior_ln,axis=1)],alpha=.4,color='#d62728',histtype='stepfilled',edgecolor='black')
axes[0,0].hist([np.mean(full_prior_dir,axis=1)],alpha=.4,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
axes[0,0].hist([np.mean(sparse_prior_dir,axis=1)],alpha=.4,color='#d62728',histtype='stepfilled',edgecolor='black')
axes[1,1].hist([np.mean(full_posterior_ln,axis=1)],alpha=.4,color='#1f77b4',histtype='stepfilled',edgecolor='black')
axes[1,1].hist([np.mean(z_posterior_ln,axis=1)],alpha=.4,color='#17becf',histtype='stepfilled',edgecolor='black')
axes[0,1].hist([np.mean(full_posterior_dir,axis=1)],alpha=.4,color='#1f77b4',histtype='stepfilled',edgecolor='black')
axes[0,1].hist([np.mean(z_posterior_dir,axis=1)],alpha=.4,color='#17becf',histtype='stepfilled',edgecolor='black')


plt.savefig('output/accuracy_word.pdf')
plt.clf()

fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
axes[1,0].hist([np.mean(full_prior_ln_T,axis=1)],alpha=.4,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
axes[1,0].hist([np.mean(sparse_prior_ln_T,axis=1)],alpha=.4,color='#d62728',histtype='stepfilled',edgecolor='black')
axes[0,0].hist([np.mean(full_prior_dir_T,axis=1)],alpha=.4,color='#ff7f0e',histtype='stepfilled',edgecolor='black')
axes[0,0].hist([np.mean(sparse_prior_dir_T,axis=1)],alpha=.4,color='#d62728',histtype='stepfilled',edgecolor='black')
axes[1,1].hist([np.mean(full_posterior_ln_T,axis=1)],alpha=.4,color='#1f77b4',histtype='stepfilled',edgecolor='black')
axes[1,1].hist([np.mean(z_posterior_ln_T,axis=1)],alpha=.4,color='#17becf',histtype='stepfilled',edgecolor='black')
axes[0,1].hist([np.mean(full_posterior_dir_T,axis=1)],alpha=.4,color='#1f77b4',histtype='stepfilled',edgecolor='black')
axes[0,1].hist([np.mean(z_posterior_dir_T,axis=1)],alpha=.4,color='#17becf',histtype='stepfilled',edgecolor='black')


plt.savefig('output/accuracy_sound.pdf')
plt.clf()


f = open('output/ppc.tex','a')
print("%<*FullPriorDir>\n"+'%.4f' % np.mean(np.mean(full_prior_dir,axis=1))+"%</FullPriorDir>\n",file=f)
print("%<*SparsePriorDir>\n"+'%.4f' % np.mean(np.mean(sparse_prior_dir,axis=1))+"%</SparsePriorDir>\n",file=f)
print("%<*FullPosteriorDir>\n"+'%.4f' % np.mean(np.mean(full_posterior_dir,axis=1))+"%</FullPosteriorDir>\n",file=f)
print("%<*ZPosteriorDir>\n"+'%.4f' % np.mean(np.mean(z_posterior_dir,axis=1))+"%</ZPosteriorDir>\n",file=f)
print("%<*FullPriorLn>\n"+'%.4f' % np.mean(np.mean(full_prior_ln,axis=1))+"%</FullPriorLn>\n",file=f)
print("%<*SparsePriorLn>\n"+'%.4f' % np.mean(np.mean(sparse_prior_ln,axis=1))+"%</SparsePriorLn>\n",file=f)
print("%<*FullPosteriorLn>\n"+'%.4f' % np.mean(np.mean(full_posterior_ln,axis=1))+"%</FullPosteriorLn>\n",file=f)
print("%<*ZPosteriorLn>\n"+'%.4f' % np.mean(np.mean(z_posterior_ln,axis=1))+"%</ZPosteriorLn>\n",file=f)
print("%<*MaxPosteriorDir>\n"+'%.4f' % max(list(np.mean(full_posterior_dir,axis=1))+list(np.mean(z_posterior_dir,axis=1)))+"%</MaxPosteriorDir>\n",file=f)
print("%<*MaxPosteriorLn>\n"+'%.4f' % max(list(np.mean(full_posterior_ln,axis=1))+list(np.mean(z_posterior_ln,axis=1)))+"%</MaxPosteriorLn>\n",file=f)


f.close()



f = open('output/sound_change_accuracies.tex','w')

#for i in range(X):
#    k = list(reflex.keys())[i]
#    print('{\\IPA',k[1],'/',k[0],'\\underline{\\phantom{X}}',k[2],'}','&',np.mean(full_posterior_dir[:,i]),np.mean(full_posterior_ln[:,i]),file=f)


for k in sorted(reflex.keys(),key=lambda x:x[1]):
    i = list(reflex.keys()).index(k)
    print('{\\IPA',k[1],'/',k[0].replace('#','$\#$'),'\\underline{\\phantom{X}}',k[2].replace('#','$\#$'),'}','&','%.4f' % np.mean(full_posterior_dir_T[:,i]),'&','%.4f' % np.mean(full_posterior_ln_T[:,i]),'\\\\',file=f)


f.close()



f = open('output/sound_change_mean_accuracies.tex','w')

#for i in range(X):
#    k = list(reflex.keys())[i]
#    print('{\\IPA',k[1],'/',k[0],'\\underline{\\phantom{X}}',k[2],'}','&',np.mean(full_posterior_dir[:,i]),np.mean(full_posterior_ln[:,i]),file=f)

avg_acc_dir = defaultdict(list)
avg_acc_ln = defaultdict(list)

for k in reflex.keys():
    i = list(reflex.keys()).index(k)
    avg_acc_dir[k[1]].append(np.mean(full_posterior_dir_T[:,i]))
    avg_acc_ln[k[1]].append(np.mean(full_posterior_ln_T[:,i]))



for k in segs_to_keep:
    print('{\\IPA',k,'}','&','%.4f' % mean(avg_acc_dir[k]),'&','%.4f' % mean(avg_acc_ln[k]),'\\\\',file=f)


f.close()


langdict_dir=defaultdict(list)
for i in range(N):
    langdict_dir[changes_pruned[i][0]].append(np.mean(z_posterior_dir[:,i],axis=0))




langdict_ln=defaultdict(list)
for i in range(N):
    langdict_ln[changes_pruned[i][0]].append(np.mean(z_posterior_ln[:,i],axis=0))




f = open('output/lang_per_word_accuracy.tex','w')


for k in langs:
    print(k,'&','%.4f' % np.mean(langdict_dir[k]),'&','%.4f' % np.mean(langdict_ln[k]),'\\\\',file=f)



f.close()