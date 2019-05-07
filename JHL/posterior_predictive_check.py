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
from sklearn.metrics import f1_score



f = open('data_variables.pkl','rb')
Cutoff,change_list,reflex,Sigma,K,S,X,R,N,langs,L,lang_ind,sound_ind,s_breaks,nchains = pkl.load(f)
f.close()

assert(nchains==4)

f = open('posterior_ln_full.pkl','rb')
posterior_ln = pkl.load(f)
f.close()


f = open('posterior_dir_full.pkl','rb')
posterior_dir = pkl.load(f)
f.close()



np.random.seed(0)
T = 100


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


plt.hist(H_dir,bins=25,alpha=.8,color='#1f77b4')
tikz_save('output/entropy_dir_1.tex')
plt.clf()


plt.hist(H_ln,bins=25,alpha=.8,color='#1f77b4')
tikz_save('output/entropy_ln_1.tex')
plt.clf()


H_dir_2=[[entropy(z_list_dir[t,i,:]) for i in range(N)] for t in range(T)]
H_dir_2 = np.array(H_dir_2)
plt.hist(np.mean(H_dir_2,axis=0),bins=25,alpha=.8,color='#1f77b4')
tikz_save('output/entropy_dir_2.tex')
plt.clf()


H_ln_2=[[entropy(z_list_ln[t,i,:]) for i in range(N)] for t in range(T)]
H_ln_2 = np.array(H_dir_2)
plt.hist(np.mean(H_ln_2,axis=0),bins=25,alpha=.8,color='#1f77b4')
tikz_save('output/entropy_ln_2.tex')
plt.clf()


"""
word_entropy = {}
for i in range(N):
  word_entropy[i]=H_dir[i]



for k in sorted(word_entropy.keys(),key=lambda x:word_entropy[x]):
  print(ref_changes_pruned[k],changes_pruned[k][1],word_entropy[k])
"""




M = [[sum(sound_ind[i,x[0]:x[1]]) for x in s_breaks] for i in range(len(sound_ind))]




full_prior_ln = []
full_prior_ln_T = []
for t in range(T):
    alpha = uniform(0,100)
    while alpha == 0:
        alpha = uniform(0,100)
    theta_star = np.array([np.random.dirichlet([alpha,alpha]) for l in range(L)])
    psi_star = np.array([np.random.multivariate_normal([0]*S,Sigma*10) for k in range(K)])
    phi_star = np.array([np.concatenate([(softmax([psi_star[k][s_breaks[x][0]:s_breaks[x][1]]])[0]**10)/np.sum(softmax([psi_star[k][s_breaks[x][0]:s_breaks[x][1]]])[0]**10) for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = [list(multinomial(1,p_z[i,:])).index(1) for i in range(N)]
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_star[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    full_prior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_prior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])





#sparse prior model
sparse_prior_ln = []    
sparse_prior_ln_T = []
for t in range(T):
    theta_star = np.array([np.random.dirichlet([.1,.1]) for l in range(L)])
    psi_star = np.array([np.random.multivariate_normal([0]*S,Sigma*10) for k in range(K)])
    phi_star = np.array([np.concatenate([(softmax([psi_star[k][s_breaks[x][0]:s_breaks[x][1]]])[0]**10)/np.sum(softmax([psi_star[k][s_breaks[x][0]:s_breaks[x][1]]])[0]**10) for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = [list(multinomial(1,p_z[i,:])).index(1) for i in range(N)]
#    for i in range(N):
#        s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_ln_hat[z[i]][x[0]:x[1]]) for k,x in enumerate(s_breaks)])])
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_star[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    sparse_prior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    sparse_prior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])




#Pure generative model
full_posterior_ln = []
full_posterior_ln_T = []
for t in range(T):
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_ln_hat = np.array([posterior_ln[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_ln_hat = np.array([np.concatenate([posterior_ln[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_ln_hat)))
    z = [list(multinomial(1,p_z[i,:])).index(1) for i in range(N)]
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_ln_hat[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    full_posterior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_posterior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])






# aided generative model
z_posterior_ln = []
z_posterior_ln_T = []
for t in range(T):
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_ln_hat = np.array([posterior_ln[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_ln_hat = np.array([np.concatenate([posterior_ln[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    Z_ln_unnorm = exp(np.dot(lang_ind,log(theta_ln_hat)) + np.dot(sound_ind,log(phi_ln_hat.T)))
    Z = Z_ln_unnorm/np.sum(Z_ln_unnorm,axis=1)[:,np.newaxis]
    z = [list(multinomial(1,Z[i])).index(1) for i in range(N)]
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_ln_hat[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    z_posterior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    z_posterior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])





full_prior_ln = np.array(full_prior_ln)
sparse_prior_ln = np.array(sparse_prior_ln)
full_posterior_ln = np.array(full_posterior_ln)
z_posterior_ln = np.array(z_posterior_ln)




plt.hist([np.mean(full_prior_ln,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_ln,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_ln,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_ln,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_ln_word.tex')
plt.clf()




full_prior_ln_T = np.array(full_prior_ln_T)
sparse_prior_ln_T = np.array(sparse_prior_ln_T)
full_posterior_ln_T = np.array(full_posterior_ln_T)
z_posterior_ln_T = np.array(z_posterior_ln_T)




plt.hist([np.mean(full_prior_ln_T,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_ln_T,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_ln_T,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_ln_T,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_ln_sound.tex')
plt.clf()



    
full_prior_dir = []
full_prior_dir_T = []
for t in range(100):
    alpha = uniform(0,100)
    while alpha == 0:
        alpha = uniform(0,100)
    theta_star = np.array([np.random.dirichlet([alpha,alpha]) for l in range(L)])
    phi_star = np.array([np.concatenate([np.random.dirichlet([.01]*R[x]) for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = [list(multinomial(1,p_z[i,:])).index(1) for i in range(N)]
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_star[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    full_prior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_prior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])




    

#sparse prior model
sparse_prior_dir = []
sparse_prior_dir_T = []
for t in range(100):
    theta_star = np.array([np.random.dirichlet([.1,.1]) for l in range(L)])
    phi_star = np.array([np.concatenate([np.random.dirichlet([.01]*R[x]) for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = [list(multinomial(1,p_z[i,:])).index(1) for i in range(N)]
#    for i in range(N):
#        s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_ln_hat[z[i]][x[0]:x[1]]) for k,x in enumerate(s_breaks)])])
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_star[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    sparse_prior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    sparse_prior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])



    

full_posterior_dir = []
full_posterior_dir_T = []
for t in range(100):
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_dir_hat = np.array([posterior_dir[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_dir_hat = np.array([np.concatenate([posterior_dir[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_dir_hat)))
    z = [list(multinomial(1,p_z[i,:])).index(1) for i in range(N)]
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_dir_hat[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    full_posterior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_posterior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])





    
    
# aided generative model
z_posterior_dir = []
z_posterior_dir_T = []
for t in range(100):
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_dir_hat = np.array([posterior_dir[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_dir_hat = np.array([np.concatenate([posterior_dir[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    Z_dir_unnorm = exp(np.dot(lang_ind,log(theta_dir_hat)) + np.dot(sound_ind,log(phi_dir_hat.T)))
    Z = Z_dir_unnorm/np.sum(Z_dir_unnorm,axis=1)[:,np.newaxis]
    z = [list(multinomial(1,Z[i])).index(1) for i in range(N)]
    s_hat = np.array([np.concatenate([multinomial(M[i][k],phi_dir_hat[z[i]][s_breaks[k][0]:s_breaks[k][1]]) for k in range(len(s_breaks))]) for i in range(N)])
    z_posterior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    z_posterior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])







full_prior_dir = np.array(full_prior_dir)
sparse_prior_dir = np.array(sparse_prior_dir)
full_posterior_dir = np.array(full_posterior_dir)
z_posterior_dir = np.array(z_posterior_dir)



plt.hist([np.mean(full_prior_dir,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_dir,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_dir,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_dir,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_dir_word.tex')
plt.clf()



full_prior_dir_T = np.array(full_prior_dir_T)
sparse_prior_dir_T = np.array(sparse_prior_dir_T)
full_posterior_dir_T = np.array(full_posterior_dir_T)
z_posterior_dir_T = np.array(z_posterior_dir_T)



plt.hist([np.mean(full_prior_dir_T,axis=1)],alpha=.4,color='#ff7f0e')
plt.hist([np.mean(sparse_prior_dir_T,axis=1)],alpha=.4,color='#d62728')
plt.hist([np.mean(full_posterior_dir_T,axis=1)],alpha=.4,color='#1f77b4')
plt.hist([np.mean(z_posterior_dir_T,axis=1)],alpha=.4,color='#17becf')
tikz_save('output/accuracy_dir_sound.tex')
plt.clf()



f = open('output/ppc.tex','w')


print("%<*T>\n"+str(T)+"%</T>\n", file=f)
print("%<*HLn1>\n"+'%.2f' % np.mean(H_ln)+"%</HLn1>\n",file=f)
print("%<*HDir1>\n"+'%.2f' % np.mean(H_dir)+"%</HDir1>\n",file=f)
print("%<*HLn2>\n"+'%.2f' % np.mean(np.mean(H_ln_2,axis=0))+"%</HLn2>\n",file=f)
print("%<*HDir2>\n"+'%.2f' % np.mean(np.mean(H_dir_2,axis=0))+"%</HDir2>\n",file=f)
print("%<*FullPriorDir>\n"+'%.2f' % np.mean(np.mean(full_prior_dir,axis=1))+"%</FullPriorDir>\n",file=f)
print("%<*SparsePriorDir>\n"+'%.2f' % np.mean(np.mean(sparse_prior_dir,axis=1))+"%</SparsePriorDir>\n",file=f)
print("%<*FullPosteriorDir>\n"+'%.2f' % np.mean(np.mean(full_posterior_dir,axis=1))+"%</FullPosteriorDir>\n",file=f)
print("%<*ZPosteriorDir>\n"+'%.2f' % np.mean(np.mean(z_posterior_dir,axis=1))+"%</ZPosteriorDir>\n",file=f)
print("%<*FullPriorLn>\n"+'%.2f' % np.mean(np.mean(full_prior_ln,axis=1))+"%</FullPriorLn>\n",file=f)
print("%<*SparsePriorLn>\n"+'%.2f' % np.mean(np.mean(sparse_prior_ln,axis=1))+"%</SparsePriorLn>\n",file=f)
print("%<*FullPosteriorLn>\n"+'%.2f' % np.mean(np.mean(full_posterior_ln,axis=1))+"%</FullPosteriorLn>\n",file=f)
print("%<*ZPosteriorLn>\n"+'%.2f' % np.mean(np.mean(z_posterior_ln,axis=1))+"%</ZPosteriorLn>\n",file=f)
print("%<*MaxPosteriorDir>\n"+'%.2f' % max(list(np.mean(full_posterior_dir,axis=1))+list(np.mean(z_posterior_dir,axis=1)))+"%</MaxPosteriorDir>\n",file=f)
print("%<*MaxPosteriorLn>\n"+'%.2f' % max(list(np.mean(full_posterior_ln,axis=1))+list(np.mean(z_posterior_ln,axis=1)))+"%</MaxPosteriorLn>\n",file=f)


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
