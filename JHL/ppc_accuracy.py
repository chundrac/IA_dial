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


#M = [[sum(sound_ind[i,x[0]:x[1]]) for x in s_breaks] for i in range(N)]


#def gen_data(M,phi_z):
#    s_hat = []
#    for i in range(N):
#        s_hat_i = []
#        for x in range(len(s_breaks)):
#            if M[i][x]!=0:
#                s_hat_i+=list(multinomial(M[i][x],phi_z[i][s_breaks[x][0]:s_breaks[x][1]]))
#            else:
#                s_hat_i+=[0]*R[x]
#        s_hat.append(s_hat_i)
#    return(np.array(s_hat))


M = {(i,x[0],x[1]):sum(sound_ind[i,x[0]:x[1]]) for i in range(N) for x in s_breaks}
M = {k:M[k] for k in M.keys() if M[k] != 0}


def gen_data(M,phi_z):
    s_hat = np.zeros([N,S])
    for i,j,k in M.keys():
        s_hat[i,j:k] += multinomial(M[(i,j,k)],phi_z[i,j:k])
    return(s_hat)



def nonzero(array):
    inds = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(int(array[i,j])):
                inds.append((i,j))
    return(inds)


def nonzero(array):
    x,y = np.nonzero(array)
    counts = array[x,y]
    return([(x[i],y[i]) for i,c in enumerate(counts) for j in range(int(c))])



#generate simulated datasets

full_prior_ln = []
sparse_prior_ln = []
full_posterior_ln = []
z_posterior_ln = []

full_prior_ln_T = []
sparse_prior_ln_T = []
full_posterior_ln_T = []
z_posterior_ln_T = []

for t in range(T):
    print(t)
    #full prior
    alpha = uniform(0,100)
    while alpha == 0:
        alpha = uniform(0,100)
    theta_star = np.array([np.random.dirichlet([alpha,alpha]) for l in range(L)])
    psi_star = np.array([np.random.multivariate_normal([0]*S,Sigma) for k in range(K)])
    phi_star = np.array([np.concatenate([softmax([psi_star[k][s_breaks[x][0]:s_breaks[x][1]]])[0] for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_star)
    s_hat = gen_data(M,phi_z)
    full_prior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_prior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    #sparse prior
    theta_star = np.array([np.random.dirichlet([.1,.1]) for l in range(L)])
    psi_star = np.array([np.random.multivariate_normal([0]*S,Sigma) for k in range(K)])
    phi_star = np.array([np.concatenate([softmax([psi_star[k][s_breaks[x][0]:s_breaks[x][1]]])[0] for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_star)
    s_hat = gen_data(M,phi_z)
    sparse_prior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    sparse_prior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    #pure generative model
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_ln_hat = np.array([posterior_ln[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_ln_hat = np.array([np.concatenate([posterior_ln[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_ln_hat)))
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_ln_hat)
    s_hat = gen_data(M,phi_z)
    full_posterior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_posterior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    #aided generative model
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_ln_hat = np.array([posterior_ln[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_ln_hat = np.array([np.concatenate([posterior_ln[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    Z_ln_unnorm = exp(np.dot(lang_ind,log(theta_ln_hat)) + np.dot(sound_ind,log(phi_ln_hat.T)))
    Z = Z_ln_unnorm/np.sum(Z_ln_unnorm,axis=1)[:,np.newaxis]
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_ln_hat)
    s_hat = gen_data(M,phi_z)
    z_posterior_ln.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    z_posterior_ln_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    


full_prior_dir = []
sparse_prior_dir = []
full_posterior_dir = []
z_posterior_dir = []
full_prior_dir_T = []
sparse_prior_dir_T = []
full_posterior_dir_T = []
z_posterior_dir_T = []


for t in range(T):
    print(t)
    #full prior
    alpha = uniform(0,100)
    while alpha == 0:
        alpha = uniform(0,100)
    theta_star = np.array([np.random.dirichlet([alpha,alpha]) for l in range(L)])
    phi_star = np.array([np.concatenate([np.random.dirichlet([.01]*R[x]) for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_star)
    s_hat = gen_data(M,phi_z)
    full_prior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_prior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    #sparse prior
    theta_star = np.array([np.random.dirichlet([.1,.1]) for l in range(L)])
    phi_star = np.array([np.concatenate([np.random.dirichlet([.01]*R[x]) for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_star)))
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_star)
    s_hat = gen_data(M,phi_z)
    sparse_prior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    sparse_prior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    #pure generative model
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_dir_hat = np.array([posterior_dir[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_dir_hat = np.array([np.concatenate([posterior_dir[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    p_z = np.exp(np.dot(lang_ind,log(theta_dir_hat)))
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_dir_hat)
    s_hat = gen_data(M,phi_z)
    full_posterior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    full_posterior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])
    #aided generative model
    c = np.random.randint(nchains)
    j = np.random.randint(500)
    theta_dir_hat = np.array([posterior_dir[c]['theta_'+str(l)][j,:] for l in range(L)])
    phi_dir_hat = np.array([np.concatenate([posterior_dir[c]['phi_'+str(k)+'_'+str(x)][j,:] for x in range(X)]) for k in range(K)])
    Z_dir_unnorm = exp(np.dot(lang_ind,log(theta_dir_hat)) + np.dot(sound_ind,log(phi_dir_hat.T)))
    Z = Z_dir_unnorm/np.sum(Z_dir_unnorm,axis=1)[:,np.newaxis]
    z = np.array([multinomial(1,p_z[i,:]) for i in range(N)])
    phi_z = np.dot(z,phi_dir_hat)
    s_hat = gen_data(M,phi_z)
    z_posterior_dir.append(1-np.sum(abs(s_hat-sound_ind),axis=1)/(np.sum(sound_ind,axis=1)*2)[:np.newaxis])
    z_posterior_dir_T.append([1-np.sum(abs(s_hat[:, x[0]:x[1]]-sound_ind[:, x[0]:x[1]]))/(np.sum(sound_ind[:,x[0]:x[1]])*2) for x in s_breaks])




full_prior_ln = np.array(full_prior_ln)
sparse_prior_ln = np.array(sparse_prior_ln)
full_posterior_ln = np.array(full_posterior_ln)
z_posterior_ln = np.array(z_posterior_ln)



full_prior_ln_T = np.array(full_prior_ln_T)
sparse_prior_ln_T = np.array(sparse_prior_ln_T)
full_posterior_ln_T = np.array(full_posterior_ln_T)
z_posterior_ln_T = np.array(z_posterior_ln_T)



full_prior_dir = np.array(full_prior_dir)
sparse_prior_dir = np.array(sparse_prior_dir)
full_posterior_dir = np.array(full_posterior_dir)
z_posterior_dir = np.array(z_posterior_dir)



full_prior_dir_T = np.array(full_prior_dir_T)
sparse_prior_dir_T = np.array(sparse_prior_dir_T)
full_posterior_dir_T = np.array(full_posterior_dir_T)
z_posterior_dir_T = np.array(z_posterior_dir_T)



accuracies = (full_prior_ln,
              sparse_prior_ln,
              full_posterior_ln,
              z_posterior_ln,
              full_prior_ln_T,
              sparse_prior_ln_T,
              full_posterior_ln_T,
              z_posterior_ln_T,
              full_prior_dir,
              sparse_prior_dir,
              full_posterior_dir,
              z_posterior_dir,
              full_prior_dir_T,
              sparse_prior_dir_T,
              full_posterior_dir_T,
              z_posterior_dir_T)


f = open('ppc_accuracies.pkl','wb')
pkl.dump(accuracies,f)
f.close()


