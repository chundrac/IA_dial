#!usr/bin/env python3.4.3

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


def sigmoid(x):
    return(1/(1+np.exp(-x)))


def softmax3d(x):
    x = x-x.max(axis=1,keepdims=True)
    return(np.exp(x)/np.sum(np.exp(x),axis=0))


def GEM(beta):  #stick-breaking construction
    pi = np.concatenate([np.ones([beta.shape[0],1]), np.cumprod(1 - beta,axis=1)],axis=1)
    return(np.concatenate([beta,np.ones([beta.shape[0],1])],axis=1) * pi)
    
    
def GEM(beta):  #stick-breaking construction
    pi = np.concatenate([[1.], np.cumprod(1 - beta)],axis=0)
    return(np.concatenate([beta,[1.]],axis=0) * pi)



def laplace_Dir(alpha,shape):
    mu = np.log(alpha) - np.mean(np.log(alpha))
    sd = (1/alpha)*(1-(2/shape))+((1/shape**2)*np.sum(1/alpha))
    return(mu,sd)


def SEK(delta,alpha,inv_rhos,sigma):
    return(np.linalg.cholesky(np.power(alpha,2) * np.exp(-.5*np.sum(delta*tf.pow(inv_rhos,2))) + np.eye(S)*np.power(sigma,2)))


def gen_H_diag(post):
    curr = {k:np.array([np.random.normal(loc=post[c][k][0],scale=post[c][k][1]) for c in range(len(post)) for i in range(100)]) for k in post[0].keys()}
    zeta = GEM(sigmoid(curr['W']*curr['tau']))
    mu_zeta,sd_zeta = laplace_Dir(zeta,K)
    eta = np.repeat(mu_zeta.reshape(mu_zeta.shape[0],1,mu_zeta.shape[1]),50,1) + curr['eta']*sd_zeta.reshape(sd_zeta.shape[0],1,sd_zeta.shape[1])
    theta = np.array([softmax(eta[i,:,:]) for i in range(curr['psi'].shape[0])])
    print('yay')
    phi = np.array([np.concatenate([softmax(curr['psi'][i,:,partition[x][0]:partition[x][1]]) for x in range(X)],axis=1) for i in range(curr['psi'].shape[0])])
    Z = np.exp(np.einsum('nl,slk->snk',lang_ind,np.log(theta))+np.einsum('nd,sdk->snk',sound_ind,np.log(np.transpose(phi,[0,2,1]))))
    
    

def gen_H_diag(post):
    Hs = []
    N = 100
    curr = {k:[np.random.normal(loc=post[c][k][0],scale=post[c][k][1]) for c in range(len(post)) for i in range(N)] for k in post[0].keys()}
    N = N*len(post)
    for n in range(N):
        print(n)
        zeta = GEM(sigmoid(curr['W'][n]*curr['tau'][n]))
        mu_zeta,sd_zeta = laplace_Dir(zeta,K)
        eta = mu_zeta + curr['eta'][n]*sd_zeta
        theta = softmax(eta)
        phi = np.concatenate([softmax(curr['psi'][n][:,partition[x][0]:partition[x][1]]) for x in range(X)],axis=1)
        Z = np.exp(np.dot(lang_ind,np.log(theta+1e-20)) + np.dot(sound_ind,np.log(phi.T+1e-20)))
        H = np.mean(entropy(Z.T))
        Hs.append(H)
    return(Hs)


H_diag = gen_H_diag(post_diag)



    






gen_H_diag(post_diag)





def gen_H_bin(post):
    curr = {k:np.array([np.random.normal(loc=post[c][k][0],scale=post[c][k][1]) for c in range(len(post)) for i in range(1)]) for k in post[0].keys()}
    zeta = GEM(sigmoid(curr['W']*curr['tau']))
    L = SEK(delta,curr['alpha'],curr['inv_rhos'],curr['sigma'])
    psi = (np.dot(L,curr['z'])).T
    mu_zeta,sd_zeta = laplace_Dir(zeta,K)
    eta = np.repeat(mu_zeta.reshape(mu_zeta.shape[0],1,mu_zeta.shape[1]),50,1) + curr['eta']*sd_zeta.reshape(sd_zeta.shape[0],1,sd_zeta.shape[1])
    theta = np.array([softmax(eta[i,:,:]) for i in range(curr['psi'].shape[0])])
    #theta = softmax3d(np.repeat(mu_zeta.reshape(mu_zeta.shape[0],1,mu_zeta.shape[1]),50,1) + curr['eta']*sd_zeta.reshape(sd_zeta.shape[0],1,sd_zeta.shape[1]))
    phi = np.array([np.concatenate([softmax(curr['psi'][i,:,partition[x][0]:partition[x][1]]) for x in range(X)],axis=1) for i in range(curr['psi'].shape[0])])
    Z = np.exp(np.einsum('nl,slk->sk',lang_ind,np.log(theta))+np.einsum('nd,dks->sk',sound_ind,np.log(phi.T)))
    
    
    
    


def gen_H_grad(post):
    curr = {k:np.array([np.random.normal(loc=post[c][k][0],scale=post[c][k][1]) for c in range(len(post)) for i in range(100)]) for k in post[0].keys()}
    zeta = GEM(tf.nn.sigmoid(curr['W']*curr['tau']))
    L = SEK(delta,curr['alpha'],curr['inv_rhos'],curr['sigma'])
    psi = tf.transpose(tf.matmul(L,curr['z']))
    mu_zeta,sd_zeta = laplace_Dir(zeta,K)
    eta = np.repeat(mu_zeta.reshape(mu_zeta.shape[0],1,mu_zeta.shape[1]),50,1) + curr['eta']*sd_zeta.reshape(sd_zeta.shape[0],1,sd_zeta.shape[1])
    theta = np.array([softmax(eta[i,:,:]) for i in range(curr['psi'].shape[0])])
    #theta = softmax3d(np.repeat(mu_zeta.reshape(mu_zeta.shape[0],1,mu_zeta.shape[1]),50,1) + curr['eta']*sd_zeta.reshape(sd_zeta.shape[0],1,sd_zeta.shape[1]))
    phi = np.array([np.concatenate([softmax(curr['psi'][i,:,partition[x][0]:partition[x][1]]) for x in range(X)],axis=1) for i in range(curr['psi'].shape[0])])
    Z = np.exp(np.einsum('nl,slk->sk',lang_ind,np.log(theta))+np.einsum('nd,sdk->sk',sound_ind,np.log(phi.T)))
    