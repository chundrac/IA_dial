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



f_V_H = []
for c in range(3):
    H_s = []
    V = posterior_flat[c][1][1]
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            H = entropy(softmax(V[i,j:j+1,:])[0])/entropy(np.ones(V.shape[2])/V.shape[2])
            H_s.append(H)
    f_V_H.append(H_s)



f_Wx_H = []
for c in range(3):
    H_s = []
    V = posterior_RNN[c][1][1]
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            H = entropy(softmax(V[i,j:j+1,:])[0])/entropy(np.ones(V.shape[2])/V.shape[2])
            H_s.append(H)
    f_Wx_H.append(H_s)



#forward pass entropy

def log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y):
    mask = tf.expand_dims(tf.sequence_mask(lens,maxlen=T,dtype=tf.float32),2)
    f_U = tf.nn.softmax(q_U)
    #for each batch, for each timepoint, for each component, compute hidden layer activation
    h = []
    for t in range(T):
        if t == 0:
            h.append(tf.nn.softmax(tf.gather(q_W_x,inputs[:,t])))
        else:
            h.append(tf.nn.softmax(tf.einsum('nki,kij->nkj',h[t-1],q_W_h) + tf.gather(q_W_x,inputs[:,t])))
    h = tf.stack(h,axis=1)
    #compute forward pass, and marginalize over all k \in K
    logits = tf.nn.log_softmax(tf.einsum('ntkj,njy->ntky',h,tf.gather(q_W_y,langs)))
    losses = -tf.reduce_sum(mask*tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*logits,-1),1)
    return(tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',tf.gather(f_U,langs),tf.exp(-losses))+1e-35)))



entropies = []
for c in range(3):
  entropies_c = []
  q_W_x = posterior_RNN[c][1][1]
  q_W_h = posterior_RNN[c][1][2]
  q_W_y = posterior_RNN[c][1][3]
  for k in range(10):
    for i in range(N):
        print(i)
        h = []
        for t in range(lens_[i]):
            if t == 0:
                h.append(softmax(q_W_x[inputs_[i,t]]))
            else:
                h.append(softmax(np.einsum('ki,ij->kj',h[t-1],q_W_h[k,:,:]) + q_W_x[inputs_[i,t]]))
        h = np.stack(h)
        output_pre = np.einsum('tkj,jy->tky',h,q_W_y[langs_[i],:,:])
        for a in range(output_pre.shape[0]):
            for b in range(output_pre.shape[1]):
                entropies_c.append(entropy(softmax(output_pre[a,b:b+1,:])[0])/entropy(np.ones(output_pre.shape[2])/output_pre.shape[2]))
  entropies.append(entropies_c)