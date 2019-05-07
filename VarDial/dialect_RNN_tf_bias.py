#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import tensorflow as tf
import pickle as pkl
import time


f = open('data_and_variables.pkl','rb')
langs_,lens_,inputs_,outputs_,T,N,X,Y,L = pkl.load(f)
f.close()


langs = tf.placeholder(tf.int32,shape=(None,),name='langs')
lens = tf.placeholder(tf.int32,shape=(None,),name='lens')
inputs = tf.placeholder(tf.int32,shape=(None,T),name='inputs')
outputs = tf.placeholder(tf.int32,shape=(None,T),name='outputs')


K = 10
J = 100


def gen_params():
    q_U = tf.get_variable('q_U',[L,K])
    q_W_x = tf.get_variable('q_W_x',[X,K,J])
    q_W_h = tf.get_variable('q_W_h',[K,J,J])
    q_W_y = tf.get_variable('q_W_y',[L,J,Y])
    q_b = tf.nn.softplus(tf.get_variable('q_b',J))
    return(q_U,q_W_x,q_W_h,q_W_y,q_b)

    

##WRONG!!!!
def log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y,q_b):
    mask = tf.expand_dims(tf.sequence_mask(lens,maxlen=T,dtype=tf.float32),2)
    f_U = tf.nn.softmax(q_U)
    #for each batch, for each timepoint, for each component, compute hidden layer activation
    h = []
    for t in range(T):
        if t == 0:
            h.append(tf.nn.softmax(tf.gather(q_W_x,inputs[:,t]) + tf.expand_dims(q_b,-2)))
        else:
            h.append(tf.nn.softmax(tf.einsum('nki,kij->nkj',h[t-1],q_W_h) + tf.gather(q_W_x,inputs[:,t]) + tf.expand_dims(q_b,-2)))
    h = tf.stack(h,axis=1)
    #compute forward pass
    logits = tf.nn.log_softmax(tf.einsum('ntkj,njy->ntky',h,tf.gather(q_W_y,langs)))
    #pi = tf.nn.softmax(logits)
    #exclude masked indices 
    #out = tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*pi,-1)
    out = tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*logits,-1)
    out = tf.exp(tf.reduce_sum(mask*out,1))
    return(tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',tf.gather(f_U,langs),out))))



def log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y,q_b):
    mask = tf.expand_dims(tf.sequence_mask(lens,maxlen=T,dtype=tf.float32),2)
    f_U = tf.nn.softmax(q_U)
    #for each batch, for each timepoint, for each component, compute hidden layer activation
    h = []
    for t in range(T):
        if t == 0:
            h.append(tf.nn.softmax(tf.gather(q_W_x,inputs[:,t]) + tf.expand_dims(q_b,-2)))
        else:
            h.append(tf.nn.softmax(tf.einsum('nki,kij->nkj',h[t-1],q_W_h) + tf.gather(q_W_x,inputs[:,t]) + tf.expand_dims(q_b,-2)))
    h = tf.stack(h,axis=1)
    #compute forward pass
    logits = tf.nn.log_softmax(tf.einsum('ntkj,njy->ntky',h,tf.gather(q_W_y,langs)))
    losses = -tf.reduce_sum(mask*tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*logits,-1),1)
    return(tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',tf.gather(f_U,langs),tf.exp(-losses))+1e-35)))


q_U,q_W_x,q_W_h,q_W_y,q_b = gen_params()


n_batches = int(N/100)
batch_size = int(N/n_batches)


llik = log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y,q_b)*N/batch_size


tf.summary.scalar("llik", llik)


optimizer = tf.train.AdamOptimizer(.1)
train_op = optimizer.minimize(-llik)
check_op = tf.add_check_numerics_ops()
init=tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)



n_epochs = int(10000/n_batches)
MAPs = defaultdict(list)
elbos = defaultdict(list)
idx = np.arange(N)
for c in range(3):
    sess.run(init)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        langs_train,lens_train,inputs_train,outputs_train = langs_[idx],lens_[idx],inputs_[idx],outputs_[idx]
        for t in range(int(N/batch_size)):
            start_time = time.time()
            langs_batch,lens_batch,inputs_batch,outputs_batch = langs_train[t * batch_size:(t + 1) * batch_size],lens_train[t * batch_size:(t + 1) * batch_size],inputs_train[t * batch_size:(t + 1) * batch_size],outputs_train[t * batch_size:(t + 1) * batch_size]
            _, llik_ = sess.run([(train_op,check_op), llik],feed_dict={langs:langs_batch,lens:lens_batch,inputs:inputs_batch,outputs:outputs_batch})
            elbos[c].append(llik_)
            duration = time.time() - start_time
            print("Step: {:>3d} Log ELBO: {:.3f} ({:.3f} sec)".format(epoch*int(N/batch_size)+t, llik_, duration))


