#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import time
tfd = tfp.distributions


#load the data and variables
f = open('data_and_variables.pkl','rb')
langs_,lens_,inputs_,outputs_,T,N,X,Y,L = pkl.load(f)
f.close()


#create placeholder objects for minibatches to be passed to
langs = tf.placeholder(tf.int32,shape=(None,),name='langs')
lens = tf.placeholder(tf.int32,shape=(None,),name='lens')
inputs = tf.placeholder(tf.int32,shape=(None,T),name='inputs')
outputs = tf.placeholder(tf.int32,shape=(None,T),name='outputs')


#set number of components
K = 10


#create the variable objects
def gen_params():
    q_U = tf.get_variable('q_U',[L,K])
    q_W = tf.get_variable('q_W_x',[X,K,Y])
    return(q_U,q_W)


#compute prior density of parameter values
def log_prior(q_U,q_W):
    U = tfd.Normal(tf.ones([L,K]),1.)
    W = tfd.Normal(tf.ones([X,K,Y]),10.)
    lprior = tf.reduce_sum(U.log_prob(q_U)) + tf.reduce_sum(W.log_prob(q_W))
    return(lprior)    


#compute likelihood of parameter values
def log_lik(langs,lens,inputs,outputs,q_U,q_W):
    mask = tf.expand_dims(tf.sequence_mask(lens,maxlen=T,dtype=tf.float32),2)
    f_U = tf.nn.softmax(q_U)
    logits = tf.nn.log_softmax(tf.gather(q_W,inputs))
    losses = -tf.reduce_sum(mask*tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*logits,-1),1)
    return(tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',tf.gather(f_U,langs),tf.exp(-losses))+1e-35)))



q_U,q_W = gen_params()



batch_size = 100
n_batches = int(N/batch_size)



llik = log_lik(langs,lens,inputs,outputs,q_U,q_W)*N/batch_size
lprior = log_prior(q_U,q_W)
lpost = lprior + llik
tf.summary.scalar("lpost", lpost)


lr = .1  #learning rate
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(-lpost)
check_op = tf.add_check_numerics_ops()
init=tf.global_variables_initializer()



sess = tf.Session()



n_epochs = int(10000/n_batches)
idx = np.arange(N)
for c in range(4):
    lps = []
    sess.run(init)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        langs_train,lens_train,inputs_train,outputs_train = langs_[idx],lens_[idx],inputs_[idx],outputs_[idx]
        for t in range(n_batches):
            start_time = time.time()
            langs_batch,lens_batch,inputs_batch,outputs_batch = langs_train[t * batch_size:(t + 1) * batch_size],lens_train[t * batch_size:(t + 1) * batch_size],inputs_train[t * batch_size:(t + 1) * batch_size],outputs_train[t * batch_size:(t + 1) * batch_size]
            _, lpost_ = sess.run([(train_op,check_op), lpost],feed_dict={langs:langs_batch,lens:lens_batch,inputs:inputs_batch,outputs:outputs_batch})
            lps.append(lpost_)
            duration = time.time() - start_time
            print("Step: {:>3d} Log Posterior: {:.3f} ({:.3f} sec)".format(epoch*int(n_batches)+t, lpost_, duration))
    MAP = (sess.run(q_U),sess.run(q_W))
    f = open('posterior_discrete_flat_sd1_{}_{}.pkl'.format(lr,c),'wb')
    pkl.dump((lps,MAP),f)
    f.close()