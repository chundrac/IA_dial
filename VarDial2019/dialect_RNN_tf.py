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


f = open('data_and_variables.pkl','rb')
langs_,lens_,inputs_,outputs_,T,N,X,Y,L,lang_list = pkl.load(f)
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
    return(q_U,q_W_x,q_W_h,q_W_y)


def log_prior(q_U,q_W_x,q_W_h,q_W_y):
    U = tfd.Normal(tf.ones([L,K]),10.)
    W_x = tfd.Normal(tf.ones([X,K,J]),10.)
    W_h = tfd.Normal(tf.ones([K,J,J]),10.)
    W_y = tfd.Normal(tf.ones([L,J,Y]),10.)
    lprior = tf.reduce_sum(U.log_prob(q_U)) + tf.reduce_sum(W_x.log_prob(q_W_x)) + tf.reduce_sum(W_h.log_prob(q_W_h)) + tf.reduce_sum(W_y.log_prob(q_W_y))
    return(lprior)    


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


q_U,q_W_x,q_W_h,q_W_y = gen_params()


batch_size = 100
n_batches = int(N/batch_size)


llik = log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y)*N/batch_size
lprior = log_prior(q_U,q_W_x,q_W_h,q_W_y)

lpost = lprior + llik


tf.summary.scalar("lpost", lpost)


lr = .1                                 #learning rate
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(-lpost)
check_op = tf.add_check_numerics_ops()
init=tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)



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
            print("Step: {:>3d} Log ELBO: {:.3f} ({:.3f} sec)".format(epoch*int(n_batches)+t, lpost_, duration))
    MAP = (sess.run(q_U),sess.run(q_W_x),sess.run(q_W_h),sess.run(q_W_y))
    f = open('posterior_discrete_RNN_{}.pkl'.format(c),'wb')
    pkl.dump((lps,MAP),f)
    f.close()


