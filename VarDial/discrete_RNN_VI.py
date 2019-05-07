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
langs_,lens_,inputs_,outputs_,T,N,X,Y,L = pkl.load(f)
f.close()


langs = tf.placeholder(tf.int32,shape=(None,),name='langs')
lens = tf.placeholder(tf.int32,shape=(None,),name='lens')
inputs = tf.placeholder(tf.int32,shape=(None,T),name='inputs')
outputs = tf.placeholder(tf.int32,shape=(None,T),name='outputs')


K = 10
J = 100


#variational parameters
def gen_var_dist():
    loc_q_U,scale_q_U = tf.get_variable('loc_q_U',[L,K]),tf.nn.softplus(tf.get_variable('scale_q_U',[L,K]))+1e-5
    loc_q_W_x,scale_q_W_x = tf.get_variable('loc_q_W_x',[X,K,J]),tf.nn.softplus(tf.get_variable('scale_q_W_x',[X,K,J]))+1e-5
    loc_q_W_h,scale_q_W_h = tf.get_variable('loc_q_W_h',[K,J,J]),tf.nn.softplus(tf.get_variable('scale_q_W_h',[K,J,J]))+1e-5
    loc_q_W_y,scale_q_W_y = tf.get_variable('loc_q_W_y',[L,J,Y]),tf.nn.softplus(tf.get_variable('scale_q_W_y',[L,J,Y]))+1e-5
    q_U = tfd.Normal(loc_q_U,scale_q_U)
    q_W_x = tfd.Normal(loc_q_W_x,scale_q_W_x)
    q_W_h = tfd.Normal(loc_q_W_h,scale_q_W_h)
    q_W_y = tfd.Normal(loc_q_W_y,scale_q_W_y)
    return(q_U,q_W_x,q_W_h,q_W_y)


#prior distribution
def gen_prior():
    U = tfd.Normal(tf.ones([L,K]),10.)
    W_x = tfd.Normal(tf.ones([X,K,J]),10.)
    W_h = tfd.Normal(tf.ones([K,J,J]),10.)
    W_y = tfd.Normal(tf.ones([L,J,Y]),10.)
    return(U,W_x,W_h,W_y)    


#log likelihood under sample from variational distribution
def log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y):
    q_U_ = q_U.sample()
    q_W_x_ = q_W_x.sample()
    q_W_h_ = q_W_h.sample()
    q_W_y_ = q_W_y.sample()
    mask = tf.expand_dims(tf.sequence_mask(lens,maxlen=T,dtype=tf.float32),2)
    f_U = tf.nn.softmax(q_U_)
    #for each batch, for each timepoint, for each component, compute hidden layer activation
    h = []
    for t in range(T):
        if t == 0:
            h.append(tf.nn.softmax(tf.gather(q_W_x_,inputs[:,t])))
        else:
            h.append(tf.nn.softmax(tf.einsum('nki,kij->nkj',h[t-1],q_W_h_) + tf.gather(q_W_x_,inputs[:,t])))
    h = tf.stack(h,axis=1)
    #compute forward pass, and marginalize over all k \in K
    logits = tf.nn.log_softmax(tf.einsum('ntkj,njy->ntky',h,tf.gather(q_W_y_,langs)))
    losses = -tf.reduce_sum(mask*tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*logits,-1),1)
    return(tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',tf.gather(f_U,langs),tf.exp(-losses))+1e-35)))


q_U,q_W_x,q_W_h,q_W_y = gen_var_dist()
U,W_x,W_h,W_y = gen_prior()


batch_size = 500
n_batches = int(N/batch_size)


llik = log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y)*N/batch_size
kldiv = tf.reduce_sum(q_U.kl_divergence(U)) + tf.reduce_sum(q_W_x.kl_divergence(W_x)) + tf.reduce_sum(q_W_h.kl_divergence(W_h)) + tf.reduce_sum(q_W_y.kl_divergence(W_y))

ELBO = llik - kldiv


tf.summary.scalar("ELBO", ELBO)


lr = .1                                 #learning rate
optimizer = tf.train.AdamOptimizer(lr,epsilon=.001)
train_op = optimizer.minimize(-ELBO)
check_op = tf.add_check_numerics_ops()
init=tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)



n_epochs = int(6000/n_batches)
idx = np.arange(N)
for c in range(4):
    ELBOs = []
    sess.run(init)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        langs_train,lens_train,inputs_train,outputs_train = langs_[idx],lens_[idx],inputs_[idx],outputs_[idx]
        for t in range(n_batches):
            start_time = time.time()
            langs_batch,lens_batch,inputs_batch,outputs_batch = langs_train[t * batch_size:(t + 1) * batch_size],lens_train[t * batch_size:(t + 1) * batch_size],inputs_train[t * batch_size:(t + 1) * batch_size],outputs_train[t * batch_size:(t + 1) * batch_size]
            _, ELBO_ = sess.run([(train_op,check_op), ELBO],feed_dict={langs:langs_batch,lens:lens_batch,inputs:inputs_batch,outputs:outputs_batch})
            ELBOs.append(ELBO_)
            duration = time.time() - start_time
            print("Step: {:>3d} Log ELBO: {:.3f} ({:.3f} sec)".format(epoch*int(n_batches)+t, ELBO_, duration))
    MAP = (sess.run(q_U.mean()),sess.run(q_U.stddev()),sess.run(q_W_x.mean()),sess.run(q_W_x.stddev()),sess.run(q_W_h.mean()),sess.run(q_W_h.stddev()),sess.run(q_W_y.mean()),sess.run(q_W_y.stddev()))
    f = open('posterior_discrete_RNN_VI_{}_{}.pkl'.format(c,lr),'wb')
    pkl.dump((ELBOs,MAP),f)
    f.close()


