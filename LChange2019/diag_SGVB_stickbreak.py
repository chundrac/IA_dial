#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import pickle as pkl
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time


f = open('sound_changes.pkl','rb')
change_list,S,X,R,N,L,partition,lang_ind,sound_ind,lang_list = pkl.load(f)
f.close()

lang_ind = np.array(lang_ind,dtype=np.float32)
sound_ind = np.array(sound_ind,dtype=np.float32)

lang_array = tf.placeholder(tf.float32,shape=(None,lang_ind.shape[1]),name='lang_array')
sound_array = tf.placeholder(tf.float32,shape=(None,sound_ind.shape[1]),name='sound_array')
n_eps = tf.placeholder(tf.int32,shape=(),name='n_eps')

K = 10


def GEM(beta):  #stick-breaking construction
    pi = tf.concat([tf.ones([n_eps,1]), tf.cumprod(1 - beta,axis=1)],axis=1)
    return(tf.concat([beta,tf.ones([n_eps,1])],axis=1) * pi)


def laplace_Dir(alpha,shape):
    mu = tf.log(alpha) - tf.expand_dims(tf.reduce_mean(tf.log(alpha),axis=1),1)
    sd = (1/alpha)*(1-(2/shape))+((1/shape**2)*tf.expand_dims(tf.reduce_sum(1/alpha,axis=1),axis=1))
    return(mu,sd)


def make_priors():
    priors = {}
    priors['W'] = tfd.Normal(tf.zeros(K-1),10.)
    priors['tau'] = tfd.Normal(tf.zeros(K-1),1.)
    priors['eta'] = tfd.Normal(tf.zeros([L,K]),1.)
    priors['psi'] = tfd.Normal(tf.zeros([K,S]),10.)
    return(priors)


def make_variational_params(priors):
    var_params = {}
    for k in priors.keys():
        var_params[k] = tfd.Normal(
                        tf.get_variable('{}_loc'.format(k),priors[k].batch_shape),
                        tf.nn.softplus(tf.get_variable('{}_scale'.format(k),priors[k].batch_shape))
                        )
    return(var_params)


def log_lik(var_params,lang_array,sound_array,n_eps):
    curr = {k:tf.expand_dims(var_params[k].mean(),0)+
            tf.expand_dims(var_params[k].stddev(),0)*
            tf.random_normal(shape=[n_eps]+
            var_params[k].stddev().get_shape().as_list()) 
            for k in var_params.keys()}
    zeta = GEM(tf.nn.sigmoid(curr['W']*curr['tau']))
    mu_zeta,sd_zeta = laplace_Dir(zeta,K)
    theta = tf.log(tf.nn.softmax(tf.expand_dims(mu_zeta,1) + curr['eta']*tf.expand_dims(sd_zeta,1))+1e-15)
    #theta = tf.nn.log_softmax(tf.expand_dims(mu_zeta,1) + curr['eta']*tf.expand_dims(sd_zeta,1))
    phi = tf.concat([tf.nn.log_softmax(curr['psi'][:,partition[x][0]:partition[x][1]]) for x in range(X)],axis=1)
    llik = tf.reduce_sum(tf.reduce_logsumexp(tf.einsum('nl,elk->enk',lang_array,theta) + tf.einsum('ns,esk->enk',sound_array,tf.transpose(phi,[0,2,1])),axis=2),axis=1)
    return(tf.reduce_mean(llik))


def get_kl_div(priors,var_params):
    return(tf.reduce_sum([tf.reduce_sum(var_params[k].kl_divergence(priors[k])) for k in priors.keys()]))


priors = make_priors()
var_params = make_variational_params(priors)


batch_size = N
llik = log_lik(var_params,lang_array,sound_array,n_eps)*N/batch_size
kldiv = get_kl_div(priors,var_params)
ELBO = llik-kldiv

tf.summary.scalar("ELBO", ELBO)

optimizer = tf.train.AdamOptimizer(.1)
#gradients, variables = zip(*optimizer.compute_gradients(-ELBO))
#gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
#train_op = optimizer.apply_gradients(zip(gradients, variables))
train_op = optimizer.minimize(-ELBO)
check_op = tf.add_check_numerics_ops()
init=tf.global_variables_initializer()


n_iters = 10000
n_epochs = int(n_iters*batch_size/N)
chains = 3
posterior = {}
idx = np.arange(N)
for c in range(chains):
    ELBOs = []
    posterior[c] = {}
    np.random.seed(0)
    tf.set_random_seed(0)
    sess = tf.Session()
    sess.run(init)
    n_eps_ = 1
    for epoch in range(n_epochs):
        #if epoch in range(int(n_epochs/100)):
        #    n_eps_ = 100
        #elif epoch in range(int(n_epochs/100),int(n_epochs/10)):
        #    n_eps_ = 50
        #else:
        #n_eps_ = 1
        np.random.shuffle(idx)
        lang_train,sound_train=lang_ind[idx],sound_ind[idx]
        for t in range(int(N/batch_size)):
            print(c,epoch*int(N/batch_size)+t,end=' ')
            lang_minibatch,sound_minibatch = lang_train[t * batch_size:(t + 1) * batch_size],sound_train[t * batch_size:(t + 1) * batch_size]
            start_time = time.time()
            _, ELBO_ = sess.run([(train_op, check_op), ELBO],feed_dict={lang_array:lang_minibatch,sound_array:sound_minibatch,n_eps:n_eps_})
            duration = time.time() - start_time
            ELBOs.append(ELBO_)
            print(duration,ELBO_)
    for k in var_params.keys():
        posterior[c][k] = (sess.run(var_params[k].mean()),sess.run(var_params[k].stddev()))
    posterior[c]['ELBO'] = ELBOs



f = open('diag_var_params.pkl','wb')
pkl.dump(posterior,f)
f.close()
