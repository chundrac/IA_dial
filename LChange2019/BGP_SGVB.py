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


delta_change = np.zeros([S,S])
delta_env = np.zeros([S,S])


for i in range(len(change_list)):
    for j in range(i):
        if change_list[i][0][1] != change_list[j][0][1] or change_list[i][1] != change_list[j][1]:
            delta_change[i,j] = delta_change[j,i] = 1
        if change_list[i][0][0] != change_list[j][0][0] or change_list[i][0][2] != change_list[j][0][2]:
            delta_env[i,j] = delta_env[j,i] = 1


delta = np.stack([delta_change,delta_env])
D = delta.shape[0]



def GEM(beta):  #stick-breaking construction
    pi = tf.concat([[1.], tf.cumprod(1 - beta)],axis=0)
    return(tf.concat([beta,[1.]],axis=0) * pi)



def make_priors():
    priors = {}
    priors['eta'] = tfd.Normal(tf.zeros([L,K]),10.)
    priors['z'] = tfd.Normal(tf.zeros([S,K]),1.)
    priors['alpha'] = tfd.Normal(tf.zeros([1,1]),.1)      #tfd.Normal(0.,1.)
    priors['inv_rhos'] = tfd.Normal(tf.zeros([D,1,1]),.1)
    priors['sigma'] = tfd.Normal(tf.zeros([1,1]),10.)      #tfd.Normal(0.,10.)
    return(priors)



def make_variational_params(priors):
    var_params = {}
    for k in priors.keys():
        var_params[k] = tfd.Normal(
                        tf.get_variable('{}_loc'.format(k),priors[k].batch_shape),
                        tf.nn.softplus(tf.get_variable('{}_scale'.format(k),priors[k].batch_shape))
                        )
    return(var_params)



def SEK(delta,alpha,inv_rhos,sigma):
    return(tf.cholesky(tf.pow(alpha,2)*tf.exp(-.5*tf.reduce_sum(delta*tf.pow(inv_rhos,2),1))+tf.expand_dims(tf.eye(S),0)*(tf.pow(sigma,2)+1e-5)))



def log_lik(var_params,lang_array,sound_array,n_eps):
    curr = {k:tf.expand_dims(var_params[k].mean(),0)+
            tf.expand_dims(var_params[k].stddev(),0)*
            tf.random_normal(shape=[n_eps]+
            var_params[k].stddev().get_shape().as_list()) 
            for k in var_params.keys()}
    theta = tf.nn.log_softmax(curr['eta'])
    L = SEK(delta,curr['alpha'],curr['inv_rhos'],curr['sigma'])
    psi = tf.transpose(tf.matmul(L,curr['z']),[0,2,1])
    phi = tf.concat([tf.nn.log_softmax(psi[:,partition[x][0]:partition[x][1]]) for x in range(X)],axis=1)
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


n_iters = 5000
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
    n_eps_ = 10
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



f = open('BGP_var_params.pkl','wb')
pkl.dump(posterior,f)
f.close()
