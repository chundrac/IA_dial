#!usr/bin/env python3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import pymc3 as pm
import theano.tensor as tt
import theano
theano.config.gcc.cxxflags = "-fbracket-depth=6000"
#THEANO_FLAGS = "fbracket-depth=6000"
import pickle as pkl
import time


f = open('data_and_variables.pkl','rb')
langs_,lens_,inputs_,outputs_,T,N,X,Y,L = pkl.load(f)
f.close()


mask_ = np.zeros([N,T])
for i in range(N):
    mask_[i,:lens_[i]] = 1


K = 10
J = 100


def loglik(U,W_x,W_h,W_y):
    def llik(langs,mask,inputs,outputs):
        f_U = tt.nn.softmax(U)
        h = []
        for t in range(T):
            if t == 0:
                h.append(tt.nnet.softmax(W_x[inputs_array[:,t],:,:]))
            else:
                h.append(tt.nnet.softmax(tt.batched_tensordot(h[t-1],W_h,(2,1))))
        h = tt.stack(h,axis=1)
        logits = tt.nnet.logsoftmax(tt.batched_tensordot(h,W_y[lang_array,:],(3,1)))
        losses = -tt.sum(mask*tf.sum(tt.extra_ops.to_one_hot(outputs,Y)*logits,-1),axis=1)
        return(tt.sum(tt.log(tt.batched_tensordot(f_U[langs,:],tt.exp(-losses),1))))



langs_minibatch = pm.Minibatch(langs_,500)
mask_minibatch = pm.Minibatch(mask_,500)
inputs_minibatch = pm.Minibatch(inputs_,500)
outputs_minibatch = pm.Minibatch(outputs_,500)



model = pm.Model()
with model:
    U = tt.stack([[pm.Normal('U_{}_{}'.format(l,k),0.,10.) for k in range(K)] for l in range(L)])
    W_x = tt.stack([[[pm.Normal('W_x_{}_{}_{}'.format(x,k,j),0.,10.) for j in range(J)] for k in range(K)] for x in range(X)])
    W_h = tt.stack([[[pm.Normal('W_h_{}_{}_{}'.format(k,i,j),0.,10.) for j in range(J)] for i in range(J)] for k in range(K)])
    W_y = tt.stack([[[pm.Normal('W_y_{}_{}_{}'.format(l,j,y),0.,10.) for y in range(Y)] for j in range(J)] for l in range(L)])
    target = pm.Density('target',loglik(U=U,W_x=W_x,W_h=W_h,W_y=W_y),observed=dict(langs=langs_minibatch,mask=mask_minibatch,inputs=inputs_minibatch,outputs=outputs_minibatch),total_size=N)
    inference = pm.ADVI()
    inference_ln.fit(50000, obj_optimizer=pm.adam(learning_rate=.01,beta1=uniform(.7,.9)),callbacks=[pm.callbacks.CheckParametersConvergence()])
    