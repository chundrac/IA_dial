#!usr/bin/env python3.4.3

import sys
import os
import numpy as np
import itertools
from collections import defaultdict
from numpy import log,exp,mean
from numpy.random import uniform
import re
import pymc3 as pm
from functools import reduce
import theano.tensor as tt
import theano
#theano.config.gcc.cxxflags = "-fbracket-depth=6000"
#THEANO_FLAGS = "fbracket-depth=6000"
import pickle as pkl


assert(pm.__version__=='3.3')


f = open('data_variables.pkl','rb')
segs_to_keep,changes_pruned,Cutoff,change_list,reflex,Sigma,K,S,X,R,N,langs,L,lang_ind,sound_ind,s_breaks,nchains = pkl.load(f)
f.close()


def logprob(theta,phi):
    def lprob(lang_array,sound_array):
        lps = pm.math.logsumexp(
                   tt.dot(lang_array      #N by L matrix
                   ,tt.log(theta)         #L by K matrix
                  )+                      #N by K matrix
            tt.dot(sound_array            #N by S matrix
                          ,tt.log(phi.T)  #S by K matrix
                         )                #N by K matrix      
        ,axis=1)                          #N-length vector                              
        return(tt.sum(lps))               #constant
    return(lprob)



def softmax_trans(l):
    l = tt.nnet.softmax(l)[0]**2
    z = tt.sum(l)
    return(l/z)



def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('usage: python3 inference_dir.py [chain no] [optional output no]')
        sys.exit()
    elif len(sys.argv) == 2:
        c = int(sys.argv[1])
        d = int(sys.argv[1])
    if len(sys.argv) == 3:
        c = int(sys.argv[1])
        d = int(sys.argv[2])
    np.random.seed(c)
    lang_minibatch = pm.Minibatch(lang_ind,500)
    sound_minibatch = pm.Minibatch(sound_ind,500)
    model_ln = pm.Model()
    with model_ln:
        beta = pm.HalfFlat('beta')
        "theta = language-level prior over components"
        theta = tt.stack([pm.Dirichlet('theta_{}'.format(l), a=tt.ones(K)*beta, shape=K) for l in range(L)])
        psi = [pm.MvNormal('psi_{}'.format(k),mu=[0]*S,cov=Sigma,shape=S) for k in range(K)]
        "phi = component-level collection of distributions over sound change"
        phi = tt.stack([tt.concatenate([pm.Deterministic('phi_{}_{}'.format(k,x),tt.nnet.softmax(psi[k][s_breaks[x][0]:s_breaks[x][1]])[0]) for x in range(X)]) for k in range(K)])
        #phi = tt.stack([tt.concatenate([pm.Deterministic('phi_{}_{}'.format(k,x),softmax_trans(psi[k][s_breaks[x][0]:s_breaks[x][1]])) for x in range(X)]) for k in range(K)])
        target = pm.DensityDist('target',logprob(theta=theta,phi=phi),observed=dict(lang_array=lang_minibatch,sound_array=sound_minibatch),total_size=N)
        inference_ln = pm.ADVI()
        inference_ln.fit(50000, obj_optimizer=pm.adam(learning_rate=.01,beta1=uniform(.7,.9)),callbacks=[pm.callbacks.CheckParametersConvergence()])
        trace_ln = inference_ln.approx.sample()
        posterior = {k:trace_ln[k] for k in trace_ln.varnames if not k.endswith('__')}
        posterior['ELBO'] = inference_ln.hist
        f = open('posterior_ln_{}.pkl'.format(d),'wb')
        pkl.dump(posterior,f)
        f.close()



if __name__=="__main__":
    main()
