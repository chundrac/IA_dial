#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO,TraceEnum_ELBO
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta,AutoGuide,AutoContinuous,AutoDiagonalNormal
import time


text = []

for l in open('cdial_stripped.csv','r'):
    text.append(l.strip().split('\t'))


for i in range(len(text)):
    for j in range(1,3):
        text[i][j] = text[i][j].split()


etym_counts = defaultdict(int)
for l in text:
    etym_counts[tuple(l[2])]+=1



alignments = []
for l in open('alignments.txt','r'):
    alignments.append([int (i) for i in l.split()])


langs_raw = []
inputs_raw = []
outputs_raw = []
for i,l in enumerate(text):
  if l[0] != 'Pa.' and l[0] != 'Pk.' and etym_counts[tuple(l[2])] > 10:
    inputs = []
    outputs = []
    x,y=text[i][2],text[i][1]
    A = alignments[i]
    lang = l[0]
    etymon = ''.join(l[2])
    for j in range(1,len(A)-2):
        inputs.append(''.join(x[j-1:j+2]))
        outputs.append(''.join(y[A[j]:A[j+1]]))
    langs_raw.append(lang)
    inputs_raw.append(inputs)
    outputs_raw.append(outputs)


lang_list = sorted(set(langs_raw))
input_list = sorted(set([s for l in inputs_raw for s in l]))
output_list = sorted(set([s for l in outputs_raw for s in l]))


T = max([len(l) for l in inputs_raw])
N = len(langs_raw)
X = len(input_list)
Y = len(output_list)
L = len(lang_list)


langs = np.array([lang_list.index(l) for l in langs_raw],dtype=np.int64)
#inputs = np.zeros([N,T,X],dtype=np.float64)
inputs = np.zeros([N,T],dtype=np.int64)
outputs = np.zeros([N,T],dtype=np.int64)

for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        inputs[i,j] = input_list.index(s)


for i,l in enumerate(outputs_raw):
    for j,s in enumerate(l):
        outputs[i,j] = output_list.index(s)


lengths = torch.tensor(np.array([len(l) for l in inputs_raw],dtype=np.int64))


langs = torch.tensor(langs)
inputs = torch.tensor(inputs)
outputs = torch.tensor(outputs)
mask_inds = torch.arange(T)[None, :] < lengths[:, None]


K = 10
J = 100


batch_size = 500
def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros(L*K),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros(X*K*Y),10.))
    U_ = U.reshape([L,K])
    W_ = W.reshape([X,K,Y])
#    with pyro.plate('data_loop',N,batch_size) as ind:
    z = pyro.sample('z',dist.Normal(U_[langs[ind],:],torch.ones([batch_size,K])))
    for t in range(T):
            pi = torch.softmax(torch.einsum('nk,nky->ny',z,W_[inputs[ind,t],:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    with pyro.plate('data_loop',N,batch_size) as ind:
        #z = pyro.sample('z',dist.Normal(U_[langs[ind],:],torch.ones([batch_size,K])))
        z = U_[langs[ind],:]+pyro.sample('z',dist.Normal(torch.zeros([batch_size,K]),torch.ones([batch_size,K])))
        for t in range(T):
            pi = torch.softmax(torch.einsum('nk,nky->ny',z,W_[inputs[ind,t],:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    z = U_[langs,:]+pyro.sample('z',dist.Normal(torch.zeros([N,K]),torch.ones([N,K])))
    for t in range(T):
        pi = torch.softmax(torch.einsum('nk,nky->ny',z,W_[inputs[:,t],:,:]),-1)
        pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[:,t]),obs=outputs[:,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    with pyro.plate('data_loop',N,batch_size) as ind:
        z = U[langs[ind],:]+pyro.sample('z',dist.Normal(torch.zeros([batch_size,K]),torch.ones([batch_size,K])))
        for t in range(T):
            pi = torch.softmax(torch.einsum('nk,nky->ny',z,W[inputs[ind,t],:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    for ind in pyro.plate('data_loop',N,batch_size):
        z = pyro.sample('z_{}'.format(ind),dist.Normal(U[langs[ind],:],1.))
        for t in range(T):
            pi = torch.softmax(torch.einsum('k,ky->y',z,W[inputs[ind,t],:,:]),-1)
            pyro.sample('outputs_{}_{}'.format(ind,t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    with pyro.plate('data_loop',N,batch_size) as ind:
        z = pyro.sample('z',dist.Normal(U[langs[ind],:],1.).to_event(1))
        print(z.shape)
        for t in range(T):
            print(W[inputs[ind,t],:,:].shape)#==batch_size)
            pi = torch.softmax(torch.einsum('nk,nky->ny',z,W[inputs[ind,t],:,:]),-1)
            #pi = torch.softmax(torch.einsum('nk,nky->ny',z,W[inputs[:,t],:,:][ind,:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    z = pyro.sample('z',dist.Normal(U[langs,:],1.))
    with pyro.plate('data_loop',N,batch_size) as ind:
        print(z.shape)
        for t in range(T):
            print(W[inputs[ind,t],:,:].shape)#==batch_size)
            pi = torch.softmax(torch.einsum('nk,nky->ny',z[ind],W[inputs[ind,t],:,:]),-1)
            #pi = torch.softmax(torch.einsum('nk,nky->ny',z,W[inputs[:,t],:,:][ind,:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model2(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    with pyro.plate('data_loop',N,batch_size) as ind:
        z = pyro.sample('z',dist.Normal(U[langs.index_select(0,ind),:],1.))
        for t in range(T):
            pi = torch.softmax(torch.einsum('nk,nky->ny',z,W[inputs[:,t].index_select(0,ind),:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[:,t].index_select(0,ind)),obs=outputs[:,t].index_select(0,ind))



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W = pyro.sample('W',dist.Normal(torch.zeros([X,K,Y]),10.))
    with pyro.plate('data_loop',N,batch_size) as ind:
        z = pyro.sample('z',dist.Normal(torch.zeros([batch_size,K]),1.).to_event(1))
        for t in range(T):
            pi = torch.softmax(torch.einsum('nk,nky->ny',U[langs[ind],:]+z,W[inputs[ind,t],:,:]),-1)
            #pi = torch.softmax(torch.einsum('nk,nky->ny',z,W[inputs[:,t],:,:][ind,:,:]),-1)
            pyro.sample('outputs_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])

                   
            

softplus=torch.nn.Softplus()
def guide(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    mu_U = pyro.param('mu_U',torch.randn([L,K]))
    sd_U = pyro.param('sd_U',softplus(torch.randn([L,K])),constraint=constraints.positive)
    mu_W = pyro.param('mu_W',torch.randn([X,K,Y]))
    sd_W = pyro.param('sd_W',softplus(torch.randn([X,K,Y])),constraint=constraints.positive)
    #mu_z = pyro.param('mu_z',torch.randn([N,K]))
    #sd_z = pyro.param('sd_z',softplus(torch.randn([N,K])),constraint=constraints.positive)
    pyro.sample('U',dist.Normal(mu_U,sd_U))
    pyro.sample('W',dist.Normal(mu_W,sd_W))
    #pyro.sample('z',dist.Normal(mu_z,sd_z))




guide = AutoDiagonalNormal(poutine.block(model,expose=['U','W']))
guide = guide
n_steps = 10000
# setup the optimizer
adam_params = {"lr": .15, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())


losses = defaultdict(list)
for c in range(3):
    pyro.clear_param_store()
    for step in range(n_steps):
        start_time = time.time()
        print(step,end=' ')
        loss = svi.step(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L)
        print(loss,time.time() - start_time)
        losses[c].append(loss)