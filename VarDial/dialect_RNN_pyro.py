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



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L*K]),10.))
    W_x = pyro.sample('W_x',dist.Normal(torch.zeros([K*X*J]),10.))
    W_h = pyro.sample('W_h',dist.Normal(torch.zeros([K*J*J]),10.))
    W_y = pyro.sample('W_y',dist.Normal(torch.zeros([L*J*Y]),10.))
    U_ = U.reshape([L,K])
    W_x_ = W_x.reshape([K,X,J])
    W_h_ = W_h.reshape([K,J,J])
    W_y_ = W_y.reshape([L,J,Y])
    f_U = torch.softmax(U_,-1)
    with pyro.plate('data_loop',N,500) as ind:
        z = pyro.sample('z',dist.Categorical(f_U[langs[ind],:]))
        h = None
        for t in range(T):
            if t == 0:
                h = torch.softmax(W_x_[z,inputs[ind,t],:],-1)
            else:
                h = torch.softmax(torch.einsum('ni,nij->nj',h,W_h_[z,:,:]) + W_x_[z,inputs[ind,t],:],-1)
            pi = torch.softmax(torch.einsum('nj,njy->ny',h,W_y_[langs[ind],:,:]),-1)
            pyro.sample('obs_y_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W_x = pyro.sample('W_x',dist.Normal(torch.zeros([K,X,J]),10.))
    W_h = pyro.sample('W_h',dist.Normal(torch.zeros([K,J,J]),10.))
    W_y = pyro.sample('W_y',dist.Normal(torch.zeros([L,J,Y]),10.))
    f_U = torch.softmax(U,-1)
    with pyro.plate('data_loop',N,500) as ind:
        print(f_U[langs[ind],:].shape)
        z = pyro.sample('z',dist.Categorical(f_U[langs[ind],:]),infer={"enumerate": "sequential"})
        print(z.shape)
        h = None
        for t in range(T):
            print(t)
            if t == 0:
                h = torch.softmax(W_x[z,inputs[ind,t],:],-1)
                print(h.shape)
            else:
                h = torch.softmax(torch.einsum('ni,nij->nj',h,W_h[z,:,:]) + W_x[z,inputs[ind,t],:],-1)
            print(W_y[langs[ind],:,:].shape)
            pi = torch.softmax(torch.einsum('nj,njy->ny',h,W_y[langs[ind],:,:]),-1)
            pyro.sample('obs_y_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros(L*K),10.))
    W_x = pyro.sample('W_x',dist.Normal(torch.zeros(K*X*J),10.))
    W_h = pyro.sample('W_h',dist.Normal(torch.zeros(K*J*J),10.))
    W_y = pyro.sample('W_y',dist.Normal(torch.zeros(L*J*Y),10.))
    U_ = U.reshape([L,K])
    W_x_ = W_x.reshape([K,X,J])
    W_h_ = W_h.reshape([K,J,J])
    W_y_ = W_y.reshape([L,J,Y])
    f_U = torch.softmax(U_,-1)
    with pyro.plate('data_loop',N,500) as ind:
        z = pyro.sample('z',dist.Categorical(f_U[langs[ind],:]),infer={"enumerate": "sequential"})
        h = None
        for t in range(T):
            if t == 0:
                h = torch.softmax(W_x_[z,inputs[ind,t],:],-1)
            else:
                h = torch.softmax(torch.einsum('ni,nij->nj',h,W_h_[z,:,:]) + W_x_[z,inputs[ind,t],:],-1)
            pi = torch.softmax(torch.einsum('nj,njy->ny',h,W_y_[langs[ind],:,:]),-1)
            pyro.sample('obs_y_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])

            

guide = AutoDiagonalNormal(poutine.block(model,expose=['U','W_x','W_h','W_y']))






def model(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    U = pyro.sample('U',dist.Normal(torch.zeros([L,K]),10.))
    W_x = pyro.sample('W_x',dist.Normal(torch.zeros([K,X,J]),10.))
    W_h = pyro.sample('W_h',dist.Normal(torch.zeros([K,J,J]),10.))
    W_y = pyro.sample('W_y',dist.Normal(torch.zeros([L,J,Y]),10.))
    f_U = torch.softmax(U,-1)
    with pyro.plate('data_loop',N,500) as ind:
        print(f_U[langs[ind],:].shape)
        z = pyro.sample('z',dist.Categorical(f_U[langs[ind],:]),infer={"enumerate": "sequential"})
        print(z.shape)
        h = None
        for t in range(T):
            print(t)
            if t == 0:
                h = torch.softmax(W_x[z,inputs[ind,t],:],-1)
                print(h.shape)
            else:
                h = torch.softmax(torch.einsum('ni,nij->nj',h,W_h[z,:,:]) + W_x[z,inputs[ind,t],:],-1)
            print(W_y[langs[ind],:,:].shape)
            pi = torch.softmax(torch.einsum('nj,njy->ny',h,W_y[langs[ind],:,:]),-1)
            pyro.sample('obs_y_{}'.format(t),dist.Categorical(pi).mask(mask_inds[ind,t]),obs=outputs[ind,t])



softplus=torch.nn.Softplus()
def guide(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L):
    l_U = pyro.param('l_U',torch.randn([L,K]))
    s_U = pyro.param('s_U',softplus(torch.randn([L,K])),constraint=constraints.positive)
    l_W_x = pyro.param('l_W_x',torch.randn([K,X,J]))
    s_W_x = pyro.param('s_W_x',softplus(torch.randn([K,X,J])),constraint=constraints.positive)
    l_W_h = pyro.param('l_W_h',torch.randn([K,J,J]))
    s_W_h = pyro.param('s_W_h',softplus(torch.randn([K,J,J])),constraint=constraints.positive)
    l_W_y = pyro.param('l_W_y',torch.randn([L,J,Y]))
    s_W_y = pyro.param('s_W_y',softplus(torch.randn([L,J,Y])),constraint=constraints.positive)
    pyro.sample('U',dist.Normal(l_U,s_U))
    pyro.sample('W_x',dist.Normal(l_W_x,s_W_x))
    pyro.sample('W_h',dist.Normal(l_W_h,s_W_h))
    pyro.sample('W_y',dist.Normal(l_W_y,s_W_y))













n_steps = 10000
# setup the optimizer
adam_params = {"lr": .15, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=0))


losses = defaultdict(list)
for c in range(3):
    pyro.clear_param_store()
    for step in range(n_steps):
        start_time = time.time()
        print(step,end=' ')
        loss = svi.step(langs,inputs,outputs,mask_inds,K,J,T,N,X,Y,L)
        print(loss,time.time() - start_time)
        #losses[c].append(loss)
        
        