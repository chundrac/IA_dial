#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import tensorflow as tf
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
inputs = np.zeros([N,T],dtype=np.int64)
outputs = np.zeros([N,T],dtype=np.int64)


for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        inputs[i,j] = input_list.index(s)


for i,l in enumerate(outputs_raw):
    for j,s in enumerate(l):
        outputs[i,j] = output_list.index(s)


lens = np.array([len(l) for l in inputs_raw],dtype=np.int64)


langs_ = langs
lens_ = lens
inputs_ = inputs
outputs_ = outputs


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




def log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y):
    mask = tf.expand_dims(tf.sequence_mask(lens,maxlen=T,dtype=tf.float32),2)
    f_U = tf.nn.softmax(q_U)
    #for each batch, for each timepoint, for each component, compute hidden layer activation
    h = []
    for t in range(T):
        if t == 0:
            h.append(tf.nn.softmax(tf.einsum('nk,nki->nki',tf.gather(f_U,langs),tf.gather(q_W_x,inputs[:,t]))))
        else:
            h.append(tf.nn.softmax(tf.einsum('nki,kij->nkj',h[t-1],q_W_h) + tf.gather(q_W_x,inputs[:,t])))
    h = tf.stack(h,axis=1)
    #compute forward pass
    pi = tf.nn.softmax(tf.einsum('ntkj,njy->ntky',h,tf.gather(q_W_y,langs)))
    #exclude masked indices 
    out = tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*pi,-1)
    out = tf.exp(tf.reduce_sum(mask*tf.log(out),1))
    return(tf.reduce_sum(tf.log(tf.reduce_sum(out,1)+1e-20)))


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
    #compute forward pass
    logits = tf.einsum('ntkj,njy->ntky',h,tf.gather(q_W_y,langs))
    #exclude masked indices 
    out = tf.reduce_sum(tf.expand_dims(tf.one_hot(outputs,depth=Y),2)*pi,-1)
    out = tf.exp(tf.reduce_sum(mask*tf.log(out),1))
    return(tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',tf.gather(f_U,langs),out)+1e-20)))


q_U,q_W_x,q_W_h,q_W_y = gen_params()


n_batches = int(N/100)
batch_size = int(N/n_batches)


llik = log_lik(langs,lens,inputs,outputs,q_U,q_W_x,q_W_h,q_W_y)*N/batch_size


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
        if epoch in range(int(n_epochs/2),n_epochs):
            MAPs[c].append([sess.run(q_U),sess.run(q_V),sess.run(q_W_h),sess.run(q_W_i),sess.run(q_W_j)])


