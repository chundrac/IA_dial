#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import tensorflow as tf
import pickle as pkl


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


#langs = tf.placeholder(tf.int32,shape=(None,),name='langs')
#lens = tf.placeholder(tf.int32,shape=(None,),name='lens')
#inputs = tf.placeholder(tf.int32,shape=(None,T),name='inputs')
#outputs = tf.placeholder(tf.int32,shape=(None,T),name='outputs')

data_and_variables = (langs_,lens_,inputs_,outputs_,T,N,X,Y,L,lang_list)
f = open('data_and_variables.pkl','wb')
pkl.dump(data_and_variables,f)
f.close()