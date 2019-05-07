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
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Bidirectional,LSTM,TimeDistributed,Dense,RepeatVector,Lambda



text = []

for l in open('cdial_stripped.csv','r'):
    text.append(l.strip().split('\t'))



langs_raw = []
outputs_raw = []

for l in text:
    if l[0] != 'Pa.' and l[0] != 'Pk.':
        outputs_raw.append(l[1].split())
        langs_raw.append(l[0])            


output_list = sorted(set([s for l in outputs_raw for s in l]))

T = max([len(l) for l in outputs_raw])
Y = len(output_list)
N = len(outputs_raw)

outputs_ = np.zeros([N,T,Y])
for i,l in enumerate(outputs_raw):
    for j,s in enumerate(l):
        outputs_[i,j,output_list.index(s)] = 1


J = 100
model = Sequential()
model.add(LSTM(J,activation='softmax',input_shape=(T,Y),mask=))
model.add(RepeatVector(T))
model.add(LSTM(J, activation='softmax', return_sequences=True))
model.add(TimeDistributed(Dense(Y)))
model.compile(optimizer='adam', loss='categorical_crossentropy')


model.fit(outputs_,outputs_,epochs=100,batch_size=32)