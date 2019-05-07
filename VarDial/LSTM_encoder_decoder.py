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
from tensorflow.keras.layers import Input,Bidirectional,LSTM,TimeDistributed,Dense,RepeatVector,Lambda



text = []

for l in open('cdial_stripped.csv','r'):
    text.append(l.strip().split('\t'))



langs_raw = []
inputs_raw = []
outputs_raw = []

for l in text:
    if l[0] != 'Pa.' and l[0] != 'Pk.':
        inputs_raw.append(l[2].split())
        outputs_raw.append(l[1].split())
        langs_raw.append(l[0])            


output_list = sorted(set([s for l in outputs_raw for s in l]))
input_list = sorted(set([s for l in inputs_raw for s in l]))

T_i = max([len(l) for l in inputs_raw])
T_o = max([len(l) for l in outputs_raw])
X = len(input_list)
Y = len(output_list)
N = len(outputs_raw)
J = 12


inputs_ = np.zeros([N,T_i,X])
for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        inputs_[i,j,input_list.index(s)] = 1



outputs_ = np.zeros([N,T_o,Y])
for i,l in enumerate(outputs_raw):
    for j,s in enumerate(l):
        outputs_[i,j,output_list.index(s)] = 1




input = Input(shape=(T_i,X))
hidden = Bidirectional(LSTM(J, activation='softmax'),merge_mode='sum')(input)
hidden = RepeatVector(T_o)(hidden)
hidden = LSTM(J, activation='softmax', return_sequences=True)(hidden)
output = TimeDistributed(Dense(Y))(hidden)

model = Model(input,output)
model.compile(optimizer='adam', loss='categorical_crossentropy')


model.fit(inputs_,outputs_,epochs=20,batch_size=32)
model_fitted = Model(inputs=model.inputs, outputs=model.layers[1].output)