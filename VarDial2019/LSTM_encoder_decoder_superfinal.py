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
from tensorflow.keras.layers import Lambda,Input,Bidirectional,LSTM,TimeDistributed,Dense,RepeatVector,Lambda
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import mpu


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
langs_list = sorted(set(langs_raw))
L = len(langs_list)

T_i = max([len(l) for l in inputs_raw])
T_o = max([len(l) for l in outputs_raw])
X = len(input_list)
Y = len(output_list)
N = len(outputs_raw)
J = 8


inputs_ = np.zeros([N,T_i,X])
for i,l in enumerate(inputs_raw):
    for j,s in enumerate(l):
        inputs_[i,j,input_list.index(s)] = 1



outputs_ = np.zeros([N,T_o,Y])
for i,l in enumerate(outputs_raw):
    for j,s in enumerate(l):
        outputs_[i,j,output_list.index(s)] = 1


input = Input(shape=(T_i,X), dtype='int32')
tofloat = Lambda(function=lambda x: tf.cast(x,'float32'))(input)
#hidden = Bidirectional(LSTM(J, activation='softmax'),merge_mode='sum')(tofloat)
hidden = Bidirectional(LSTM(J),merge_mode='sum')(tofloat)
hidden = RepeatVector(T_o)(hidden)
hidden = LSTM(J, activation='softmax', return_sequences=True)(hidden)
output = TimeDistributed(Dense(Y, activation='softmax'))(hidden)
model = Model(input,output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(inputs_,outputs_,epochs=20,batch_size=32)
model_fitted = Model(inputs=model.inputs, outputs=model.layers[2].output)


latent = model_fitted.predict(inputs_)
latent = (latent-np.min(latent,0))

f = open('LSTM_ED_embeddings.pkl','wb')
pkl.dump(latent,f)
f.close()


lang_reps = defaultdict(list)
for i in range(N):
    lang_reps[langs_raw[i]].append(latent[i])


for k in lang_reps.keys():
    lang_reps[k] = np.mean(np.array(lang_reps[k]),0)


embed = np.array([lang_reps[k] for k in langs_list])


avgdists = {}
for i in range(L):
    for j in range(i+1,L):
        avgdists[(i,j)]=(sum((embed[i]-embed[j])**2))**.5


coph_dists_full = {}
for l in open('patristic.txt','r'):
    line = l.strip().split()
    coph_dists_full[(langs_list.index(line[0]),langs_list.index(line[1]))] = float(line[2])



coph_dists = {}
for k in avgdists.keys():
    coph_dists[k] = coph_dists_full[k]



locations = []
for l in open('glottolog_IE.csv','r'):
    locations.append(l.strip().split('\t'))


lon = {}
lat = {}
for l in locations:
    if len(l) > 7:
        lon[l[0]] = float(l[7])
        lat[l[0]] = float(l[6])



geo_dists = {}
for i,j in avgdists.keys():
    geo_dists[(i,j)]=mpu.haversine_distance((lat[langs_list[i]],lon[langs_list[i]]),(lat[langs_list[j]],lon[langs_list[j]]))



cor_geo = spearmanr(list(geo_dists.values()),list(avgdists.values()))
cor_gen = spearmanr(list(coph_dists.values()),list(avgdists.values()))
print('geographic correlation: {}, {}'.format(cor_geo[0],cor_geo[1]))
print('genetic correlation: {}, {}'.format(cor_gen[0],cor_gen[1]))