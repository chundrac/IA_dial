#!usr/bin/env python3.4.3

import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import pickle as pkl
#from copy import copy



text = []

for l in open('cdial_stripped.csv','r'):
    text.append(l.strip().split('\t'))


for i in range(len(text)):
    for j in range(1,3):
        text[i][j] = text[i][j].split()



alignments = []
for l in open('alignments.txt','r'):
    alignments.append([int (i) for i in l.split()])


    
lang_counts = defaultdict(int)
for l in text:
    lang_counts[l[0]] += 1


exclude = [k for k in lang_counts.keys() if lang_counts[k] < 100] + ['Pa.','Pk.']


change_counts=defaultdict(int)
changes = []
for i in range(len(text)):
    if text[i][0] not in exclude:
        lang = text[i][0]
        word_changes = []
        x,y=text[i][2],text[i][1]
        A = alignments[i]
        for j in range(1,len(A)-2):
            edit = (tuple(x[j-1:j+2]),tuple(y[A[j]:A[j+1]]))
            word_changes.append(edit)
            change_counts[edit]+=1
        changes.append([lang,word_changes])



Cutoff = 7


all_reflex = defaultdict(list)
for k in change_counts.keys():
    if change_counts[k] > Cutoff:
        all_reflex[k[0]].append(k[1])
        

reflex = defaultdict(list)
for k in sorted(list(all_reflex.keys())):
    if len(all_reflex[k]) > 1:
        reflex[k] = sorted(all_reflex[k])


change_list = [(k,v) for k in reflex.keys() for v in reflex[k]]



segs_to_keep = [#'J',
#'N',
'S',
'V',
'\\*n',
#'\\:d',
#'\\:d\\tsup{H}',
'\\:n',
'\\:s',
#'\\:t',
#'\\:t\\tsup{h}',
#'\\;N',
'\\s{r}',
#'a',
#'a:',
#'b',
#'b\\tsup{H}',
#'c',
#'c\\tsup{h}',
#'d',
#'d\\tsup{H}',
#'e',
#'g',
#'g\\tsup{h}',
'h',
'i',
'i:',
'j',
#'k',
'k\\:s',
#'k\\tsup{h}',
'l',
#'m',
'n',
#'o',
#'p',
'r',
's',
#'t',
#'t\\tsup{h}',
'u',
'u:']






final_change_list = [s for s in change_list if s[0][1] in segs_to_keep and s[0][2] != '#'] #make sure weird stuff not happening



final_reflex = defaultdict(list)
for k in reflex.keys():
    if k[1] in segs_to_keep and k[2] != '#':
        final_reflex[k] = reflex[k]



change_list = sorted(final_change_list)
reflex = {k:final_reflex[k] for k in sorted(final_reflex.keys())}



changes_pruned = []
for l in changes:
    new_line = [l[0],[s for s in l[1] if s in change_list]]
    if new_line[1] != []:
        changes_pruned.append(new_line)



S = len(change_list)
X = len(reflex.keys())
R = [len(reflex[k]) for k in reflex.keys()]
N = len(changes_pruned)




langs = sorted(set([l[0] for l in changes_pruned]))
L = len(langs)


lang_ind = np.zeros([N,L])
for i,l in enumerate(changes_pruned):
    lang_ind[(i,langs.index(l[0]))] = 1
    

sound_ind = np.zeros([N,S])
for i,l in enumerate(changes_pruned):
    for s in l[1]:
        sound_ind[(i,change_list.index(s))] += 1


#lang_counts={langs[i]:np.sum(lang_ind,axis=0)[i] for i in range(L)}
#[k for k in lang_counts.keys() if lang_counts[k] <= 5]


partition = [[0,R[0]]]+[[reduce(lambda x,y:x+y,R[:i]),reduce(lambda x,y:x+y,R[:i+1])] for i in range(1,len(R))]



f = open('sound_changes.pkl','wb')
pkl.dump((change_list,S,X,R,N,L,partition,lang_ind,sound_ind,langs),f)
f.close()
