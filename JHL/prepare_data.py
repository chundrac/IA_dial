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



nondard = ['bhat1263',
           'chur1258',
           'marw1260',
           'west2386',
           'dhiv1236',
           'kach1277', 
           #'mand1409',
           'bhoj1244',
           'assa1263',
           'bagh1251',
           'khet1238',
           'beng1280', 
           'panj1256',
           'dogr1250',
           'jaun1243',
           'garh1243',
           #'sirm1239',
           'kuma1273', 
           'sind1272',
           #'halb1244',
           'oriy1255',
           'kang1280',
           'maha1287',
           'paha1251', 
           'awad1243',
           'braj1242',
           'hind1269',
           'cham1307',
           'mara1378',
           'mait1250', 
           'nepa1254',
           'konk1267',
           #'sera1259',
           'bhad1241',
           'kull1236',
           #'malv1243', 
           #'maga1260',
           'pang1282',
           'sinh1246',
           'guja1252']



    
change_counts = defaultdict(int)
changes = []
for i in range(len(text)):
    if text[i][0] in nondard:
        lang = text[i][0]
        word_changes = []
        x,y=text[i][2],text[i][1]
        A = alignments[i]
        for j in range(1,len(A)-2):
            edit = (tuple(x[j-1:j+2]),tuple(y[A[j]:A[j+1]]))
            word_changes.append(edit)
            change_counts[edit]+=1
        changes.append([lang,word_changes])



Cutoff = 1


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
#'V',
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



feat_mat = {}
for line in open('feat_matrix.csv','r'):
    l = line.split('\t')
    feat_mat[l[0]]=tuple(l[1:])


feat_mat['#'] = ('-','-','-','-','-')

    
K = 2
S = len(change_list)
X = len(reflex.keys())
R = [len(reflex[k]) for k in reflex.keys()]
N = len(changes_pruned)




langs = sorted(set([l[0] for l in changes_pruned]))
L = len(langs)


lang_ind = np.zeros([N,L])
for i,l in enumerate(changes_pruned):
    lang_ind[(i,langs.index(l[0]))] = 1


lang_key = {l.strip().split('\t')[1]:l.strip().split('\t')[0] for l in open('language_key_for_pub.txt','r')}
    

sound_ind = np.zeros([N,S])
for i,l in enumerate(changes_pruned):
    for s in l[1]:
        sound_ind[(i,change_list.index(s))] += 1


#lang_counts={langs[i]:np.sum(lang_ind,axis=0)[i] for i in range(L)}
#[k for k in lang_counts.keys() if lang_counts[k] <= 5]


s_breaks = [[0,R[0]]]+[[reduce(lambda x,y:x+y,R[:i]),reduce(lambda x,y:x+y,R[:i+1])] for i in range(1,len(R))]


break_rev = {j:i for i,b in enumerate(s_breaks) for j in range(b[0],b[1])}

disp = 4

D = 5 #list(set([len(v) for v in feat_mat.values()]))[0]

Alpha = np.zeros([S,S])
Sigma = np.zeros([S,S])
for i in range(S):
    Alpha[i,i] = disp
    for j in range(i+1,S):
        change_i = [change_list[i][0][1]]+list(change_list[i][1])
        change_j = [change_list[j][0][1]]+list(change_list[j][1])
        env_i = [change_list[i][0][0]]+[change_list[i][0][2]]
        env_j = [change_list[j][0][0]]+[change_list[j][0][2]]
        for d in range(D):
            if [feat_mat[s][d] for s in change_i] != [feat_mat[s][d] for s in change_j]:
                Sigma[i,j] += 1
                Sigma[j,i] += 1
            if [feat_mat[s][d] for s in env_i] != [feat_mat[s][d] for s in env_j]:
                Sigma[i,j] += 1
                Sigma[j,i] += 1
        if break_rev[i] != break_rev[j]:
            Alpha[i,j] = disp
            Alpha[j,i] = disp

            
Sigma = Alpha*np.exp(-Sigma) + np.eye(S)*100



"""
Sigma = np.zeros([S,S])
for i in range(S):
    Sigma[i,i] = disp
    for j in range(i+1,S):
        S1 = similarity[(change_list[i][0][1],change_list[j][0][1])] #fuck, fix this
        S2 = similarity[(change_list[i][0][0],change_list[j][0][0])]
        S3 = similarity[(change_list[i][0][2],change_list[j][0][2])]
        if len(change_list[i][1]) > 0 and len(change_list[j][1]) > 0:
            S4 = np.mean([similarity[(r,s)] for r in change_list[i][1] for s in change_list[j][1]])
        else:
            S4 = 1/25
        kernel = .8*S1*S4 + .2*S2*S3
        if break_rev[i] != break_rev[j]:
            kernel *= disp
#        kernel *= disp
        Sigma[i,j] = kernel
        Sigma[j,i] = kernel


Sigma += np.eye(S)*8
"""

#values,vectors=np.linalg.eig(Sigma)
#values[values<0]=1e-4
#Q=vectors
#L=np.diag(values)
#R=np.linalg.inv(Q)
#Sigma=Q.dot(L).dot(R)



assert(np.all(np.linalg.eigvals(Sigma) > 0)) #is matrix positive definite?


nchains = 4

data_and_variables = (segs_to_keep,changes_pruned,Cutoff,change_list,reflex,Sigma,K,S,X,R,N,langs,L,lang_ind,sound_ind,s_breaks,nchains)

f = open('data_variables.pkl','wb')
pkl.dump(data_and_variables,f)
f.close()




if 'output' not in os.listdir('.'):
    os.makedirs('output/')

    

f = open('output/lang_summary.tex','w')
for l in range(L):
    print (langs[l],'&',lang_key[langs[l]],'&',int(np.sum(lang_ind,axis=0)[l]),'\\\\',file=f)
    print ('\\hline',file=f)


f.close()


f = open('output/variables.tex','w')

#segs_to_keep[-1] = '} and {\\IPA '+segs_to_keep[-1]


print("%<*nchains>\n"+str(nchains)+"%</nchains>\n",file=f)
print("%<*Cutoff>\n"+str(Cutoff)+"%</Cutoff>\n", file=f)
print("%<*S>\n"+str(S)+"%</S>\n", file=f)
print("%<*X>\n"+str(X)+"%</X>\n", file=f)
print("%<*N>\n"+str(N)+"%</N>\n", file=f)
print("%<*L>\n"+str(L)+"%</L>\n", file=f)
print("%<*SegsToKeep>\n"+', '.join(segs_to_keep)+'\n'+"%</SegsToKeep>\n", file=f)


f.close()
