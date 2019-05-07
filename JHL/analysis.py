from __future__ import division
import os
import codecs
import numpy as np
import random
import itertools
from collections import defaultdict
from numpy.random import multinomial,normal,uniform,multivariate_normal
from numpy import log,exp,mean
from scipy.special import gamma
import re
import pymc3 as pm
from pymc3.math import logsumexp
from sklearn.utils.extmath import softmax
from functools import reduce
import theano.tensor as tt
import theano
theano.config.gcc.cxxflags = "-fbracket-depth=6000"
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from scipy.stats import entropy
import pickle as pkl
from statsmodels.stats.weightstats import ztest


f = open('data_variables.pkl','rb')
segs_to_keep,changes_pruned,Cutoff,change_list,reflex,Sigma,K,S,X,R,N,langs,L,lang_ind,sound_ind,s_breaks,nchains = pkl.load(f)
f.close()
assert(nchains==4)


f = open('posterior_ln_full.pkl','rb')
posterior_ln = pkl.load(f)
f.close()


f = open('posterior_dir_full.pkl','rb')
posterior_dir = pkl.load(f)
f.close()


posterior_ln_shuffled = []
for c in range(4):
    f = open('posterior_ln_shuffle_{}.pkl'.format(c),'rb')
    posterior_ln_shuffled.append(pkl.load(f))
    f.close()


posterior_dir_shuffled = []
for c in range(4):
    f = open('posterior_dir_shuffle_{}.pkl'.format(c),'rb')
    posterior_dir_shuffled.append(pkl.load(f))
    f.close()


#posterior_ln = defaultdict()
#posterior_dir = defaultdict()
#for c in range(nchains):
#    posterior_ln[c] = defaultdict(list)
#    f = open('posterior_ln_{}.pkl'.format(c),'rb')
#    posterior_ln[c] = pkl.load(f)
#    f.close()
#    posterior_dir[c] = defaultdict(list)
#    f = open('posterior_dir_{}.pkl'.format(c),'rb')
#    posterior_dir[c] = pkl.load(f)
#    f.close()



#make visuals

for i in posterior_ln.keys():
#    plt.plot(range(0,50000,50),posterior_ln[i]['ELBO'][::50],color='#1f77b4',alpha=.4)
    plt.plot(posterior_ln[i]['ELBO'],color='#1f77b4',alpha=.4,rasterized=True)


#tikz_save('output/ELBO_ln.tex')
plt.savefig('output/ELBO_ln.pdf')
plt.clf()





for i in posterior_dir.keys():
#    plt.plot(range(0,50000,50),posterior_dir[i]['ELBO'][::50],color='#1f77b4',alpha=.4)
    plt.plot(posterior_dir[i]['ELBO'],color='#1f77b4',alpha=.4,rasterized=True)


#tikz_save('output/ELBO_dir.tex')
plt.savefig('output/ELBO_dir.pdf')
plt.clf()


#fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
fig, axes = plt.subplots(2, 2)
#axes.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
#axes[0,1].axvline(x=1,linestyle=':')
#axes[1,1].axvline(x=1,linestyle=':')
for c in range(nchains):
#    axes[0,1].axvline(x=1,ymax=200)
    axes[0,0].hist(posterior_dir[c]['beta'],color='#1f77b4',alpha=.4,histtype='stepfilled',edgecolor='black')
#    axes[0,1].axvline(x=1,ymax=200)
    axes[1,0].hist(posterior_ln[c]['beta'],color='#1f77b4',alpha=.4,histtype='stepfilled',edgecolor='black')
    axes[0,1].axvline(x=1,ymax=200)#,linestyle=':')
    axes[0,1].hist(posterior_dir_shuffled[c]['beta'],color='#ff7f0e',alpha=.4,histtype='stepfilled',edgecolor='black')
    axes[1,1].axvline(x=1,ymax=200)#,linestyle=':')
    axes[1,1].hist(posterior_ln_shuffled[c]['beta'],color='#ff7f0e',alpha=.4,histtype='stepfilled',edgecolor='black')


#for i in range(2):
#    for j in range(2):
#        axes[i,j].ticklabel_format(axis='x',style='plain')



#tikz_save('output/betas.tex')
plt.savefig('output/betas.pdf')
plt.clf()


#for i in posterior_ln.keys():
#    plt.hist(posterior_ln[i]['beta'],color='#1f77b4',alpha=.4)


#tikz_save('output/beta_ln.tex')
#plt.clf()


#for i in posterior_ln.keys():
#    plt.hist(posterior_ln_shuffled[i]['beta'],color='#ff7f0e',alpha=.4)



#tikz_save('output/beta_ln_shuffled.tex')
#plt.clf()


#for i in posterior_dir.keys():
#    plt.hist(posterior_dir[i]['beta'],color='#1f77b4',alpha=.4)
    


#tikz_save('output/beta_dir.tex')
#plt.clf()


#for i in posterior_dir.keys():
#    plt.hist(posterior_dir_shuffled[i]['beta'],color='#ff7f0e',alpha=.4)



#tikz_save('output/beta_dir_shuffled.tex')
#plt.clf()


# monitor convergence

thetas = [key for key in posterior_ln[0].keys() if key.startswith('theta')]
phis = [key for key in posterior_ln[0].keys() if key.startswith('phi')]


R_beta_ln = pm.gelman_rubin(np.array([posterior_ln[i]['beta'] for i in posterior_ln.keys()]))
R_beta_dir = pm.gelman_rubin(np.array([posterior_dir[i]['beta'] for i in posterior_dir.keys()]))
R_theta_ln = {(param,k):pm.gelman_rubin(np.array([posterior_ln[c][param][:,k] for c in posterior_ln.keys()])) for param in thetas for k in range(K)}
R_theta_dir = {(param,k):pm.gelman_rubin(np.array([posterior_dir[c][param][:,k] for c in posterior_dir.keys()])) for param in thetas for k in range(K)}
R_phi_ln = {(param,x):pm.gelman_rubin(np.array([posterior_ln[c][param][:,x] for c in posterior_ln.keys()])) for param in phis for x in range(posterior_ln[0][param].shape[1])}
R_phi_dir = {(param,x):pm.gelman_rubin(np.array([posterior_dir[c][param][:,x] for c in posterior_dir.keys()])) for param in phis for x in range(posterior_dir[0][param].shape[1])}


#z tests

z_test_Dir = [ztest(np.concatenate([posterior_dir[c]['beta'] for c in range(nchains)]),posterior_dir_shuffled[c]['beta'],alternative='smaller') for c in range(nchains)]
z_test_Ln = [ztest(np.concatenate([posterior_ln[c]['beta'] for c in range(nchains)]),posterior_ln_shuffled[c]['beta'],alternative='smaller') for c in range(nchains)]

zDir = '$z={:.5},{:.5},{:.5},{:.5}$'.format(z_test_Dir[0][0],z_test_Dir[1][0],z_test_Dir[2][0],z_test_Dir[3][0])
pDir = '$p<1e-10$'
#if z_test_Dir[1] < .001:
#    pDir = '$p<1e-10$'
#else:
#    pDir = '$p={}$'.format(z_test_Dir[1])



zLn = '$z={:.5},{:.5},{:.5},{:.5}$'.format(z_test_Ln[0][0],z_test_Ln[1][0],z_test_Ln[2][0],z_test_Ln[3][0])
pLn = '$p<1e-10$'
#if z_test_Ln[1] < .001:
#    pLn = '$p<1e-10$'
#else:
#    pLn = '$p={}$'.format(z_test_Ln[1])
    

#z_test_Dir1 = [ztest(posterior_dir[c]['beta'],value=1,alternative='smaller') for c in range(nchains)]
#z_test_Ln1 = [ztest(posterior_ln[c]['beta'],value=1,alternative='smaller') for c in range(nchains)]    
#z_test_Dir1_shuf = [ztest(posterior_dir_shuffled[c]['beta'],value=1,alternative='smaller') for c in range(nchains)]
#z_test_Ln1_shuf = [ztest(posterior_ln_shuffled[c]['beta'],value=1,alternative='smaller') for c in range(nchains)]

#z_shuf_1 = '$Z={:.2},p={};Z={:.2},p={}$'.format(str(set([z[0] for z in z_test_Dir1_shuf]))[1:-1],
#                                                str(set([z[1] for z in z_test_Dir1_shuf]))[1:-1],
#                                                str(set([z[0] for z in z_test_Ln1_shuf]))[1:-1],
#                                                str(set([z[1] for z in z_test_Ln1_shuf]))[1:-1])

#z_1 = '$Z={:.2},p<1e-10;Z={:.2},p<1e-10$'.format(str(set([z[0] for z in z_test_Dir1_shuf]))[1:-1],
#                                                 str(set([z[0] for z in z_test_Ln1_shuf]))[1:-1])

f = open('output/metrics.tex','w')


print("%<*zDir>\n"+zDir+"\n%</zDir>\n", file=f)
print("%<*zLn>\n"+zLn+"\n%</zLn>\n", file=f)
#print("%<*zShuf1>\n"+z_shuf_1+"\n%</zShuf1>\n", file=f)
#print("%<*z1>\n"+z_1+"\n%</z1>\n", file=f)
print("%<*pDir>\n"+pDir+"\n%</pDir>\n", file=f)
print("%<*pLn>\n"+pLn+"\n%</pLn>\n", file=f)
print("%<*RhatBetaDir>\n"+'%.3f' % R_beta_dir+"\n%</RhatBetaDir>\n", file=f)
print("%<*RhatBetaLn>\n"+'%.3f' % R_beta_ln+"\n%</RhatBetaLn>\n", file=f)
print("%<*RhatThetaDir>\n"+str(len([v for v in list(R_theta_dir.values()) if v < 1.5]))+"/"+str(len(list(R_theta_dir.values())))+"\n%</RhatThetaDir>\n", file=f)
print("%<*RhatThetaLn>\n"+str(len([v for v in list(R_theta_ln.values()) if v < 1.5]))+"/"+str(len(list(R_theta_ln.values())))+"\n%</RhatThetaLn>\n", file=f)
print("%<*RhatPhiDir>\n"+str(len([v for v in list(R_phi_dir.values()) if v < 1.5]))+"/"+str(len(list(R_phi_dir.values())))+"\n%</RhatPhiDir>\n", file=f)
print("%<*RhatPhiLn>\n"+str(len([v for v in list(R_phi_ln.values()) if v < 1.5]))+"/"+str(len(list(R_phi_ln.values())))+"\n%</RhatPhiLn>\n", file=f)

f.close()


# give sound change probabilities according to jensen-shannon divergence between group 0 and group 1 distributions

JS_dir = {}
for i in range(X):
    k = list(reflex.keys())[i]
    param0 = 'phi_0_'+str(i)
    param1 = 'phi_1_'+str(i)
    P = np.array([np.mean([np.mean(posterior_dir[c][param0][:,j]) for c in range(nchains)]) for j in range(R[i])])
    Q = np.array([np.mean([np.mean(posterior_dir[c][param1][:,j]) for c in range(nchains)]) for j in range(R[i])])
    M = .5*(P+Q)
    JS_dir[k] = 0.5 * (entropy(P, M) + entropy(Q, M))




JS_ln = {}
for i in range(X):
    k = list(reflex.keys())[i]
    param0 = 'phi_0_'+str(i)
    param1 = 'phi_1_'+str(i)
    P = np.array([np.mean([np.mean(posterior_ln[c][param0][:,j]) for c in range(nchains)]) for j in range(R[i])])
    Q = np.array([np.mean([np.mean(posterior_ln[c][param1][:,j]) for c in range(nchains)]) for j in range(R[i])])
    M = .5*(P+Q)
    JS_ln[k] = 0.5 * (entropy(P, M) + entropy(Q, M))


f = open('output/sound_changes_dir.tex','w')
print('\\hline',file=f)
for k in sorted(JS_dir.keys(),key=lambda x:JS_dir[x]):
    i = list(reflex.keys()).index(k)
    for j in range(R[i]):
        param0 = 'phi_0_'+str(i)
        param1 = 'phi_1_'+str(i)
        print ('{\\IPA ',k[1],'$>$',end=' ',file=f)
        if reflex[k][j] != ():
            print(reflex[k][j][0],end=' ',file=f)
        else:
            print('$\emptyset$',end=' ',file=f)
        print ('/',k[0].replace('#','$\#$'),'\\underline{\\phantom{X}}',k[2].replace('#','$\#$'),'}',end=' ',file=f)
        for c in range(nchains):
            print('&','%.4f' % np.mean(posterior_dir[c][param0][:,j]),'&','%.4f' % np.mean(posterior_dir[c][param1][:,j]),end=' ',file=f)
        print ('\\\\',file=f)
    print('\\hline',file=f)



f.close()



f = open('output/sound_changes_ln.tex','w')
print('\\hline',file=f)
for k in sorted(JS_ln.keys(),key=lambda x: JS_ln[x]):
    i = list(reflex.keys()).index(k)
    for j in range(R[i]):
        param0 = 'phi_0_'+str(i)
        param1 = 'phi_1_'+str(i)
        print ('{\\IPA ',k[1],'$>$',end=' ',file=f)
        if reflex[k][j] != ():
            print(reflex[k][j][0],end=' ',file=f)
        else:
            print('$\emptyset$',end=' ',file=f)
        print ('/',k[0].replace('#','$\#$'),'\\underline{\\phantom{X}}',k[2].replace('#','$\#$'),'}',end=' ',file=f)
        for c in range(nchains):
            print('&','%.4f' % np.mean(posterior_ln[c][param0][:,j]),'&','%.4f' % np.mean(posterior_ln[c][param1][:,j]),end=' ',file=f)
        print ('\\\\',file=f)
    print('\\hline',file=f)



f.close()



#make the component maps

coords = []
for l in open('glottolog_IE.csv','r'):
    coords.append(l.strip().split('\t'))


lat = {}
lon = {}


for l in coords:
    if l[0] in langs:
        lat[l[0]]=float(l[-1])

        

for l in coords:
    if l[0] in langs:
        lon[l[0]]=float(l[-2])



component_dir = {}
for i,l in enumerate(langs):
    #component_dir[l] = np.mean([np.mean(posterior_dir[c]['theta_'+str(i)],axis=0)[0] for c in range(1)])
    component_dir[l] = np.mean(np.concatenate([posterior_dir[c]['theta_'+str(i)][:,0] for c in range(nchains)]))



component_ln = {}
for i,l in enumerate(langs):
    #component_ln[l] = np.mean([np.mean(posterior_ln[c]['theta_'+str(i)],axis=0)[0] for c in range(nchains)])
    component_ln[l] = np.mean(np.concatenate([posterior_ln[c]['theta_'+str(i)][:,0] for c in range(nchains)]))


f = open('output/mapcoords.csv','w')
for l in langs:
    print (l,lon[l],lat[l],component_dir[l],component_ln[l],file=f)


f.close()
