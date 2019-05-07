import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax


np.random.seed(0)

rho = .01
Sigma = np.eye(4)*10
Sigma[0,2] = Sigma[2,0] = Sigma[1,3] = Sigma[3,1] = 8
phis = []
for i in range(10000):
    psi = np.random.multivariate_normal([0]*4,Sigma)
    phi = np.concatenate([softmax([psi[:2]])[0],softmax([psi[2:]])[0]])
    phis.append(phi)
    



phis = np.array(phis)    
g = sns.JointGrid(phis[:,0],phis[:,2])
g.plot_joint(sns.kdeplot, clip = (0.0, 1.0), shade=True)
g.ax_marg_x.set_axis_off()
g.ax_marg_y.set_axis_off()


plt.savefig('output/LNsamp.pdf')
plt.clf()


phis = []
for i in range(10000):
    phis.append(np.concatenate([np.random.dirichlet([.01,.01]),np.random.dirichlet([.01,.01])]))
    
    
phis=np.array(phis)
g = sns.JointGrid(phis[:,0],phis[:,2])
g.plot_joint(sns.kdeplot, clip = (0.0, 1.0), shade=True)
g.ax_marg_x.set_axis_off()
g.ax_marg_y.set_axis_off()

plt.savefig('output/dirsamp.pdf')
plt.clf()