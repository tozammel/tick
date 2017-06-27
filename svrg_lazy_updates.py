
# coding: utf-8

# In[1]:

import sys

sys.path.append('/Users/stephane.gaiffas/Code/tick')


# In[2]:

from tick.simulation import SimuLinReg, weights_sparse_gauss
from tick.optim.model import ModelLinReg
from tick.plot import stems
from bokeh.plotting import output_notebook
from scipy.linalg import norm
import numpy as np
output_notebook()

n_samples = 2000
n_features = 20


# In[4]:

from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz
from scipy.sparse import csr_matrix


def simu_linreg(x, n, interc=None, std=1., corr=0.5, p_nnz=0.3):
    """
    Simulation of the least-squares problem
    
    Parameters
    ----------
    x : np.ndarray, shape=(d,)
        The coefficients of the model
    
    n : int
        Sample size
    
    std : float, default=1.
        Standard-deviation of the noise

    corr : float, default=0.5
        Correlation of the features matrix
    """    
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    A *= np.random.binomial(1, p_nnz, size=A.shape)
    idx = np.nonzero(A.sum(axis=1))
    A = csr_matrix(A[idx])
    n = A.shape[0]
    noise = std * randn(n)
    b = A.dot(x_truth) + noise
    if interc:
        b += interc
    return A, b

d = 5
n = 10000
idx = np.arange(d)

# Ground truth coefficients of the model
x_truth = (-1) ** (idx - 1) * np.exp(-idx / 10.)

intercept0 = -1.
A_spars, b = simu_linreg(x_truth, n, interc=-1., std=1., corr=0.8)

weights0 = np.append(x_truth, -1.)

n, d = A_spars.shape

A_dense = A_spars.toarray()


from tick.optim.model import ModelLinReg
from tick.optim.solver import SVRG
from tick.optim.prox import ProxL2Sq, ProxZero, ProxTV

#
model_dense = ModelLinReg(fit_intercept=True).fit(A_dense, b)


# prox = ProxZero()
# prox = ProxL2Sq(strength=1e-3)
prox = ProxTV(strength=1e-2)


solver1 = SVRG(step=1e-2, print_every=1, max_iter=10, seed=123) \
    .set_model(model_dense).set_prox(prox)
solver1.solve()


model_spars = ModelLinReg(fit_intercept=True).fit(A_spars, b)
solver2 = SVRG(step=1e-2, print_every=1, max_iter=10, seed=123) \
    .set_model(model_spars).set_prox(prox)

solver2.solve()

print(np.abs(weights0 - solver1.solution).max())
print(np.abs(weights0 - solver2.solution).max())

print(np.abs(solver2.solution - solver2.solution).max())

print(solver1.solution)
print(solver2.solution)
print(weights0)

