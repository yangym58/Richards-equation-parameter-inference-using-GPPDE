#!/usr/bin/env python
# coding: utf-8

####### packages #######
print("Importing packages...")
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.linalg as la

sys.path.insert(1, os.path.join(sys.path[0], 'src'))
from hydro import *
from gp import *
from funcs import *
from richards import picard_solver
from viz import plot_3d
print("Importing packages...Done")

# random seeds and printing 
# np.random.seed(0)
np.set_printoptions(precision=4)

# Hydrological constants
theta_r = 0.075
theta_s = 0.287
k_s = 1.5

# Van Genuchten model parameters
alpha = 0.05
m = 0.5

# Richards equation resolution parameters
h_t0 = -65
h_z0 = -65
h_z60 = -20.7
z_max = 60
t_max = 50
thta = np.loadtxt('theta.txt')


### extract a sub-matrix as the data 
dz = 1
dt = 1
z = np.arange(0, z_max + dz, dz)
nz = len(z)
t = np.arange(0, t_max + dt, dt)
nt = len(t)
gap_z = int(dz/0.1+1e-6)
gap_t = int(dt/0.1+1e-6)
z_sel = np.arange(0, thta.shape[0],  gap_z)
t_sel = np.arange(0, thta.shape[1],  gap_t)
thta = thta[np.ix_(z_sel, t_sel)]
print("thta dimension:", thta.shape)

noise_pct = 0.01
X = np.dstack(np.meshgrid(z, t)).reshape(-1, 2)
y = thta.T.flatten()
y += noise_pct * np.std(y) * np.random.randn(len(y))
ny = len(y)

bound_ind = np.array([])
#skip = int(sys.argv[1])
skip = 3
for i in range(ny):
    if (X[i, 0] <= skip) or (X[i, 0] >= z_max-1-skip) or (X[i, 1] <= skip):
        bound_ind = np.append(bound_ind, i)

bad = bound_ind.astype(int)
X = np.delete(X, bad, 0)
y = np.delete(y, bad)
ny = len(y)
print("sample size", ny)
fix_boundary = np.full(ny, 0)
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
prior_mean = np.full(ny, y_mean)

diff_z = np.subtract.outer(X[:,0], X[:,0])
diff_t = np.subtract.outer(X[:,1], X[:,1])

def obj_func(theta):
    param = np.exp(theta)
    Ky, gradient = Ky_and_derivs(diff_z, diff_t, param, fix_boundary)
    log_like, d_param = log_like_and_derivs(y, prior_mean, Ky, gradient)
    return -log_like, -d_param

l_z = 4.8709
l_t = 3.8568
sigma_s2 = 0.00039125
sigma_y2 = 1.4317e-7

param = np.array([sigma_s2, l_z, l_t, sigma_y2])
param_range = np.array([
        [1e-4, 1],
        [0.01, 100],
        [0.01, 100],
        [1e-9, 1]
    ])

log_like, d_param = obj_func(np.log(param))
print("initial log_like =", log_like)

if 1:
    opt_res = minimize(obj_func, 
        np.log(param), 
        method="L-BFGS-B", 
        jac=True, 
        bounds=np.log(param_range),
        callback=show_progress
    )
    param_opt = np.exp(opt_res.x)
    func_max = -opt_res.fun
    print("opt para =", param_opt)
    print("obj =", func_max)
    param = param_opt.copy()

post_mean, dys = Ky_and_derivs_arg(diff_z, diff_t, param, y, prior_mean, fix_boundary)
[dydz, dydt, d2ydz2, d2ydt2] = dys
diff_y = y - post_mean
print("y diff =", np.max(np.abs(diff_y)))


######## FIT #########
def ssre(pde_param, theta, dthetadt, dthetadz, d2thetadz2, S, k_s, theta_r, theta_s) :
    alpha, m = pde_param
    h = np.vectorize(h_)(theta, theta_r, theta_s, 1/(1-m), m, alpha)
    K = np.vectorize(k_)(h, S, k_s, m)
    resid = dthetadt - (term1(dthetadz, S, k_s, h, alpha, m, theta_r, theta_s) + K * term2(dthetadz, d2thetadz2, S, h, alpha, m, theta_r, theta_s) + term3(dthetadz, S, k_s, m, theta_r, theta_s))
    return(np.sum(resid**2))


par_range = np.array([
        [0.001, 100],
        [0.1, 10],
    ])

s_obs = np.vectorize(S_)(post_mean, theta_r, theta_s)
other_param = (post_mean, dydt, dydz, d2ydz2, s_obs, k_s, theta_r, theta_s)
print("initial SSR =", ssre(np.array([0.05, 0.5]), post_mean, dydt, dydz, d2ydz2, s_obs, k_s, theta_r, theta_s) )
opt_res = minimize(fun=ssre, x0=np.array([0.2, 0.8]), args=other_param, method='Nelder-Mead')
alpha_gp, m_gp = opt_res.x
print("param =", alpha_gp, m_gp)
print("minSSR =", opt_res.fun)



