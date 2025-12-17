print("Importing packages...")
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.linalg as la

sys.path.insert(1, 'C:/Users/31676/Desktop/research/pde/Python code/GPPDE_v0/src')
from hydro import *
from gp import *
from funcs import *
from richards import picard_solver
from viz import plot_3d
print("Importing packages...Done")

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.inf)

thta = np.loadtxt('thta_sink.txt')
print("theta dimension", thta.shape)

lm_est = np.zeros((10,1))
beta_est = np.zeros((10,1))
para_est = np.zeros((10, 2))

for q in range(10):
    dz = -0.025
    initi = np.loadtxt('init.txt', skiprows = 1)
    z = - np.copy(initi[:, 0])
    
    dt = 86400
    t = np.arange(dt, dt * 112, dt)
    
    noise_pct = 0.01
    X = np.dstack(np.meshgrid(z, t)).reshape(-1, 2)
    y = thta.T.flatten()
    y += noise_pct * np.std(y) * np.random.randn(len(y))
    
    # extract some data points 
    # ny = len(y)
    # ind = np.random.choice(range(0, ny), 1300, replace = False)
    # # ind = np.arange(0, ny, 4)
    # y = y[ind]
    # X = X[ind, :]
    
    h_para = np.loadtxt('para.txt', skiprows = 1)
    
    ny = len(y)
    bound_ind = np.array([])
    #skip = int(sys.argv[1])
    # skip = 0.05
    # bound = np.append(0.0, h_para[:,0])
    # nl = len(bound)
    # for i in range(ny):
    #     for j in range(nl):
    #         if (abs(X[i, 0]) <= bound[j] + skip) and (abs(X[i, 0]) >= bound[j] - skip) or (X[i, 1] <= 86400 * 4):
    #             bound_ind = np.append(bound_ind, i)
    
    for i in range(ny):
        if(abs(X[i, 0]) <= 0.45) and (abs(X[i, 0]) >= 0.32) and (X[i, 1] >= 86400 * 5):
            bound_ind = np.append(bound_ind, i)
        if(abs(X[i, 0]) <= 0.58) and (abs(X[i, 0]) >= 0.46) and (X[i, 1] >= 86400 * 5):
            bound_ind = np.append(bound_ind, i)
        # if(abs(X[i, 0]) <= 0.25) and (abs(X[i, 0]) >= 0.16) and (X[i, 1] >= 86400 * 5):
        #     bound_ind = np.append(bound_ind, i)
    
    bad = bound_ind.astype(int)
    # X = np.delete(X, bad, 0)
    # y = np.delete(y, bad)
    
    X = X[bad, :]
    y = y[bad]
        
    print("X shape", X.shape)
    print("y length", len(y))
    
    ny = len(y)
    fix_boundary = np.full(ny, 0)
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    prior_mean = np.full(ny, y_mean)
    
    diff_z = np.subtract.outer(X[:,0], X[:,0])
    diff_t = np.subtract.outer(X[:,1], X[:,1])
    
    # hyperparameter estimation
    def obj_func(theta):
        param = np.exp(theta)
        Ky, gradient = Ky_and_derivs(diff_z, diff_t, param, fix_boundary)
        log_like, d_param = log_like_and_derivs(y, prior_mean, Ky, gradient)
        return -log_like, -d_param
    
    # initial value
    l_z = np.std(X[:, 0])
    l_t = np.std(X[:, 1])
    sigma_s2 = np.std(y) / np.sqrt(2)
    sigma_y2 = np.std(y) / np.sqrt(2)
    
    param = np.array([sigma_s2, l_z, l_t, sigma_y2])
    param_range = np.array([
            [1e-4, 1],
            [0.01, 100],
            [0.01, 10000000],
            [1e-9, 1]
        ])
    
    log_like, d_param = obj_func(np.log(param))
    print("initial log_like =", log_like)
    
    if 1:
        opt_res = minimize(obj_func, 
            np.log(param), 
            method="L-BFGS-B", 
            jac=True, 
            bounds=np.log(param_range)
            # ,callback=show_progress
        )
        param_opt = np.exp(opt_res.x)
        func_max = -opt_res.fun
        print("opt para =", param_opt)
        print("obj =", func_max)
        param = param_opt.copy()
        
    post_mean, dys = Ky_and_derivs_arg(diff_z, diff_t, param, y, prior_mean, fix_boundary)
    [dydz, dydt, d2ydz2, d2ydt2] = dys
    diff_y = y - post_mean
    print("post mean dimension", post_mean.shape)
    print("y diff =", np.max(np.abs(diff_y)))
    
    # estimate sink parameter
    nz = len(z)
    alpha = np.zeros(nz)
    theta_r = np.zeros(nz)
    theta_s = np.zeros(nz)
    m = np.zeros(nz)
    k_s = np.zeros(nz)
    
    cur = 0
    for i in range(nz):
        if z[i] < -h_para[cur, 0]:
            cur = cur + 1
        theta_r[i] = h_para[cur, 1]
        theta_s[i] = h_para[cur, 2]
        alpha[i] = h_para[cur, 3]
        m[i] = h_para[cur, 4]
        k_s[i] = h_para[cur, 5]
    
    nt = len(t)
    theta_r = np.tile(theta_r, 111)
    # theta_r = theta_r[ind]
    # theta_r = np.delete(theta_r, bad)
    theta_r = theta_r[bad]
    
    theta_s = np.tile(theta_s, 111)
    # theta_s = theta_s[ind]
    # theta_s = np.delete(theta_s, bad)
    theta_s = theta_s[bad]
    
    alpha = np.tile(alpha, 111)
    # alpha = alpha[ind]
    # alpha = np.delete(alpha, bad)
    alpha = alpha[bad]
    
    m = np.tile(m, 111)
    # m = m[ind]
    # m = np.delete(m, bad)
    m = m[bad]
    
    k_s = np.tile(k_s, 111)
    # k_s = k_s[ind]
    # k_s = np.delete(k_s, bad)
    k_s = k_s[bad]
        
    def alpha_func(H, a1, a2, a3, a4):
        if H < a4 or H >= a1:
            return 0.0
        if H >= a4 and H < a3:
            return (H - a4)/(a3 - a4)
        if H >= a3 and H < a2:
            return 1.0
        if H >= a2 and H < a1:
            return (a1 - H)/(a1 - a2)
            
    def sink(z, t, h, sink_para, beta, lm) :
        nz = len(z)
        tp = sink_para[:, 0]
        zr = sink_para[:, 1]
        a1 = sink_para[:, 2]
        a2 = sink_para[:, 3]
        a3 = sink_para[:, 4]
        a4 = sink_para[:, 5]
        sink_term = np.zeros(nz)
        for i in range(nz):
            index = int(t[i]/86400 - 1)
            if (abs(z[i]) <= lm[0] * zr[index]):
                alph = alpha_func(h[i] , a1[index], a2[index], a3[index], a4[index])
                fir = (1.0 + beta[0]) * tp[index] / (lm[0] *zr[index]) 
                sink_term[i] = fir * alph * pow(1.0 - abs(z[i]) / (lm[0] * zr[index]), beta[0])
        return sink_term
            
    cov = np.loadtxt('cov.txt', skiprows = 1)
    sink_para = cov[:, 4:10]
    
    ######## FIT #########
    h = np.vectorize(h_)(post_mean, theta_r, theta_s, 1/(1-m), m, alpha)
    s_obs = np.vectorize(S_)(post_mean, theta_r, theta_s)
    X_z = X[:, 0]
    X_t = X[:, 1]
    K = np.vectorize(k_)(h, s_obs, k_s, m)
    resid = dydt - (term1(dydz, s_obs, k_s, h, alpha, m, theta_r, theta_s) + K * term2(dydz, d2ydz2, s_obs, h, alpha, m, theta_r, theta_s) + term3(dydz, s_obs, k_s, m, theta_r, theta_s))
    # print("resid=",resid)
    beta = np.array([2.0])
    lm = np.array([1.35])
    
    def sse(lm,beta,  resid, X_z, X_t, h, sink_para,w):
        residfull = resid + sink(X_z, X_t, h, sink_para, beta, lm)
        # w = lm_weight(X_z, X_t, h, sink_para, beta, lm)
        w = w/ np.sum(w)
        return(np.sum(w * residfull ** 2))
    w = np.full(len(resid), 1.0)
    opt_ini = minimize(fun=sse, x0 =np.array([2.0]) ,bounds =[(0.0,None)],  args= (beta, resid, X_z, X_t, h, sink_para,w),method = 'Nelder-Mead') 
    lm_gp = opt_ini.x
    lm_est[q,:] = lm_gp
    
    def sse1(beta,lm,  resid, X_z, X_t, h, sink_para,w):
        residfull = resid + sink(X_z, X_t, h, sink_para, beta, lm)
        # w = beta_weight(X_z, X_t, h, sink_para, beta, lm)
        w = w/ np.sum(w)
        return(np.sum(w * residfull ** 2))
    
    opt_beta = minimize(fun=sse1, x0 =np.array([2.5]) ,bounds =[(0.0,None)],  args= (lm, resid, X_z, X_t, h, sink_para,w),method = 'Nelder-Mead') 
    beta_gp = opt_beta.x
    beta_est[q,:] = beta_gp
    
    def ssre(par_est,  resid, X_z, X_t, h, sink_para,w):
        [beta, lm] = par_est
        beta = np.array([beta])
        lm = np.array([lm])
        residfull = resid + sink(X_z, X_t, h, sink_para, beta, lm)
        # w = beta_weight(X_z, X_t, h, sink_para, beta, lm)
        w = w/ np.sum(w)
        return(np.sum(w * residfull ** 2))
    
    opt_para = minimize(fun=ssre, x0 =np.array([1.5,1.0]) ,bounds =[(0.0,None),(0.0,None)],  args= (resid, X_z, X_t, h, sink_para,w),method = 'Nelder-Mead') 
    para_est[q,] = opt_para.x
print("lm", lm_est)
print("beta", beta_est)
print("para",para_est)

