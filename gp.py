import numpy as np
import scipy.linalg as la

EPSILON = 1e-10 

def Ky_and_derivs (diff_z, diff_t, param, fix):
    # the param here is the exponential of actual param
    [sigma_s2, l_z, l_t, sigma_y2] = param
    Ky = sigma_s2 * np.exp(-0.5 * ((diff_z / l_z)**2 + (diff_t / l_t)**2))
    dkdl_z = np.multiply(Ky, diff_z**2/(l_z**3)) * l_z
    dkdl_t = np.multiply(Ky, diff_t**2/(l_t**3)) * l_t
    dkdsigma_s = Ky.copy() 
    ny = Ky.shape[0]
    dkdsigma_y = np.eye(Ky.shape[0]) * sigma_y2
    
    if fix is None:
        fix = np.full(ny, 0)
    
    for i in range(ny):
        Ky[i, i] += EPSILON 
        if fix[i] == 0:
            Ky[i, i] += sigma_y2
        else:
            dkdsigma_y[i, i] = 0

    return Ky, np.array([dkdsigma_s, dkdl_z, dkdl_t, dkdsigma_y])

def log_like_and_derivs(y, mu, K, gradient):
    C, low = la.cho_factor(K)
    z = y - mu
    u = la.cho_solve((C, low), z)
    n = len(y)
    log_like = -0.5 * np.dot(z, u) - np.sum(np.log(np.diag(C))) - 0.5 * n * np.log(2 * np.pi)
    K_inv = la.cho_solve((C, low), np.eye(n))
    d_param = np.array([])
    for g in gradient:
        d = 0.5 * np.dot(u, g @ u) - 0.5 * np.trace( K_inv @ g)
        d_param = np.append(d_param, d)
    return log_like, d_param

def Ky_and_derivs_arg (diff_z, diff_t, param, y, prior_mean, fix):
    z = y - prior_mean
    [sigma_s2, l_z, l_t, sigma_y2] = param
    Ky = sigma_s2 * np.exp(-0.5 * ((diff_z / l_z)**2 + (diff_t / l_t)**2))
    dkdz = np.multiply(Ky, diff_z) / (l_z ** 2)
    dkdt = np.multiply(Ky, diff_t) / (l_t ** 2)
    d2kdz2 = np.multiply(Ky, ((diff_z**2)/(l_z**2)-1)) / (l_z ** 2)
    d2kdt2 = np.multiply(Ky, ((diff_t**2)/(l_t**2)-1)) / (l_t ** 2)
    K = Ky.copy() 

    ny = Ky.shape[0]
    if fix is None:
        fix = np.full(ny, 0) 
    
    for i in range(ny):
        Ky[i, i] += EPSILON 
        if fix[i] == 0:
            Ky[i, i] += sigma_y2
            
    C, low = la.cho_factor(Ky)
    u = la.cho_solve((C, low), z)
    dydz = np.transpose(dkdz) @ u 
    dydt = np.transpose(dkdt) @ u 
    d2ydz2 = np.transpose(d2kdz2) @ u 
    d2ydt2 = np.transpose(d2kdt2) @ u 
    post_mean = np.transpose(K) @ u + prior_mean  
    
    return post_mean, np.array([dydz, dydt, d2ydz2, d2ydt2])



