import numpy as np
import scipy.linalg as lalg
from cst import *
from hydro import *

# Richards equation solver

def picard_solver(z_max, t_max, dz, dt, param, 
    h_t0=-100, h_z0=-100, h_z_max=-50, J_z0=0, J_z_max=0, 
    downbound="dirichlet", topbound="dirichlet",
    n_iter_max=100, h_err_bound=1e-7):

    """Solves the Richards equation using the modified Picard method for mass conservation
    Args:
        z_max (float) : maximum z value to consider. Must be positive.
        t_max (float) : maximum t value to consider. Must be positive.
        dz (float, optional) : z step, ideally a divider of z_max.
        dt (float, optional) : t step, ideally a divider of t_max.
        h_t0 (float, optional) : value of h(z, t=0)
        downbound (string, optional) : type of boundary condition for z=0. Either "dirichlet", "flux" or "free_drainage".
        h_z0 (float, optional) : value of h(z=0, t) if a Dirichlet boundary condition is set
        J_z0 (float, optional) : value of J(z=0, t) if a flux (Neumann) boundary condition is set
        topbound (string, optional) : type of boundary condition for z=z_max. Either "dirichlet", "flux" or "free_drainage".
        h_z_max (float, optional) : value of h(z=z_max, t) if a Dirichlet boundary condition is set
        timevar (function, optional) : time variation of h(z=z_max, t) to add to h_z_max if a Dirichlet boundary condition is set
        J_z_max (float, optional) : value of J(z=z_max, t) if a flux (Neumann) boundary condition is set
        param (array-like, optional) : hydrological hyperparameters of the Richards equation
        n_iter (int, optional) : number of Picard iterations to run

    Returns:
        z (1D ndarray) : mesh defined by z_max and dz
        t (1D ndarray) : mesh defined by t_max and dt
        h (2D ndarray) : h values taken by the solution over z and t
        thta (2D ndarray) : theta values taken by the solution over z and t
    """
    
    theta_r, theta_s, k_s, alpha, m = param
    n = 1/(1-m)

    Svec = np.vectorize(S_)
    kvec = np.vectorize(k_)
    thvec = np.vectorize(theta_)
    Cvec = np.vectorize(C_)

    z = np.arange(0, z_max + dz, dz)
    n_z = len(z)
    t = np.arange(0, t_max + dt, dt)
    n_t = len(t)

    # Initialization :
    h = np.zeros((n_z, n_t))
 
    # Initial condition t = 0 (h_t0 can be a vector or a scalar):
    # Boundary conditions :
    h[:, 0] = h_t0
    if downbound == "dirichlet" :
        h[0, :] = h_z0
    if topbound == "dirichlet" :
        h[n_z - 1, :] = h_z_max

    
    # Time iteration
    for i in range(1, n_t):
        
        # Initializing h and theta as their final values during the previous time iteration
        h0 = h[:, i - 1]
        thta0 = thvec(h0, theta_r, theta_s, n, m, alpha)
        h[:, i] = np.copy(h0)
        h[n_z - 1, i] = h_z_max

        # Picard iteration
        for j in range(n_iter_max) :
            hm = h[:, i]
            thtam = thvec(hm, theta_r, theta_s, n, m, alpha)
            hdiff = np.convolve(hm, [1, -1], mode='valid')

            s = Svec(thtam, theta_r, theta_s)
            K = kvec(hm, s, k_s, m)

            K05 = np.convolve(K, [1/2, 1/2], mode='valid')
            Kplus05 = K05[1:]
            Kminus05 = K05[:(n_z - 2)]

            c = Cvec(hm, theta_r, theta_s, n, m, alpha)

            # Computing interior tridiagonal matrix coefficients
            A = -Kminus05 / (dz**2)
            B = c[1:(n_z - 1)] / dt + (Kplus05 + Kminus05) / (dz**2)
            C = -Kplus05 / (dz**2)
            D = (Kplus05 * hdiff[1:] - Kminus05 * hdiff[:(n_z - 2)]) / (dz**2) + (Kplus05 - Kminus05) / dz - (thtam - thta0)[1:(n_z - 1)] / dt

            # Setting boundary coefficients
            if downbound == "dirichlet" :
                B0 = 1
                C0 = 0
                D0 = 0
            
            elif downbound == "free_drainage" :
                B0 = c[0] / dt + 2 * K[0] / (dz**2)
                C0 = -2 * K[0] / (dz**2)
                D0 = 2 * K[0] * hdiff[0] / (dz**2) + (K05[0] - K[0]) / dz - (thtam - thta0)[0] / dt
            
            elif downbound == "flux" :
                B0 = c[0] / dt + 2 * K[0] / (dz**2)
                C0 = -2 * K[0] / (dz**2)
                D0 = 2 * K[0] * hdiff[0] / (dz**2) - (J_z0 / K[0] - 2) * (K05[0] - J_z0) / dz - 2 * (K[0] - J_z0) / dz - (thtam - thta0)[0] / dt

            if topbound == "dirichlet" :
                Anz = 0
                Bnz = 1
                Dnz = 0
            
            elif topbound == "free_drainage" :
                Anz = -2 * K[n_z - 1] / (dz**2)
                Bnz = c[n_z - 1] / dt + 2 * K[n_z - 1] / (dz**2)
                Dnz = -2 * K[n_z - 1] * hdiff[n_z - 2] / (dz**2) + (K[n_z - 1] - K05[n_z - 2]) / dz - (thtam - thta0)[n_z - 1] / dt
            
            elif topbound == "flux" :
                Anz = -2 * K[n_z - 1] / (dz**2)
                Bnz = c[n_z - 1] / dt + 2 * K[n_z - 1] / (dz**2)
                Dnz = -2 * K[n_z - 1] * hdiff[n_z - 2] / (dz**2) - (J_z_max / K[n_z - 1] - 2) * (J_z_max - K05[n_z - 2]) / dz + 2 * (K[n_z - 1] - J_z_max) / dz - (thtam - thta0)[n_z - 1] / dt

            A = np.concatenate((A, [Anz]))
            B = np.concatenate(([B0], B, [Bnz]))
            C = np.concatenate(([C0], C))
            D = np.concatenate(([D0], D, [Dnz]))

            # Solving tridiagonal system
            M = np.diag(A, -1) + np.diag(B) + np.diag(C, 1)
            lu, piv = lalg.lu_factor(M)
            delta = lalg.lu_solve((lu, piv), D)
            
            # Updating h
            h[:, i] += delta
            
            if np.max(np.abs(delta)/np.abs(h[:,i])) < h_err_bound:
                hplus = hdiff[1:]
                hminus = hdiff[:(n_z-2)]
                eq = (Kplus05 * (hplus/dz+1.0) - Kminus05 * (hminus/dz+1.0))/dz - (thtam - thta0)[1:(n_z-1)]/dt
                print("time = %.2f" % t[i], "; iterations =", j, "; err = %.2e" % np.max(np.abs(eq)))
                break
            
            if j == n_iter_max - 1:
                print("Warning! Picard iterations did not converge.")
 

    thta = thvec(h, theta_r, theta_s, n, m, alpha)

    return z, t, h, thta
    
