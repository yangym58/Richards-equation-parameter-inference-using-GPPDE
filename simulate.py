#!/usr/bin/env python
# coding: utf-8

####### packages #######
print("Importing packages...")
import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], 'src'))
from hydro import *
from richards import picard_solver
print("Importing packages...Done")

# random seeds and printing 
np.random.seed(0)
np.set_printoptions(precision=4)

# Hydrological constants
theta_r = 0.075
theta_s = 0.287
k_s = 1.5

# Van Genuchten model parameters
alpha = 0.05
m = 0.5
param = [theta_r, theta_s, k_s, alpha, m]

# Richards equation resolution parameters
h_t0 = -65
h_z0 = -65
h_z60 = -20.7

# Solving Richards equation and formatting solution as a dataset for GPR
print("Simulating the data...")
z_max = 60
t_max = 50
dz = 0.1
dt = 0.1
z, t, h, thta = picard_solver(z_max=z_max, t_max=t_max, dz=dz, dt=dt, param = param, h_t0=h_t0, h_z0=h_z0, h_z_max=h_z60)
nz = len(z)
nt = len(t)

print(thta.shape)
np.savetxt("theta.txt", thta, fmt='%.6e')


