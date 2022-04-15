import pysgpp
import numpy as np
import scipy as sp


dim = 2
S = np.ones(dim)
K = 1
sigma = 0.4*S
rho = np.eye(dim)
r_low, r_high = 0, 0.05
T = 1
stepsize = 0.1

def payoff_call(S, K, d):
    return np.maximum(np.sum(S-K)/d, 0)
    
#construct matrices:
# A = <phi_p, phi_q>_L^2, D = <x_j*dphi_p/dx_j, x_i*dphi_q/dx_i>_L^2
# E = <phi_p, x_i*dphi_q/dx_i>_L^2

# for t=0:


# CR = pysgpp.CrankNicolson(
#     nTimesteps=10,
#     timestepSize=0.1
# )
# CR.solve()
