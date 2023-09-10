import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from utils.constants import *

# Bialy analytical solution is valid only for constant density PDRs

# constant parameters
sigma_g_MW = 1.9e-21 # cm2    # mean dust cross section in the MW
sigma_d_LW = 2.36e-3 # cm2 Hz # mean H2 dissociation cross section in LW

# adjustable parameters
#H2_dissociation_coeff = 5.8e-11
H2_dissociation_coeff = 2.*7.727999999999999e-11# 2. * 2.59e-11

#H2_formation_coeff    = 3.e-17
H2_formation_coeff = 3.5e-17 #3.e-18 * np.sqrt(60)
#sigma_g=sigma_g_MW
sigma_g = 1045.7 * mp

def pdr_alpha(n, csi, sigma_g=sigma_g_MW):
    D0    = H2_dissociation_coeff * csi
    Rform = H2_formation_coeff * sigma_g / sigma_g_MW
    pdr_alpha = D0 / Rform / n
    return pdr_alpha

def pdr_Wg(N2, sigma_g=sigma_g_MW): #analytical fit valid for N2 > 1.e14
    a1 = 3.6e11; a2 = 0.62; a3 = 2.6e3
    a4 = 1.4e7 * (1 + 8.9 * sigma_g/sigma_g_MW)**(-0.93)
    y  = N2 / 1.e14
    pdr_Wg = a1 * np.log( (a2 + y) / (1. + y/a3) ) * ( (1. + y/a3) / (1. + y/a4) )**0.4
    return pdr_Wg

def pdr_Wg_tot(sigma_g=sigma_g_MW):
    pdr_Wg_tot = 8.8e13 / ( 1 + 8.9 * sigma_g/sigma_g_MW)**0.37
    return pdr_Wg_tot

def pdr_G(sigma_g=sigma_g_MW):
    pdr_G = sigma_g / sigma_d_LW * pdr_Wg_tot(sigma_g)
    return pdr_G

def f_shield_Richings(N2, T=1000.):
    btherm = np.sqrt(kB*T / mp)
    bturb = 7.1e5                           # what about this bturb??
    b5 = np.sqrt(btherm**2 + bturb**2)
    b5 = b5 / 1.e5
    omega = 0.013 * (1. + (T/2700.)**1.3)**(1./1.3) * np.exp(-(T/3900.)**14.6)
    pdr_alpha = 0.
    Ncrit = 0.
    if (T < 3000.):
        pdr_alpha = 1.4
        Ncrit = 1.3 * (1 + T/600.)**0.8
    elif (T >= 4000.):
        pdr_alpha = 1.1
        Ncrit = 2.
    else:
        pdr_alpha = (T/4500.)**(-0.8)
        Ncrit = (T/4760.)**(-3.8)
    Ncrit = Ncrit * 1.e14
    x = N2 / Ncrit
    S = (1-omega)/(1+x/b5)**pdr_alpha * np.exp(-5.e-7*(1+x)) \
        + omega/(1+x)**0.5 * np.exp(-8.5e-4*(1+x)**0.5)
    return S

def f_shield(N2):
    x = N2 / 5.e14
    b5 = 2.
    f_shield = 0.965 / (1. + x/b5)**2 + 0.035/(1. + x)**0.5 * np.exp(-8.5e-4*(1.+x)**0.5)
    return f_shield

def pdr_n1_n2(N2, n, csi, sigma_g=sigma_g_MW):
    num = pdr_alpha(n, csi, sigma_g) * pdr_G(sigma_g) * f_shield(N2) * np.exp(-2.*sigma_g*N2)
    den = pdr_alpha(n, csi, sigma_g) * pdr_G(sigma_g) * pdr_Wg(N2, sigma_g) + 2. * pdr_Wg_tot(sigma_g)
    pdr_n1_n2 = sigma_d_LW / sigma_g * num / den
    return pdr_n1_n2

def pdr_N1(N2, n, csi, sigma_g=sigma_g_MW):
    arg = pdr_alpha(n, csi, sigma_g) * pdr_G(sigma_g) / 2. * pdr_Wg(N2, sigma_g) / pdr_Wg_tot(sigma_g)
    pdr_N1 = np.log(arg + 1) / sigma_g
    return pdr_N1

def pdr_N(N2, n, csi, sigma_g=sigma_g_MW):
    return pdr_N1(N2, n, csi, sigma_g) + 2.*N2

def N2_find(N2, N_value, n, csi, sigma_g=sigma_g_MW):
    return pdr_N(N2, n, csi, sigma_g=sigma_g_MW) - N_value

def n1_and_n2_ratio(n1_n2):
    n1_n = n1_n2 / (n1_n2 + 2)
    n2_n = 1. / (n1_n2 + 2)
    return n1_n, n2_n

def pdr_profile(r, nH, csi, sigma_g=sigma_g_MW):   # r in pc, nH constant value, csi in Draine
    N_min = pdr_N(1.e14, nH, csi, sigma_g)
    dr = np.array([r[0]] + [r[i+1]-r[i] for i in range(len(r)-1)])
    N = np.cumsum(dr * pc * nH)
    N[N < N_min] = N_min
    N2 = []
    for NN in N:
        NN2 = brentq(N2_find, 1.e14, NN, args=(NN, nH, csi, sigma_g))
        N2.append(NN2)
    N2 = np.array(N2)
    n1_n2 = pdr_n1_n2(N2, nH, csi, sigma_g)
    n1_n, n2_n = n1_and_n2_ratio(n1_n2)
    return n1_n, n2_n

def pdr_transition(nH, csi, sigma_g=sigma_g_MW):
    return None ### scrivere funzione

