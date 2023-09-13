import numpy as np
from utils.constants import *
from scipy.optimize import brentq


def luminosity_star(M, Z=1):  # L in Lsun, M in Msun, Z in Zsun (Tout+1996)
    """
    Returns the luminosity of a star in Lsun, given 
    - the mass in Msun
    - the metallicity in Zsun [default = 1]
    """
    tab = [[0.39704170, -0.329135740, 0.347766880, 0.374708510, 0.09011915],
           [8.52762600, -24.41225973, 56.43597107, 37.06152575, 5.45624060],
           [0.00025546, -0.001234610, -0.00023246, 0.000455190, 0.00016176],
           [5.43288900, -8.621578060, 13.44202049, 14.51584135, 3.39793084],
           [5.56357900, -10.32345224, 19.44322980, 18.97361347, 4.16903097],
           [0.78866060, -2.908709420, 6.547135310, 4.056066570, 0.53287322],
           [0.00586685, -0.017042370, 0.038723480, 0.025700410, 0.00383376]]
    Zpol = [1., np.log10(Z), np.log10(Z)**2, np.log10(Z)**3, np.log10(Z)**4]
    alpha, beta, gamma, delta, eps, zeta, eta = np.dot(tab, Zpol)
    L = (alpha*M**5.5 + beta*M**11) / (gamma + M**3 + delta*M**5 + eps * M**7 + zeta*M**8 + eta*M**9.5)
    return L


def mass_star(L, Z=1):  # L in Lsun, M in Msun, Z in Zsun (Tout+1996)
    """
    Returns the mass of a star in Msun, given
    - the luminosity in Lsun
    - the metallicity in Zsun [default = 1]
    """
    foo = lambda mm, zz: luminosity_star(mm, zz) - L
    assert(foo(0.001, Z)*foo(100., Z) < 0.),  "Choose a luminosity between 2e-8 Lsun and 3e6 Lsun"  
    M = brentq(foo, 0.01, 100., args=(Z))
    return M


def radius_star(L):
    """
    Returns the radius of a star [cm], given the luminosity [erg/s]
    """
    return 1.33 * Rsun * (L / Lsun / 1.02) ** 0.142