import scipy.integrate as int
import numpy as np
from scipy.interpolate import griddata

from utils.constants import *


def bb(v, L):    # Wien approx black body
    Teff = (L / (4 * np.pi * R_star(L)**2 * sigmaSB))**(1./4)
    return (2 * h * v**3 / clight**2) * np.exp(- h * v / kB / Teff)


def Fv_star(v, D, L):
    return (R_star(L) / D) ** 2 * np.pi * bb(v, L)


def Fv_quasar(v, D, L):
    return 6.2e-17 * (v / (ryd / h)) ** (0.5 - 2) * L / (4 * np.pi * D**2)


def G0(D, L, type):
    if type == 'quasar':
        g0 = 6.2e-17 * L / (4 * np.pi * D**2) * (13.6 - 6) * eV/h / Habing
        return g0
    if type == 'star':
        I = int.quad(Fv_star, 6*eV/h, 13.6*eV/h, args=(D, L))
        return I[0]/Habing


def LW(D, L, type):
    if type == 'quasar':
        g0 = 6.2e-17 * L / (4 * np.pi * D**2) * (13.6 - 11.2) * eV/h / Habing
        return g0
    if type == 'star':
        I = int.quad(Fv_star, 11.2*eV/h, 13.6*eV/h, args=(D, L))
        return I[0]/Habing


def Temperature(nH, D, L, type):         # try to fit this function from Kaufman
    data = np.loadtxt("dataPDR.txt")
    logn = data[:, 0]
    logG = data[:, 1]
    T = data[:, 2]
    points = []
    for i in np.arange(len(logn)):
        points.append([logn[i], logG[i]])
    if nH > 1e6 and G0(D, L, type) > 1e4:
        return 5000
    else:
        return griddata(points, T, [np.log10(nH), np.log10(G0(D, L, type))], method='cubic')[0]


def Thickness (nH, D, L, type):         # Tielens Hollenbach 1985
    return 1.5e16 * (2.3e5 / nH)**(7./3) * (G0(D, L, type)/1e5)**(4./3)


def Thickness_Bialy(nH, D, L, type, Z=1):
    alphaG = 0.59 * LW(D, L, type) / 1.71 * (100. / nH) * (9.9 / (1 + 8.9*Z))**0.37
    Ntrans = 0.7 * np.log((0.5 * alphaG)**(1./0.7) + 1.) / (1.9e-21 * Z)
    return Ntrans / nH

def R_star(L):
    return 1.33 * Rsun * (L / 1.02 / Lsun) ** 0.142


