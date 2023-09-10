import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as int

from utils.constants import *
import GnedinCooling.gnedincooling as gc
import GnedinCooling.cross_section as cs

gc.frtinitcf(0, 'cf_table.I2.dat')
metall = 1  # metallicity in units of solar metallicity = 0.02

k_quasar = 2.2e4
# k_quasar = 1e15

def R_star(L):
    return 1.33 * Rsun * (L / 1.02 / Lsun) ** 0.142


def a_Ferland(v):       #photoionization cross section (Osterbrock, Ferland)
    eps = np.sqrt(v/(ryd/h) - 1)
    return 6.3E-18 * ((ryd/h)/v)**4 * np.exp(4-4*np.arctan(eps)/eps) / (1-np.exp(-2*np.pi/eps))


def a_Verner(nZ, ne, sh, v):        #photoionization cross section (Verner1996)
    return 1e6 * 1e-24 * cs.phfit2(nZ, ne, sh, h*v/eV)


def alphaB(T):
    # Te = (T * kB / eV)               # recombination coefficient (Abel 2017)
    # if T > 5500:
    #     coeff = [-28.61303380689232, -7.241125657826851e-1, -2.026044731984691e-2, -2.380861877349834e-3,
    #              -3.212605213188796e-4, -1.421502914054107e-5, 4.989108920299510e-6, 5.755614137575750e-7,
    #              -1.856767039775260e-8, -3.071135243196590e-9]
    #     res = 0
    #     for i in range(10):
    #         res = res + coeff[i] * np.power(np.log(Te), i)
    #     return np.exp(res)
    # return 3.92e-13 * Te**(-0.6353)
    return 8.4e-11 * T**(-0.5) * (T/1e3)**(-0.2) / (1+(T/1e6)**0.7)     # recombination coefficient (Cen 1992)


def betaB(T):                                  # recombination energy loss coefficient (Cen 1992)
    return 6.3e-11 * T**(-0.5) * (T/1e3)**(-0.2) / (1+(T/1e6)**0.7)


def gamma(T):
    return 0
    return 5.85e-11 * T**0.5 * (1 + (T/1e5)**0.5) * np.exp(-157809.1/T)


def bb(v, L):    # Wien approx black body
    if L==0: return 0
    Teff = (L / (4 * np.pi * R_star(L)**2 * sigmaSB))**(1./4)
    return (2 * h * v**3 / clight**2) * np.exp(- h * v / kB / Teff)


def Lv(v, L, type):
    if type=='quasar':
        return 6.2e-17 * (v / (ryd / h)) ** (0.5 - 2) * L  # quasars (valid only for v > ryd/h)
    if type=='star':
        return (4 * np.pi * R_star(L) ** 2) * np.pi * bb(v, L)  # stars


def vLv(v, L, type):
    return v * Lv(v, L, type)


def f_H(v, D, L, type, NH):
    return Lv(v, L, type) / (4 * np.pi * D**2) / v * (v - ryd/h) * a_Verner(1, 1, 1, v) * np.exp(- NH * a_Verner(1, 1, 1, v))


def g_H(v, D, L, type, NH):
    return Lv(v, L, type) / (4 * np.pi * D**2) / h / v * a_Verner(1, 1, 1, v) * np.exp(- NH * a_Verner(1, 1, 1, v))


def Res1(D, L, type, NH):
    if type=='quasar': k = k_quasar
    if type=='star': k = 1e2
    int1 = int.quad(f_H, ryd/h, k * ryd/h, args=(D, L, type, NH), points=[10*ryd/h, 100*ryd/h])
    return int1[0]


def Res2(D, L, type, NH):
    if type=='quasar': k = k_quasar
    if type=='star': k = 1e2
    int2 = int.quad(g_H, ryd/h, k * ryd/h, args=(D, L, type, NH), points=[10*ryd/h, 100*ryd/h])
    return int2[0]


def x(T, D, L, n, type, NH=0.):       # fraction of ionized Hydrogen (includes recombination and collisional ionization)
    a = alphaB(T) - gamma(T)
    b = Res2(D, L, type, NH) / n
    c = -b
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2 / a


def G_H(T, D, L, n, type, NH):                                     # photoionizzation heating
    return x(T, D, L, n, type, NH)**2 * alphaB(T) * Res1(D, L, type, NH) / Res2(D, L, type, NH)


def H_Compton(T, D, L, n, type):
    if type == "star":
        F = L / (4 * np.pi * D**2)
        int3 = int.quad(vLv, ryd/h, 1e2 * ryd/h, args=(L, type))
        E_ave = int3[0] * h / L
        return sigma_T * F / me / clight**2 * (E_ave - 4 * kB * T) / n
    else:
        F = L / (4 * np.pi * D**2)
        int3 = int.quad(vLv, ryd/h, k_quasar * ryd/h, args=(L, type), points=[10*ryd/h, 100*ryd/h])
        E_ave = int3[0] * h / L
        return sigma_T * F / me / clight**2 * (E_ave - 4 * kB * T) / n


def dRate(v, D, L, type, spec, NH):
    if spec=='H2':
        return Lv(v, L, type) / (4 * np.pi * D**2) / h / v * 1e-18
    if spec=='HI':
        nZ = 1
        ne = 1
    if spec=='HeI':
        nZ = 2
        ne = 2
    if spec=='CVI':
        nZ = 6
        ne = 1
    return Lv(v, L, type) / (4 * np.pi * D**2) / h / v * a_Verner(nZ, ne, 1, v) * np.exp(- NH * a_Verner(nZ, ne, 1, v))


def Rate(D, L, type, spec, NH):
    if type=='quasar': k = k_quasar
    if type=='star': k = 1e2
    if spec=='H2':
        int3 = int.quad(dRate, 11.2*eV/h, ryd / h, args=(D, L, type, spec, NH))
        return int3[0]
    if spec=='HI': lim_inf = ryd
    if spec=='HeI': lim_inf = 24.5874 * eV
    if spec=='CVI': lim_inf = 36 * ryd
    int3 = int.quad(dRate, lim_inf/h, k * ryd/h, args=(D, L, type, spec, NH), points=[1.2*lim_inf/h, 1.4*lim_inf/h, 1.6*lim_inf/h, 1.8*lim_inf/h])
    return int3[0]


def L_Gnedin(T, D, L, n, type, NH):
    Plw = Rate(D, L, type, 'H2', NH)
    Ph1 = Rate(D, L, type, 'HI', NH)
    Pg1 = Rate(D, L, type, 'HeI', NH)
    Pc6 = Rate(D, L, type, 'CVI', NH)
    return gc.frtgetcf_cool(T, n, metall, Plw, Ph1, Pg1, Pc6)


def G_Gnedin(T, D, L, n, type, NH):
    Plw = Rate(D, L, type, 'H2', NH)
    Ph1 = Rate(D, L, type, 'HI', NH)
    Ph1 = Rate(D, L, type, 'HI', NH)
    Pg1 = Rate(D, L, type, 'HeI', NH)
    Pc6 = Rate(D, L, type, 'CVI', NH)
    return gc.frtgetcf_heat(T, n, metall, Plw, Ph1, Pg1, Pc6)


def energy(T, D, L, n, type, NH):
    Plw = Rate(D, L, type, 'H2', NH)
    Ph1 = Rate(D, L, type, 'HI', NH)
    Pg1 = Rate(D, L, type, 'HeI', NH)
    Pc6 = Rate(D, L, type, 'CVI', NH)
    return gc.frtgetcf_heat(T, n, metall, Plw, Ph1, Pg1, Pc6) - gc.frtgetcf_cool(T, n, metall, Plw, Ph1, Pg1, Pc6) #+  H_Compton(T, D, L, n, type)


def Dthickness(v, T, nH, D, L, type):
    if x(T, D, L, nH, type, 0) < 0.05:
        return 0
    return Lv(v, L, type) / (4 * np.pi * D**2) / (h*v) / x(T, D, L, nH, type, 0)**2 / nH**2 / alphaB(T)


def Temperature(D, L, n, type, NH=0.):
    return opt.brentq(energy, 1e1, 1e6, args=(D, L, n, type, NH))


def Thickness(nH, D, L, type):
    temp = Temperature(D, L, nH, type)
    if type=='quasar': k = k_quasar
    if type=='star': k = 1e2
    I = int.quad(Dthickness, ryd/h, k * ryd / h, args=(temp, nH, D, L, type), points=[10*ryd/h, 100*ryd/h])
    return I[0]
