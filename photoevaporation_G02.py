import numpy as np

from utils.constants import *


def HI_temperature_krome(n, G0):
    import utils.pykrome.pykrome as pk
    from utils.spectra import Spectrum as sp
    krome_bins=[0.7542, 2.65, 6.0, 11.2, 13.6, 14.159, 15.4, 24.59, 30, 54.42, 1000.]
    flux = sp.flat(F=G0*habing, v1=6., v2=13.6)
    F = np.zeros(10)
    F[2] = flux.N_cm2s(krome_bins[2]*eV/h, krome_bins[3]*eV/h)
    F[3] = flux.N_cm2s(krome_bins[3]*eV/h, krome_bins[4]*eV/h)
    cell = pk.cell(n, 10., F=F, bins=krome_bins)
    cell.evolution(1000*Myr, verbose=False) 
    return cell.T

def HI_thickness(nH, G0, Z=1):  #Bialy, Sternberg 2016
    alphaG = 0.59 * G0 / 1.71 * (100. / nH) * (9.9 / (1 + 8.9*Z))**0.37
    Ntrans = 0.7 * np.log((0.5 * alphaG)**(1./0.7) + 1.) / (1.9e-21 * Z)
    dx = Ntrans / nH
    return dx


def evaporationtime(M, R, G0, T_H2=50):
    
    n_ave = (M*0.76) / (4./3 * np.pi * R**3) / mp
    T_HI = HI_temperature_krome(n_ave, G0)
    mu_H2 = 2.5
    mu_HI = 1.74
    delta = HI_thickness(n_ave, G0)
    eta = R/delta
    cc   = np.sqrt(kB * T_H2 / mu_H2 / mp)
    cPDR = np.sqrt(kB * T_HI / mu_HI / mp)
    nu =cPDR/cc
    lambd = (eta - 1) / 4 / nu**2
    # q = (lambd/3)**(1./3)
    # tc = SoundTime(T0, M, P_ICM, T_ICM, D, L, Rstar=1e11, type='star')
    if lambd < 1:
        print('Implosion case')
        tPE = 1e4 * (n_ave / 1e5)**(2./3) * (R / 0.01 / pc)**(5./3) * (3e4 / cc)**(2./3) * (3e5 / cPDR)**(1./3) * yr
    else:
        print("Expansion + implosion case")
        tPE = 1e5 * (R / 0.01 / pc)**(1.) * (3e4 / cc)**(2.) * (3e5 / cPDR)**(-1.) * yr
  
    return tPE


def evaporationtime2(M, R, G0, T_H2=50):
    
    alpha = 1.
    beta = 1.
    gamma = 4./3
    
    n_ave = (M*0.76) / (4./3 * np.pi * R**3) / mp
    T_HI = HI_temperature_krome(n_ave, G0)
    mu_H2 = 2.5
    mu_HI = 1.74
    delta = HI_thickness(n_ave, G0)
    cc   = np.sqrt(kB * T_H2 / mu_H2 / mp)
    cPDR = np.sqrt(kB * T_HI / mu_HI / mp)
    nu =cPDR/cc
    
    rc0 = R - delta

    eta = rc0 / delta
    #lambd = (1 + alpha + beta) * (eta - 1) / 2. / (2.*nu**2 + alpha)
    lambd = (eta - 1) / 4 / nu**2
    q = (abs(lambd) / 3.)**(1./3)
    
    rs = q * rc0 * (1 - 3.*(1. + q) / (eta - 1.) )**(4./9) / pc # not used, but can be printed
    
    tc = rc0 / cc
    ts = rc0 / cPDR  # contraction time
    
    A = 3.*(eta - 1.)**2 / (10.*q*eta*nu)
    B = (1. - 3.*(1+q)/(eta-1))**(5./9)
    C = (eta - 1) / eta / nu

    tevap = tc * (A+B + C)
     
    return tevap, [ts, tevap]
    

    
    
    