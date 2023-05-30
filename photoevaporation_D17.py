import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import integrate

from utils.constants import *
import wavesolver as ws
import arbitrarydiscontinuities_isothermal as ad

TH2 = 50

def Mshock(M_in, r_in, r):
    f = M_in + 1/M_in - np.log(r/r_in)
    return (f + np.sqrt(f**2 - 4)) / 2

def invM(r, M_in, r_in):
    return 1/Mshock(M_in, r_in, r)

def HI_temperature(n, G0):
    data = np.loadtxt("dataPDR.txt")
    logn = data[:, 0]
    logG = data[:, 1]
    T = data[:, 2]
    points = []
    for i in np.arange(len(logn)):
        points.append([logn[i], logG[i]])
    if n > 1e6 and G0 > 1e4:
        return 5000
    else:
        return griddata(points, T, [np.log10(n), np.log10(G0)], method='cubic')[0]
    
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

def HI_thickness_empirical(nH, G0):
    from utils.spectra import Spectrum as sp
    flux = sp.flat(F= G0*habing, v1=6., v2=13.6)
    sigma_d = 1000. * mp
    u_subLW = flux.N_cm2s(6.*eV/h, 11.2*eV/h)
    NHI = np.log(u_subLW / 2.1e10) / sigma_d
    dx = NHI / nH
    return dx
    

def evaporationtime(M, R, G0, P_ICM, T_ICM, n_max, T_pdr='analytical', d_pdr='bialy'):
    
    n     = M / (4./3. * np.pi * R**3) / mp    #average density
    nH    = n * 0.76
    
    assert T_pdr in ['analytical', 'krome']
    if T_pdr == 'analytical':
        T_HI = HI_temperature(n, G0)
    elif T_pdr == 'krome':
        T_HI = HI_temperature_krome(n, G0)
        
    H2 = ws.State(n_max * mp, None, 0, 7. / 5, TH2, 2.5 * mp)
    HI = ws.State(n * mp, None, 0, 5. / 3, T_HI, mp)
    ICM = ws.State(100*mp, None, 0, 5. / 3, T_ICM, 0.63 * mp)

    H2.show()
    HI.show()
    P_H2HI = ad.discontinuity_solver(H2, HI)

    H2s = ws.isothermalshocksolver_Pgiven(H2, P_H2HI)
    vH2s = ws.isothermalshockspeed(H2, H2s)
    H2s.v = - H2s.v
    
    assert d_pdr in ['bialy', 'empirical']
    if d_pdr == 'bialy':
        print '*** D17 ***', nH, G0
        dHI = HI_thickness(nH, G0)
    elif d_pdr == 'empirical':
        dHI = HI_thickness_empirical(nH, G0)
      
    print '--- D17 ---', R/pc, dHI/pc
    r0 = R - dHI  
    
    if r0 < 0:
        return 0

    tcenter = integrate.quad(invM, 1e-5*pc, r0, args=(vH2s / H2.isothermalcsound(), r0))[0] / H2.isothermalcsound()
    redge_tcenter = r0 + H2s.v * tcenter

    def f(xx):
        return integrate.quad(invM, redge_tcenter/1e6, xx, args=(vH2s / H2.isothermalcsound(), r0))[0] \
                 / H2.isothermalcsound() + (redge_tcenter - xx) / H2s.v

    xreflection = opt.brentq(f, redge_tcenter/1e6, 0.99 * redge_tcenter)
    treflection = - (redge_tcenter - xreflection) / H2s.v

    tiii = tcenter + treflection                    # after this, free expansion!
    rcore = redge_tcenter + H2s.v * treflection
    n_ave = M / (4./3 * np.pi * R**3) / mp
    Sconv = ws.State(state=H2s)
    Sdiv = ws.isothermalshocksolver_Dgiven(Sconv, vH2s)
    Sdiv = ws.State(Sdiv.rho, None, 0, 1.4, Sdiv.T, 2*mp)
    Svoid = ICM
    Pexp = ad.discontinuity_solver(Svoid, Sdiv)
    Sdiv.v = -Sdiv.v
    Sexp = ws.isothermalrarefactionsolver_Pgiven(Sdiv, Pexp)
    Sdiv.v = -Sdiv.v
    vexp = -Sexp.v

    def H2_HI(t):
        return rcore + vexp * t

    def maxHI(t):
        return HI_thickness(n_ave * (R / H2_HI(t))**3, G0)

    def left(t):
        return H2_HI(t) - maxHI(t)

    tevap = opt.brentq(left, 0, 1e15)
    t_life = tiii + tevap

    return t_life, [tcenter, t_life]
