import numpy as np
import pynbody
import pynbody.units as units
import pynbody.snapshot.tipsy
from pynbody.snapshot.ramses import RamsesSnap
from pynbody.snapshot.tipsy import TipsySnap
from pynbody.array import SimArray

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, name='Jesper')

mm = {
    'ion1' : 1.67353251819e-24 * units.g ,
    'ion2' : 9.10938188e-28    * units.g ,
    'ion3' : 1.67262158e-24    * units.g ,
    'ion4' : 6.69206503638e-24 * units.g ,
    'ion5' : 6.69115409819e-24 * units.g ,
    'ion6' : 6.69024316e-24    * units.g ,
    'ion7' : 1.67444345638e-24 * units.g ,
    'ion8' : 3.34706503638e-24 * units.g ,
    'ion9' : 3.34615409819e-24 * units.g
    }


hp = units.hP
me = units.m_e
kB = units.k
eV = units.eV

E0   = 13.6 * units.eV #1s level
E1   = 3.4  * units.eV #2p level
ELya = 10.2 * units.eV

_kB = kB.in_units('erg K^-1')
_eV = eV.in_units('erg')
_ELya = ELya.in_units('erg')

@pynbody.derived_array
def n(sim):
    nn = sim['rho'] / units.m_p
    return nn.in_units('cm^-3')

@pynbody.derived_array
def numden(sim):
    return sim['n']

@pynbody.derived_array
def v(sim):
    vel = sim['vel'].in_units('km s**-1')
    vv = np.linalg.norm(vel, axis=1)
    return SimArray(vv, 'km s**-1')

@pynbody.derived_array
def vel_r(sim):
    pos = sim['pos'].in_units('kpc')
    vel = sim['vel'].in_units('km s**-1')
    pos /= np.linalg.norm(pos, axis=1)[:, None]
    velr = np.einsum('ij,ij->i', vel, pos)
    return SimArray(velr, 'km s**-1')




#############- tipsy quantities


@TipsySnap.derived_quantity
def p(sim):
    T   = sim['temp']
    rho = sim['rho']
    mu  = sim['mu']
    pp  = rho / mu / units.m_p * units.k * T
    return pp.in_units('g  cm**-1 s**-2')



#############- ramses quantities


@RamsesSnap.derived_quantity
def mu(sim):
    out = SimArray(sim['ion1'], '') / mm['ion1']
    for i in range(2,10):
        ion = 'ion' + str(i)
        out += SimArray(sim[ion], '') / mm[ion]
    out = 1. / out / mm['ion1']
    return out

@RamsesSnap.derived_quantity
def temp(sim):
    tt =  (sim['p'] / sim['rho']) * (sim['mu'] * units.m_p / units.k)
    return tt.in_units('K')

@RamsesSnap.derived_quantity
def T(sim):
    return sim['temp']

def alphaB(T): #Grassi+14
    Te = np.asarray(T) * _kB / _eV
    k = np.where(T<=5500.,
                 3.92e-13*Te**(-0.6353),
                 np.exp(- 28.61303380689232 \
                  - 7.241125657826851e-1 * (np.log(Te))    \
                  - 2.026044731984691e-2 * (np.log(Te))**2 \
                  - 2.380861877349834e-3 * (np.log(Te))**3 \
                  - 3.212605213188796e-4 * (np.log(Te))**4 \
                  - 1.421502914054107e-5 * (np.log(Te))**5 \
                  + 4.989108920299510e-6 * (np.log(Te))**6 \
                  + 5.755614137575750e-7 * (np.log(Te))**7 \
                  - 1.856767039775260e-8 * (np.log(Te))**8 \
                  - 3.071135243196590e-9 * (np.log(Te))**9) )
    return SimArray(k, 'cm**3 s**-1')

@RamsesSnap.derived_quantity
def xHI(sim):
    return sim['ion1']

@RamsesSnap.derived_quantity
def xHII(sim):
    return sim['ion3']

@RamsesSnap.derived_quantity
def xH2(sim):
    return sim['ion8']

@RamsesSnap.derived_quantity
def nHI(sim):
    nHI = sim['rho'] * SimArray(sim['ion1'], '') / mm['ion1']
    return nHI.in_units('cm**-3')

@RamsesSnap.derived_quantity
def nHII(sim):
    nHII = sim['rho'] * SimArray(sim['ion3'], '') / mm['ion3']
    return nHII.in_units('cm**-3')

@RamsesSnap.derived_quantity
def ne(sim):
    ne = sim['rho'] * SimArray(sim['ion2'], '') / mm['ion2']
    return ne.in_units('cm**-3')

@RamsesSnap.derived_quantity
def nH2(sim):
    nH2 = sim['rho'] * SimArray(sim['ion8'], '') / mm['ion8']
    return nH2.in_units('cm**-3')

@RamsesSnap.derived_quantity
def nH(sim):
    return sim['nHI'] + sim['nHII']

@RamsesSnap.derived_quantity
def fHI(sim):
    return sim['nHI'] / sim['nH']

@RamsesSnap.derived_quantity
def MHI(sim):
    return sim['mass'] * SimArray(sim['ion1'], '')

@RamsesSnap.derived_quantity
def MHII(sim):
    return sim['mass'] * SimArray(sim['ion3'], '')

@RamsesSnap.derived_quantity
def MH2(sim):
    return sim['mass'] * SimArray(sim['ion8'], '')


def gamma_1s2p(T): #Scholz1991

    T =  np.asarray(T)
    lnT = np.log(T)
    k  = np.zeros_like(T)

    c0 = [-1.630155e2, 5.279996e2, -2.8133632e3]
    c1 = [8.795711e1, -1.939399e2, 8.1509685e2]
    c2 = [-2.057117e1, 2.718982e1, -9.4418414e1]
    c3 = [2.359573, -1.883399, 5.4280565]
    c4 = [-1.339059e-1, 6.462462e-2, -1.5467120e-1]
    c5 = [3.021507e-3, -8.811076e-4, 1.7439112e-3]

    mask1 = np.logical_and(T>2e3, T<=6e4)
    mask2 = np.logical_and(T>6e4, T<=6e6)
    mask3 = np.logical_and(T>6e6, T<1e8)

    for i, mask in enumerate([mask1, mask2, mask3]):
        k[mask] =   c0[i]                \
                  + c1[i] * lnT[mask]    \
                  + c2[i] * lnT[mask]**2 \
                  + c3[i] * lnT[mask]**3 \
                  + c4[i] * lnT[mask]**4 \
                  + c5[i] * lnT[mask]**5
        k[mask] = np.exp(k[mask]) * np.exp(-_ELya/_kB/T[mask])

    return SimArray(k, 'cm^3 s^-1')


@RamsesSnap.derived_quantity
def Lya_c(sim):
    T = sim['T'].in_units('K')
    Lya_c = sim['ne'] * sim['nHI'] * gamma_1s2p(T) * ELya
    return Lya_c.in_units('erg cm**-3 s**-1')

@RamsesSnap.derived_quantity
def Lya_r(sim):
    aB = alphaB(sim['T'].in_units('K'))
    Tq = np.array(sim['T'].in_units('K') / 1e4)
    P_lya = 0.686 - 0.106*np.log10(Tq) - 0.009*Tq**(-0.44) #Cantalupo+08
    R_rec = sim['ne'] * sim['nHII'] * aB
    Lya_r = P_lya * R_rec * ELya

    return Lya_r.in_units('erg cm**-3 s**-1')

@RamsesSnap.derived_quantity
def Lya(sim):
    Lya = (sim['Lya_c'] + sim['Lya_r'])*sim['smooth']**3
    return Lya.in_units('erg s**-1')

@RamsesSnap.derived_quantity
def SB_c(sim):
    L = sim['Lya_c']
    sim.properties['z'] = 2.
    dL = cosmo.luminosity_distance(sim.properties['z']).value * units.kpc
    as_kpc = cosmo.arcsec_per_kpc_proper(sim.properties['z']).value * units.arcsec / units.kpc
    dx = sim['smooth']
    SB_c = L *dx / 4. / np.pi / dL**2 / as_kpc**2
    return SB_c.in_units('erg s**-1 cm**-2 arcsec**-2')

@RamsesSnap.derived_quantity
def SB_r(sim):
    L = sim['Lya_r']
    sim.properties['z'] = 2.
    dL = cosmo.luminosity_distance(sim.properties['z']).value * units.kpc
    as_kpc = cosmo.arcsec_per_kpc_proper(sim.properties['z']).value * units.arcsec / units.kpc
    dx = sim['smooth']
    SB_r = L * dx / 4. / np.pi / dL**2 / as_kpc**2
    return SB_r.in_units('erg s**-1 cm**-2 arcsec**-2')

@RamsesSnap.derived_quantity
def SB(sim):
#    L = sim['Lya']
#    sim.properties['z'] = 0.3
#    dL = cosmo.luminosity_distance(sim.properties['z']) * units.kpc
#    as_kpc = cosmo.arcsec_per_kpc_proper(sim.properties['z']) * units.arcsec / units.kpc
#    dx = sim['smooth']
#    SB = L / 4. / np.pi / dL**2 / dx**2 / as_kpc**2
#    return SB.in_units('erg s**-1 cm**-2 arcsec**-2')
     return sim['SB_c'] + sim['SB_r']

@RamsesSnap.derived_quantity
def FUVa(sim):
    flux  = sim['rad_3_rho'] * 0.5*(6.+11.2) * units.eV
    flux *= sim['smooth'].in_units('cm')**2
    flux /= ((11.2 - 6.)*units.eV/hp)
    return flux.in_units('erg s**-1 Hz**-1')

@RamsesSnap.derived_quantity
def FUVb(sim):
    flux  = sim['rad_4_rho'] * 0.5*(11.2+13.6) * units.eV
    flux *= sim['smooth'].in_units('cm')**2
    flux /= ((13.6 - 11.2)*units.eV/hp)
    return flux.in_units('erg s**-1 Hz**-1')
