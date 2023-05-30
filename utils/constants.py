# CGS

import numpy as np

h = 6.626E-27
kB = 1.38E-16
G = 6.674E-8
mp = 1.6726E-24
me = 9.1094E-28

eV = 1.6021772E-12
ryd = 13.6*eV
vL = ryd / h
gamma1 = 5./3   # gamma monoatomic gas
gamma2 = 7./5   # gamma diatomic gas
Msun = 1.99E33
Rsun = 7e10
Zsun = 0.0134
Lsun = 3.828e33     # erg/s
pc = 3.09E18
kpc = 1e3*pc
yr = 3.154E+7
Myr = 1e6*yr
clight = 3E10
muion = 0.63    # mean particle mass for unit proton mass, for a fully ionized gas with Solar metallicity
atm = 1013000.
mmHg = atm/760
sigma_T = 6.652458734E-25       # Thomson cross section
sigmaSB = 5.6704e-5     # Stefan-Boltzmann constant
Habing = 1.6e-3     # Habing flux
habing = 1.6e-3
Draine = 1.71*habing
draine = 1.71*habing


def meandensity(M, R, shape='sphere'):
    assert shape in ['sphere', 'cube'], 'geometry can be sphere or cube'   
    V = 0.
    if shape == 'sphere':
        V = 4. * np.pi / 3. * R**3
    elif shape == 'cube':
        V = 8. * R**3
    return M * Msun / (V*pc**3) / mp #cm-3

def freefalltime(n):
    tff = np.sqrt(3.*np.pi/32./G/mp/n)
    return tff / Myr

def csound(T, m=mp): 
    cs = np.sqrt(kB*T/m)
    return cs / 1.e5 #km/s

def jeanslength(n, T, m=mp):
    Lj = csound(T, m) / np.sqrt(G*mp*n)    
    return Lj / pc

def jeansmass(n, T, m=mp ):
    Mj = (4.*np.pi/3.) * mp * n * jeanslength(n, T, m)**3 / 8.
    return Mj / Msun

def sigmaturb(M, R, alpha):
    return np.sqrt(3./5 * G * M / R * alpha)

