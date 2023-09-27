import numpy as np
from constants import *
import utils.phfit2 as phfit2


def meandensity(M, R, shape='sphere'):
    """
    Returns the mean density [cm-3] of a sphere or a cube, given the mass [g] and the radius [cm]
    """
    assert shape in ['sphere', 'cube'], 'geometry can be sphere or cube'   
    V = 0.
    if shape == 'sphere':
        V = 4. * np.pi / 3. * R**3
    elif shape == 'cube':
        V = 8. * R**3
    return M * Msun / (V*pc**3) / mp #cm-3

def freefalltime(n):
    """
    Returns the free fall time [Myr], given the number density [cm-3] 
    """
    tff = np.sqrt(3.*np.pi/32./G/mp/n)
    return tff / Myr

def csound(T, m=mp): 
    """
    Returns the sound speed [km/s] in a medium with temperature T [K] and mean particle mass m [g]
    Default value of m is the proton mass
    """
    cs = np.sqrt(kB*T/m)
    return cs / 1.e5 #km/s

def jeanslength(n, T, m=mp):
    """
    Returns the Jeans length [pc], given 
    - the number density n [cm-3]
    - the temperature T [K]
    - the mean particle mass m [g]. Default values is the proton mass. 
    """
    Lj = csound(T, m)*1e5 / np.sqrt(G*mp*n)    
    return Lj / pc

def jeansmass(n, T, m=mp ):
    """
    Returns the Jeans mass [Msun], given 
    - the number density n [cm-3]
    - the temperature T [K]
    - the mean particle mass m [g]. Default values is the proton mass. 
    """
    Mj = (4.*np.pi/3.) * mp * n * (pc*jeanslength(n, T, m)/2.)**3
    return Mj / Msun

def sigmaturb(M, R, alpha):
    """
    Returns the rms velocity of a turbulent cloud [cm/s], given
    - the cloud mass M [g]
    - the cloud radius R [cm]
    - the virial parametera alpha 
    """
    return np.sqrt(3./5 * G * M / R * alpha)


def a_Ferland(v):      
    """
    photoionization cross section (Osterbrock, Ferland book) in cm-2, as a function of frequency 
    """

    eps = np.sqrt(v/(ryd/h) - 1)
    return 6.3E-18 * ((ryd/h)/v)**4 * np.exp(4-4*np.arctan(eps)/eps) / (1-np.exp(-2*np.pi/eps))


def a_Verner(nZ, ne, sh, v):        
    """
    photoionization cross section (Verner1996) in cm-2
    - nZ : atomic number from 1 to 30 (integer) 
    - ne : number of electrons from 1 to iz (integer)
    - sh : shell number (integer)
    - v  : photon frequency, Hz 
    """

    return 1e6 * 1e-24 * phfit2.phfit2(nZ, ne, sh, h*v/eV)


def alphaB(T):
    return alphaB_Cen92(T)

def alphaB_Cen92(T):
    """
    recombination coefficient on hydrogen, Cen+1992
    """
    return 8.4e-11 * T**(-0.5) * (T/1e3)**(-0.2) / (1+(T/1e6)**0.7)

def alphaB_Abel17(T):
    """
    recombination coefficient on hydrogen, Abel+2017
    """
    Te = np.atleast_1d(T * kB / eV) 
    aB = np.zeros_like(Te)

    coeff = [-28.61303380689232, -7.241125657826851e-1, -2.026044731984691e-2, -2.380861877349834e-3,
             -3.212605213188796e-4, -1.421502914054107e-5, 4.989108920299510e-6, 5.755614137575750e-7,
             -1.856767039775260e-8, -3.071135243196590e-9]

    aB[T>5500.]  = np.exp(np.poly1d(coeff[::-1])(np.log(Te)))
    aB[T<=5550.] = 3.92e-13 * np.power(Te, -0.6353) 

    return aB

def betaB(T):                                  
    """
    recombination energy loss coefficient (Cen 1992)
    """
    return 6.3e-11 * T**(-0.5) * (T/1e3)**(-0.2) / (1+(T/1e6)**0.7)

def gamma_coll(T):
    return 5.85e-11 * T**0.5 * (1 + (T/1e5)**0.5) * np.exp(-157809.1/T)


