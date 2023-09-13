import numpy as np
from constants import *


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