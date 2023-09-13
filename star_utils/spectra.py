import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from pystellibs import BaSeL, Kurucz
import star_utils.stellar_properties as sprop

from utils.constants import *

v0 = ryd / h
f_one = lambda x: 1.
evh = eV/h


class Spectrum(object):

    """
    class Spectrum

    Parameters:
    flux   : function returning the flux [erg/cm2/s/Hz] as a function of frequency [Hz]
    R      : distance where the flux has been calculated [cm]. 
        If None, the radius is not known and it will not be possible to compute distance-dependent quantities
    omega  : field of view covered by the source [sr]. 4pi for an isotropic source, pi for a spherical source.
    scale  : factor 1/r**2 when changing the distance of the source with set_distance
    renorm : multiplicative factor to the flux (if you have N identical sources)

    Methods:
    renormalize  : rescale the flux by a multiplicative factor
    set_distance : change the distance at which the flux is measured 
    Fv_cm2s      : flux in [erg/cm2/s/Hz] at frequency v [Hz]
    Nv_cm2s      : flux in [photons/cm2/s/Hz] at frequency v [Hz]
    Fv_cm2srs    : flux in [erg/cm2/s/sr/Hz] at frequency v [Hz]
    Nv_cm2srs    : flux in [photons/cm2/s/sr/Hz] at frequency v [Hz]
    N_cm2s       : flux in [photons/cm2/s] in the frequency range (v1, v2) [Hz]
    F_cm2s       : flux in [erg/cm2/s] in the frequency range (v1, v2) [Hz]
    N_cm2srs     : flux in [photons/cm2/s/sr] in the frequency range (v1, v2) [Hz]
    F_cm2srs     : flux in [erg/cm2/s/sr] in the frequency range (v1, v2) [Hz]
    G            : Habing flux

    The following cannot be computed if a distance R is not provided
    Nv_s         : flux in [photons/s/Hz] at frequency v [Hz]
    Fv_s         : flux in [erg/s/Hz] at frequency v [Hz]
    Nv_srs       :  flux in [photons/s/sr/Hz] at frequency v [Hz]
    Fv_srs       : flux in [erg/s/Hz] at frequency v [Hz]
    N_s          : flux in [photons/cm2/s/sr/Hz] in the frequency range (v1, v2) [Hz]
    L            : flux in [erg/s] in the frequency range (v1, v2) [Hz]
    N_srs        : flux in [photons/s/sr] in the frequency range (v1, v2) [Hz]
    L_sr         : flux in [erg/s/sr] in the frequency range (v1, v2) [Hz]
    """

    def __init__(self, flux, R, omega):

        self.flux   = flux
        self.R      = R
        self.omega  = omega
        self.scale  = 1.
        self.renorm = 1.

    # ---------------------------------------------------------


    def renormalize(self, renorm):        # multiplicative factor to the sed
        self.renorm = self.renorm * renorm

    def Fv_cm2s(self, v, funct=f_one, *args):
        return self.flux(v) * self.scale * self.renorm * funct(v, *args)

    def Nv_cm2s(self, v, funct=f_one, *args):
        return self.Fv_cm2s(v, funct, *args) / h / v

    def Fv_cm2srs(self, v, funct=f_one, *args):
        return self.Fv_cm2s(v, funct, *args) / self.omega

    def Nv_cm2srs(self, v, funct=f_one, *args):
        return self.Fv_cm2s(v, funct, *args) / h / v / self.omega

    def N_cm2s(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        a = integrate.quad(self.Nv_cm2s, v1, v2, args=(funct,)+args, **kwargs)
        return a[0]

    def F_cm2s(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        a = integrate.quad(self.Fv_cm2s, v1, v2, args=(funct,)+args, **kwargs)
        return a[0]

    def N_cm2srs(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        a = integrate.quad(self.Nv_cm2s, v1, v2, args=(funct,)+args, **kwargs)
        return a[0] / self.omega

    def F_cm2srs(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        a = integrate.quad(self.Fv_cm2s, v1, v2, args=(funct,)+args, **kwargs)
        return a[0] / self.omega

    def G(self, funct=f_one, *args, **kwargs):
        return self.F_cm2s(6*eV/h, 13.6*eV/h, funct, *args, **kwargs)/Habing

    # ------- functions valid only for a spherical source

    def set_distance(self, r):                        # place the source at distance r
        self._checkR()
        self.scale *= (self.R / r) ** 2
        self.R = r

    def Nv_s(self, v, funct=f_one, *args):
        self._checkR()
        return self.Nv_cm2s(v, funct, *args) * 4 * np.pi * self.R**2

    def Fv_s(self, v, funct=f_one, *args):
        self._checkR()
        return self.Fv_cm2s(v, funct, *args) * 4 * np.pi * self.R**2

    def Nv_srs(self, v, funct=f_one, *args):
        self._checkR()
        return self.Nv_cm2s(v, funct, *args) * 4 * np.pi * self.R ** 2 / self.omega

    def Fv_srs(self, v, funct=f_one, *args):
        self._checkR()
        return self.Fv_cm2s(v, funct, *args) * 4 * np.pi * self.R ** 2 / self.omega

    def N_s(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        self._checkR()
        return self.N_cm2s(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def L(self, v1, v2, funct=f_one, *args, **kwargs):
        self._checkR()
        return self.F_cm2s(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def N_srs(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        self._checkR()
        return self.N_cm2srs(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def L_sr(self, v1, v2, funct=f_one, *args, **kwargs):
        self._checkR()
        return self.F_cm2srs(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def _checkR(self):
        if self.R is None:
            raise ValueError("Impossible to use this method without giving the radius at which the given flux is measured")
            


class star(Spectrum):
    """
    class star
    Generates a stellar Spectrum object for a given stellar luminosity [erg/s]
    The initial distance at which the flux is computed is the stellar radius
    You can choose black_body spectrum, basel library or kurucz library

    The additional parameters are 
    Lstar : luminosity of the star [erg/s] 
    Rstar : radius of the star [cm]
    Mstar : mass of the star [g]
    Zstar : metallicity of the star
    Teff  : effective temperature [K]
    """

    def __init__(self, Lstar, Mstar=None, Rstar=None, Zstar=Zsun, library='black_body'):

        assert library in ['black_body', 'basel', 'kurucz']

        self.Lstar = Lstar

        if Rstar is None:
            self.Rstar = sprop.radius_star(Lstar)
        else:
            self.Rstar = Rstar

        if Mstar is None:
            self.Mstar = sprop.mass_star(Lstar/Lsun, Zstar/Zsun) * Msun
        else:
            self.Mstar = Mstar

        self.Zstar = Zstar
        self.Teff  = (Lstar / (4 * np.pi * self.Rstar** 2 * sigmaSB)) ** (1. / 4)

        if library == 'black_body':
            flux = lambda v: 2 * h * v**3 / clight**2 / (np.exp(h*v/kB/self.Teff) - 1) * np.pi

        else:
            if library == 'basel':       
                stellib = BaSeL()
            elif library == 'kurucz':
                stellib = Kurucz()
            gravity = G * params['Mstar'] / params['Rstar']**2
            # logT [K], logg [cm/s2], logL [Lsun], Z
            ap = (np.log10(self.Teff), np.log10(gravity), np.log10(self.Lstar/Lsun), self.Zstar)
            wavelength = stellib._wavelength * 1.e-8
            frequency = clight / wavelength[::-1]
            try:
                sp = stellib.generate_stellar_spectrum(*ap) * 1.e8 * wavelength**2 / clight     \
                                                        / (4. * np.pi * self.Rstar**2)
                sp = sp[::-1]
            except RuntimeError:
                sp = 2 * h * frequency**3 / clight**2 / (np.exp(h*frequency/kB/params_cleaned['Teff']) - 1) * np.pi
            f = interp1d(frequency, sp, bounds_error=False, fill_value=0.)   # interpolating, all the absorption lines are lost!!
            flux = lambda v: F(v)

        super(star, self).__init__(flux, self.Rstar, np.pi)

        return 


class cmb(Spectrum):
    """
    class cmb
    Generates a CMB Spectrum object for a given CMB temperature

    The only additional parameter is Teff [K]
    """
     
    def __init__(self, Tcmb):

        self.Tcmb = Tcmb

        flux = lambda v: 2 * h * v**3 / clight**2 / (np.exp(h*v/kB/Tcmb) - 1) * np.pi

        super(cmb, self).__init__(flux, None, 4.*np.pi)

        return 


class quasar(Spectrum):
    """
    class quasar
    Generates a quasar Spectrum object for a given bolometric luminosity

    Additional parameters:
    Lquasar : quasar bolometric luminosity [erg/s]
    Rquasar : quasar radius (any initial distance at which you want the flux to be computed)
    """

    def __init__(self, Lquasar, Rquasar=pc):

        self.Lquasar = Lquasar
        self.Rquasar = Rquasar

        def flux(v):
            F = 0.
            if v >= ryd/h:
                F = 6.2e-17 * (v / (ryd / h)) ** (0.5 - 2) * self.Lquasar / (4 * np.pi * self.Rquasar**2)
            else:
                F = 6.2e-17 * self.Lquasar / (4 * np.pi * self.Rquasar**2)
            return F

        super(quasar, self).__init__(flux, self.Rquasar, np.pi)

        return 


class flat(Spectrum):
    """
    class flat
    Generates a flat Spectrum object with a constanat flux [erg/s/cm2] in the energy range (Emin, Emax) [eV]

    Additional parameters:
    - F    : constant flux [erg/s/cm2]
    - vmin : minimum frequency [Hz]
    - vmax : maximum frequency [Hz]
    The flux is 0 outside of the range.
    """

    def __init__(self, F, Emin, Emax):

        self.F = F                #flux F in erg/s/cm2
        self.vmin = Emin * eV/h   #Emin = minimum photon energy in eV
        self.vmax = Emax * eV/h   #Emax = maximum photon energy in eV

        def flux(v):
            g = 0.
            if v >= self.vmin and v <= self.vmax:
                g = self.F / abs(self.vmax - self.vmin)
            return g

        super(flat, self).__init__(flux, None, np.pi)

        return


class draine(Spectrum):
    """
    class draine
    Generates a Draine Spectrum object for a flux csi (in units of a Draine)

    The only additional parameter is csi 
    """
    def __init__(self, csi):

        self.csi = csi    #flux in units of Draine

        def flux(v, **params):
            v = np.array(v)
            if v.shape == ():
                v = np.array([v])
            g = np.zeros_like(v)
            mask = np.logical_and(v >= 5. * eV / h, v <= 13.6 * eV / h)
            E = h * v[mask] / eV
            coeff = [1.658e6, -2.152e5, 6.919e3]
            g[mask] = coeff[0] * E + coeff[1] * E**2 + coeff[2] * E**3
            F = self.csi * g * h * v * 4. * np.pi * h / eV
            if F.shape == ():
                return F[0]
            return F

        super(draine, self).__init__(flux, None, np.pi)

        return 

class bins(Spectrum):
    """
    class bins
    Generates a binned Spectrum object for given bins and flux (array of length N) in each bins (array of length N+1)

    Additional parameters: bins, fluxes 
    """
    def __init__(self, bins, fluxes): 
        #bins in eV
        #fluxes in photons/s/cm2

        assert len(bins) == len(fluxes) + 1, 'size(bins) must be size(fluxes) + 1'

        self.bins = np.array(bins)
        self.fluxes = np.array(fluxes)

        def flux(v, **params):      # erg/s/cm2/Hz
            binsHz = self.bins * eV / h
            delta  = ( self.bins[1:] - self.bins[:-1] ) * eV/h
            for i in range(len(binsHz)-1):
                if v >= binsHz[i] and v < binsHz[i+1]:
                    return self.fluxes[i] * h * v / delta[i]
            return 0.

        super(bins, self).__init__(flux, None, np.pi)

        return


def to_bins(spectrum, bins):
    binsHz = np.array(bins) * eV / h
    fluxes = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        fluxes[i] = spectrum.N_cm2s(v1=binsHz[i], v2=binsHz[i+1])
    new_spectrum   = bins(bins=bins, fluxes=fluxes)
    new_spectrum.R = spectrum.R
    new_spectrum.omega = spectrum.omega
    return new_spectrum


#TODO:
#- add average energy in a bin 
