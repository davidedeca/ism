import numpy as np
import scipy.integrate as integrate

from utils.constants import *

v0 = ryd / h
f_one = lambda x: 1.


class Spectrum:

    def __init__(self, flux, R, omega, **params):

        self.flux   = flux            # this is a function of frequency v in Hz, returning the flux in erg/cm2/s/hz
        self.R      = R
        self.omega  = omega                # effective solid angle (for a point is 4\pi, for a spherical source is \pi)s
        self.scale  = 1.                   # factor 1 / r**2 when rescaling flux with distance, only used for spherically simmetric sources
        self.renorm = 1.
        self.params = params               # dictionary with all extra-parameters stored

    @classmethod
    def star(cls, library='black_body' , **params):
        assert 'L' in params.keys(), 'Please set the star luminosity L'
        assert library in ['black_body', 'basel', 'kurucz']
        params.setdefault('Rstar' , 1.33 * Rsun * (params['L'] / Lsun / 1.02) ** 0.142)
        if 'Mstar' not in params.keys():
            params.setdefault('Mstar' , mass_star(params['L'] / Lsun) * Msun)
        params.setdefault('Zstar' , Zsun)
        params_cleaned = {'L'     : params['L']     ,
                          'Rstar' : params['Rstar'] ,
                          'Teff'  : (params['L'] / (4 * np.pi * params['Rstar'] ** 2 * sigmaSB)) ** (1. / 4),
                          'Mstar' : params['Mstar'],
                          'Zstar' : params['Zstar'] }
        if library == 'black_body':
            def flux(v, **params):
                F = 2 * h * v**3 / clight**2 / (np.exp(h*v/kB/params['Teff']) - 1) * np.pi
                return F

        else:
            from scipy.interpolate import interp1d
            if library == 'basel':
                from pystellibs import BaSeL
                stellib = BaSeL()
            elif library == 'kurucz':
                from pystellibs import Kurucz
                stellib = Kurucz()
            gravity = G * params['Mstar'] / params['Rstar']**2
            # logT [K], logg [cm/s2], logL [Lsun], Z
            ap = (np.log10(params_cleaned['Teff']), np.log10(gravity), np.log10(params['L']/Lsun), params['Zstar'])
            wavelength = stellib._wavelength * 1.e-8
            frequency = clight / wavelength[::-1]
            try:
                sp = stellib.generate_stellar_spectrum(*ap) * 1.e8 * wavelength**2 / clight     \
                                                        / (4. * np.pi * params['Rstar']**2)
                sp = sp[::-1]
            except RuntimeError:
                sp = 2 * h * frequency**3 / clight**2 / (np.exp(h*frequency/kB/params_cleaned['Teff']) - 1) * np.pi
            f = interp1d(frequency, sp, bounds_error=False, fill_value=0.)   # interpolating, all the absorption lines are lost!!
            def flux(v, **params):
                return f(v)

        return cls(flux, params['Rstar'], np.pi, **params_cleaned)

    @classmethod
    def cmb(cls, **params):
        assert 'T' in params.keys(), 'Please set the CMB temperature T'
        params_cleaned = {'T' : params['T']}

        def flux(v, **params):
            F = 2 * h * v**3 / clight**2 / (np.exp(h*v/kB/params['T']) - 1) * np.pi
            return F

        return cls(flux, None, 4.*np.pi, **params_cleaned)

    @classmethod
    def quasar(cls, **params):
        assert 'L' in params.keys(), 'Please set the quasar luminosity L'
        params.setdefault('Rquasar', pc )           ### set a reasonable default choice for quasar "emitting radius" !!
        params_cleaned = {'L'       : params['L'],
                          'Rquasar' : params['Rquasar'] }

        def flux(v, **params):
            F = 0.
            if v >= ryd/h:
                F = 6.2e-17 * (v / (ryd / h)) ** (0.5 - 2) * params['L'] / (4 * np.pi * params['Rquasar']**2)
            else:
                F = 6.2e-17 * params['L'] / (4 * np.pi * params['Rquasar']**2)
            return F

        return cls(flux, params['Rquasar'], np.pi, **params_cleaned)

    @classmethod
    def flat(cls, **params):
        assert 'F'  in params.keys(), 'Please set the flux F in erg/s/cm2'
        assert 'v1' in params.keys(), 'Please set the minimum photon energy v1 in eV'
        assert 'v2' in params.keys(), 'Please set the maximum photon energy v2 in eV'
        params_cleaned = {'F'  : params['F'] ,
                          'v1' : params['v1'] * eV/h ,
                          'v2' : params['v2'] * eV/h  }

        def flux(v, **params):
            g = 0.
            if v >= params['v1'] and v <= params['v2']:
                g = params['F'] / abs(params['v2'] - params['v1'])
            return g

        return cls(flux, None, np.pi, **params_cleaned)

    @classmethod
    def draine(cls, **params):
        assert 'csi' in params.keys(), 'Please set the draine flux csi'
        params_cleaned = {'csi' : params['csi']}

        def flux(v, **params):
            v = np.array(v)
            if v.shape == ():
                v = np.array([v])
            g = np.zeros_like(v)
            mask = np.logical_and(v >= 5. * eV / h, v <= 13.6 * eV / h)
            E = h * v[mask] / eV
            coeff = [1.658e6, -2.152e5, 6.919e3]
            g[mask] = coeff[0] * E + coeff[1] * E**2 + coeff[2] * E**3
            F = params['csi'] * g * h * v * 4. * np.pi * h / eV
            if F.shape == ():
                return F[0]
            return F

        return cls(flux, None, np.pi, **params_cleaned)

    @classmethod
    def bins(cls, **params):
        assert 'bins'   in params.keys(), 'Please set the bins in eV'
        assert 'fluxes' in params.keys(), 'Please set the flux in each bin (photons/s/cm2)'
        assert len(params['bins']) == len(params['fluxes']) + 1, 'size(bins) must be size(fluxes) + 1'
        params_cleaned = {'bins'  : np.array(params['bins'])  ,
                         'fluxes': np.array(params['fluxes'])  }

        def flux(v, **params):      # erg/s/cm2/Hz
            binsHz = params['bins'] * eV / h
            delta  = ( params['bins'][1:] - params['bins'][:-1] ) * eV/h
            for i in range(len(binsHz)-1):
                if v >= binsHz[i] and v < binsHz[i+1]:
                    return params['fluxes'][i] * h * v / delta[i]
            return 0.

        return cls(flux, None, np.pi, **params_cleaned)

    @classmethod
    def to_bins(cls, spectrum, bins):
        binsHz = np.array(bins) * eV / h
        fluxes = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            fluxes[i] = spectrum.N_cm2s(v1=binsHz[i], v2=binsHz[i+1])
        new_spectrum = cls.bins(bins=bins, fluxes=fluxes)
        new_spectrum.R      = spectrum.R
        new_spectrum.omega  = spectrum.omega
        return new_spectrum


    # ---------------------------------------------------------

    def renormalize(self, renorm):        # multiplicative factor to the sed
        self.renorm = self.renorm * renorm

    def Fv_cm2s(self, v, funct=f_one, *args):
        return self.flux(v, **self.params) * self.scale * self.renorm * funct(v, *args)

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

    def rescale(self, r):                        # place the source at distance r
        self.checkR()
        self.scale *= (self.R / r) ** 2
        self.R = r

    def Nv_s(self, v, funct=f_one, *args):
        self.checkR()
        return self.Nv_cm2s(v, funct, *args) * 4 * np.pi * self.R**2

    def Fv_s(self, v, funct=f_one, *args):
        self.checkR()
        return self.Fv_cm2s(v, funct, *args) * 4 * np.pi * self.R**2

    def Nv_srs(self, v, funct=f_one, *args):
        self.checkR()
        return self.Nv_cm2s(v, funct, *args) * 4 * np.pi * self.R ** 2 / self.omega

    def Fv_srs(self, v, funct=f_one, *args):
        self.checkR()
        return self.Fv_cm2s(v, funct, *args) * 4 * np.pi * self.R ** 2 / self.omega

    def N_s(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        self.checkR()
        return self.N_cm2s(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def L(self, v1, v2, funct=f_one, *args, **kwargs):
        self.checkR()
        return self.F_cm2s(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def N_srs(self, v1, v2, funct=f_one, *args, **kwargs):  # number of photons / s [into a range of frequencies, default = ionizing photons]
        self.checkR()
        return self.N_cm2srs(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def L_sr(self, v1, v2, funct=f_one, *args, **kwargs):
        self.checkR()
        return self.F_cm2srs(v1, v2, funct, *args, **kwargs) * 4 * np.pi * self.R**2

    def checkR(self):
        if self.R is None:
            raise ValueError("Impossible to use this method without giving the radius at which the given flux is measured")
            

def luminosity_star(M, Z=1):  # L in Lsun, M in Msun, Z in Zsun (Tout+1996)
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
    from scipy.optimize import brentq
    def foo(mm, zz):
        return luminosity_star(mm, zz) - L
    if foo(0.1, Z)*foo(100., Z) < 0:
        M = brentq(foo, 0.1, 100., args=(Z))
        return M
    else:
        raise ValueError('Luminosity out of table range')


# ------------------------------------------------------------------------------
# IDEE PER MIGLIORARE QUESTO MODULO
# 4. mettere un help per i parametri
# 5. mettere omega tra i parametri??
# 6. aggiungere average pothon energy in un bin
