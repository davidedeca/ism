import numpy as np
import os, sys
import matplotlib.pyplot as plt
import scipy.integrate as integ

sys.path.append('../..')
from spectra import Spectrum
from utils.constants import *
import GnedinCooling.gnedincooling as gc
import GnedinCooling.cross_section as cs

evh = eV/h

gc_path = os.path.dirname(os.path.abspath(__file__))
gc.frtinitcf(0, gc_path+'/GnedinCooling/cf_table.I2.dat')

e_ion = {'HI': ryd, 'HeI': 24.5874 * eV, 'CVI': 36*ryd }
nZe   = {'HI': (1, 1), 'HeI': (2, 2), 'CVI': (6, 1) }

def a_Verner(nZ, ne, sh, v):        #photoionization cross section (Verner1996)
    return 1e6 * 1e-24 * cs.phfit2(nZ, ne, sh, h*v/eV)

def sigma(v, spec):
    nZ, ne = nZe[spec]
    return a_Verner(nZ, ne, 1., v)

def rate(spectrum, spec, NH):
    def function(v, spec, NH):
        return sigma(v, spec) * np.exp(- sigma(v, spec) * NH)  # add dust absorption
    eion = e_ion[spec]
    points = np.logspace(np.log10(1.01*eion / h), np.log10(999.*eV/h), 30)
    rrate = spectrum.N_cm2s(eion/h, 1000.*eV/h, function, *(spec, NH), points=points)
    return rrate

def rateH2(spectrum, NH):
    return 1.38e9 * spectrum.Fv_cm2srs(12.87*evh)  #add dust absorption


def cooling(T, n, Z, spectrum, NH=0):
    Plw = rateH2(spectrum, NH)
    Ph1 = rate(spectrum, 'HI' , NH)
    Pg1 = rate(spectrum, 'HeI', NH)
    Pc6 = rate(spectrum, 'CVI', NH)
    return gc.frtgetcf_cool(T, n, Z, Plw, Ph1, Pg1, Pc6)


def heating(T, n, Z, spectrum, NH=0):
    Plw = rateH2(spectrum, NH)
    Ph1 = rate(spectrum, 'HI' , NH)
    Pg1 = rate(spectrum, 'HeI', NH)
    Pc6 = rate(spectrum, 'CVI', NH)
    return gc.frtgetcf_heat(T, n, Z, Plw, Ph1, Pg1, Pc6)
