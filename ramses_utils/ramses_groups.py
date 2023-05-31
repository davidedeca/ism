import sys
import numpy as np
import scipy.integrate as int

sys.path.append('..')
from utils.constants import *
import spectra as spectra
import GnedinCooling.cross_section as cs


default_bins = [13.6, 24.59, 54.52, 1000.]
default_bins_krome = [0.7542, 2.65, 6., 11.2, 13.6, 14.159, 15.4, 24.59, 30, 54.42, 1000.0]


def list_params():
    print('bins    >  array containing photon bin delimiters')
    print('krome   >  True of False wheter krome is used or not')
    print('source  >  point source or surface')
    print('spectr  >  for the moment, star and flat are implemented')
    print('keyworld parameters:')
    print('  \'star\': L  >  bolometric luminosity (in Lsun units)')
    print('        : R  >  distance from the star')
    print('  \'flat\': G  > integrated flux within v1 and v2')
    print('        :    v1  >  lower frequency limit of the flat spectrum (in eV)')
    print('        :    v2  >  upper frequency limit of the flat spectrum (in eV)')

def ramsesparameters(bins=default_bins, krome=False, source='point', spectr='star', **params):


    S = None
    if spectr == 'star':
        S = spectra.Spectrum.star(L=params['L'])
        if source == 'surface': S.rescale(params['R'])
    if spectr == 'quasar':
        S = spectra.QuasarSpectrum(Lbol=params['L'])
        if source == 'surface': S.rescale(params['R'])
    elif spectr == 'flat':
        S = spectra.FlatSpectrum(G=params['G'], v1=params['v1'], v2=params['v2'])
    elif spectr == 'draine':
        S = spectra.DraineSpectrum(csi=params['csi'])


    v = [e * eV / h for e in bins]
    Nphot = np.zeros(len(v) - 1)
    F = np.zeros(len(v) - 1)

    for i in range(len(v) - 1):
        if source == 'point':
            Nphot[i] = S.N_s(v[i], v[i + 1])
            F[i] = S.L(v[i], v[i + 1])
        elif source == 'surface':
            Nphot[i] = S.N_cm2s(v[i], v[i + 1])
            F[i] = S.F_cm2s(v[i], v[i + 1])

    print('rt_n_source =', ', '.join(map(str, Nphot)))

    print('\n&RT_GROUPS')
    print('groupL0 = ' + ', '.join(map(str, bins[0:len(bins) - 1])))
    print('groupL1 = ' + ', '.join(map(str, bins[1:len(bins)])))

    if krome is False:

        def a(v, nZ, ne, sh):  # photoionization cross section (Verner1996)
            return 1e6 * 1e-24 * cs.phfit2(nZ, ne, sh, h * v / eV)

        aHI = np.zeros(len(v) - 1)
        aHeI = np.zeros(len(v) - 1)
        aHeII = np.zeros(len(v) - 1)

        for i in range(len(v) - 1):
            if bins[i] < 13.6:
                aHI[i] = 0
            else:
                aHI[i] = int.quad(a, v[i], v[i + 1], args=(1, 1, 1))[0]
            aHI[i] = aHI[i] / (v[i + 1] - v[i])

        for i in range(len(v) - 1):
            if bins[i] < 24.59:
                aHeI[i] = 0
            else:
                aHeI[i] = int.quad(a, v[i], v[i + 1], args=(2, 2, 1))[0]
            aHeI[i] = aHeI[i] / (v[i + 1] - v[i])

        for i in range(len(v) - 1):
            if bins[i] < 54.42:
                aHeII[i] = 0
            else:
                aHeII[i] = int.quad(a, v[i], v[i + 1], args=(2, 1, 1))[0]
            aHeII[i] = aHeII[i] / (v[i + 1] - v[i])

        for i in range(len(v) - 1):
            print('group_csn(', i + 1, ',:) = ' + str(aHI[i]) + ', ' + str(aHeI[i]) + ', ' + str(aHeII[i]))

        def aw(v, nZ, ne, sh):
            return a(v, nZ, ne, sh) * S.Fv_s(v)

        aHI = np.zeros(len(v) - 1)
        aHeI = np.zeros(len(v) - 1)
        aHeII = np.zeros(len(v) - 1)

        for i in range(len(v) - 1):
            if bins[i] < 13.6:
                aHI[i] = 0
            else:
                aHI[i] = int.quad(aw, v[i], v[i + 1], args=(1, 1, 1))[0]
            aHI[i] = aHI[i] / F[i]

        for i in range(len(v) - 1):
            if bins[i] < 24.59:
                aHeI[i] = 0
            else:
                aHeI[i] = int.quad(aw, v[i], v[i + 1], args=(2, 2, 1))[0]
            aHeI[i] = aHeI[i] / F[i]

        for i in range(len(v) - 1):
            if bins[i] < 54.42:
                aHeII[i] = 0
            else:
                aHeII[i] = int.quad(aw, v[i], v[i + 1], args=(2, 1, 1))[0]
            aHeII[i] = aHeII[i] / F[i]

        for i in range(len(v) - 1):
            print('group_cse(' + str(i + 1) + ',:) =' + str(aHI[i]) + ', ' + str(aHeI[i]) + ', ' + str(aHeII[i]))

    egy = [F[i] / Nphot[i] / eV for i in range(len(F))]
    print("group_egy = " + ', '.join(map(str, egy)))
    print('/')

# --------------------------------------------------------------------------------------

