import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import ctypes
from pykrome_class import PyKROME

from utils.constants import *
import utils.plotstyle as ps
from utils.io import go_up

n_baryons = np.array([1., 0., 1., 4., 4., 4., 1., 2., 2.])

x_default = np.array([  0.92386267,      #H    0
                        0.00013688914,   #E    1
                        0.00013688475,   #H+   2
                        0.076000003,     #HE   3
                        9.3646368e-28,   #HE+  4
                        1.0910885e-53,   #HE++ 5
                        8.4043915e-12,   #H-   6
                        3.8662947e-07,   #H2   7
                        4.3502284e-14])  #H2+  8

baryons_number = np.array([1., 0., 1., 4., 4., 4., 1., 2., 2.])

bins_default = np.array([0.7542, 2.65, 6.0, 11.2, 13.6, 14.159, 15.4, 24.59, 30, 54.42, 1000.])

ion_tag = ['H', 'e', 'H+', 'He', 'He+', 'He++', 'H-', 'H2', 'H2+']

default_krome_path = os.path.join(go_up(__file__, 2), 'krome')

def krome_init(path=default_krome_path):
    libkrome = os.path.join(path, 'libkrome.so')
    print(libkrome)
    assert os.path.isfile(libkrome), 'krome not found in path ' + libkrome
    print('Using: ' + libkrome)
    pk[0] = PyKROME(path_to_lib=path)
    pk[0].lib.krome_init()
    
# automatic krome initialization when pykrome module is loaded:
pk = [None]
krome_init()

cool_function = {'H2'       : (lambda c, h: c[pk[0].krome_idx_cool_h2])     ,
                 'atomic'   : (lambda c, h: c[pk[0].krome_idx_cool_atomic]) ,
                 'Compton'  : (lambda c, h: c[pk[0].krome_idx_cool_compton]),
                 'free-free': (lambda c, h: c[pk[0].krome_idx_cool_ff])     ,
                 'metals'   : (lambda c, h: c[pk[0].krome_idx_cool_zcie]
                                           -h[pk[0].krome_idx_heat_zcie])   ,
                 'cooling_GH' : (lambda c, h: c[pk[0].krome_idx_cool_gh])   }

heat_function = {'photo'       : (lambda c, h: h[pk[0].krome_idx_heat_photo]),
                 'cosmic rays' : (lambda c, h: h[pk[0].krome_idx_heat_cr])   ,
                 'dust'        : (lambda c, h: h[pk[0].krome_idx_heat_dust]) ,
                 'chem'        : (lambda c, h: h[pk[0].krome_idx_heat_chem]) ,
                 'heating_GH'  : (lambda c, h: h[pk[0].krome_idx_heat_gh])   }

cool_heat_function = {'heating': heat_function ,
                      'cooling': cool_function }


def swap_ions(vec, flag='to_pykrome'):           #pykrome has different ions indexes!
    assert flag in ['to_pykrome', 'from_pykrome']
    if flag == 'to_pykrome':
        index = [1, 6, 0, 3, 7, 2, 4, 8, 5]
    else:
        index = [2, 0, 5, 3, 6, 8, 1, 4, 7]
    return vec[index]


def _krome_setup(kromepath='../krome', **params):
    params.setdefault('x'      , x_default)
    params.setdefault('zred'   , 0.       )
    params.setdefault('crate'  , 6.e-17   )
    params.setdefault('H2_diss', 1.38e9   )
    params.setdefault('Z'      , 1.       )
    params.setdefault('bins'   , bins_default)
    params.setdefault('F'      , np.zeros(10)) #photons / cm2 / s
    params['bins'] = np.array(params['bins'])
    params['F'] = np.array(params['F'])
    params['x'] = swap_ions(np.array(params['x']), 'to_pykrome')
    Tcmb = 2.725*(params['zred'] + 1.)
    params.setdefault('Tfloor'  , 0.      )
    pk[0].lib.krome_set_zredshift(params['zred'])  #set redshift
    pk[0].lib.krome_set_tcmb(Tcmb)       #set cmb
    pk[0].lib.krome_set_tfloor(params['Tfloor'])
    pk[0].lib.krome_set_clump(1.)       #set the clumping factor fior H2 formation on dust
    pk[0].lib.krome_set_user_crate(params['crate'])  # cosmic rays rate
    pk[0].lib.krome_set_user_myh2_dissociation(params['H2_diss'])
    pk[0].lib.krome_set_z(params['Z'])
    #pk.lib.krome_set_dust_to_gas(params['Z'])

    pk[0].lib.krome_set_photobine_lr(params['bins'][0:-1], params['bins'][1:])
    E_mid   = (params['bins'][0:-1] + params['bins'][1:]) / 2.
    E_delta = (params['bins'][1:] - params['bins'][0:-1])
    flux_for_krome = params['F'] * E_mid * h / eV / E_delta / 4 / np.pi
    pk[0].lib.krome_set_photobinj(flux_for_krome)
    fluxlw = params['F'][3] * 12.87 * h / eV / E_delta[3] / 4 / np.pi
    pk[0].lib.krome_set_user_myfluxlw(fluxlw)
    return params


class cell:

    def __init__(self, n, T, **params):   # here n is the baryon density (rho / mp)
        params = _krome_setup(**params)
        ntot = n / np.dot(np.array(params['x']), baryons_number)  # this is the number of particles
        self.narray = ntot * params['x']                # array with number of particles per species
        self.T = T

    def set_flux(self, Fnew, bins): 
        #bins should be the same as the ones used 
        #in the definition of the cell object
        E_mid   = (bins[0:-1] + bins[1:]) / 2.
        E_delta = bins[1:] - bins[0:-1]
        flux_for_krome = Fnew * E_mid * h / eV / E_delta / 4 / np.pi
        pk[0].lib.krome_set_photobinj(flux_for_krome)
        fluxlw = Fnew[3] * 12.87 * h / eV / E_delta[3] / 4 / np.pi
        pk[0].lib.krome_set_user_myfluxlw(fluxlw)          
        return 
    
    def get(self, var, ion):
        assert var in ['rho', 'n', 'nb', 'xn', 'xm'] # rho, number, rho/mp, number abundance, mass abundance
        if var in ['rho', 'n', 'nb']:
            assert ion in ion_tag + ['tot']
        elif var in ['xn', 'xm']:
            assert ion in ion_tag
        n_sorted = list(swap_ions(self.narray, 'from_pykrome'))
        if ion == 'tot':
            if var == 'rho':
                return np.dot(n_sorted, baryons_number) * mp
            elif var == 'nb':
                return np.dot(n_sorted, baryons_number)
            elif var == 'n':
                return np.sum(n_sorted)
        else:
            index = ion_tag.index(ion)
            if var == 'rho':
                return n_sorted[index] * baryons_number[index] * mp
            elif var == 'nb':
                return n_sorted[index] * baryons_number[index]
            elif var == 'n':
                return n_sorted[index]
            elif var == 'xn':
                return n_sorted[index] / np.sum(n_sorted)
            elif var == 'xm':
                return n_sorted[index] * baryons_number[index] / (np.dot(n_sorted, baryons_number))
        
        
    def nmol(self, ion):
        assert ion in ion_tag
        n_sorted = swap_ions(self.narray, 'from_pykrome')
        return n_sorted[ion_tag.index(ion)]
    
    def ntot(self, ion):
        return self.ntot * params['x']
    
    def print_info(self):
        n_sorted = swap_ions(self.narray, 'from_pykrome')
        for i in range(len(ion_tag)):
            print(ion_tag[i] + '\t' + '{0:1.5e}'.format(n_sorted[i]) + '\tcm-3')
        print('\nntot' + '\t' + '{0:1.5e}'.format(np.sum(n_sorted)) + '\tcm-3')
        print('T' + '\t' + '{0:1.5e}'.format(self.T.value) + '\tK')

    def equilibrium(self, verbose=True):
        self.T = ctypes.c_double(self.T)
        if verbose is True:
            print('\n ============== Initial abundances:')
            self.print_info()
        pk[0].lib.krome_equilibrium(self.narray, self.T)
        if verbose is True:
            print('\n ============== Final abundances:')
            self.print_info()
        self.T = self.T.value

    def evolution(self, t, dt=None, verbose=True):
        self.T = ctypes.c_double(self.T)
        if verbose is True:
            print('\n ============== Initial abundances:')
            self.print_info()
        if dt is None:
            dt = t
        time = dt
        while (time <= t):
            pk[0].lib.krome(self.narray,ctypes.byref(self.T),ctypes.byref(ctypes.c_double(dt)))
            time = time + dt
        if verbose is True:
            print('\n ============== Final abundances:')
            self.print_info()
        self.T = self.T.value

    def evolution_tconst(self, t, dt=None, verbose=True):
        self.T = ctypes.c_double(self.T)
        if verbose is True:
            print('\n ============== Initial abundances:')
            self.print_info()
        if dt is None:
            dt = t
        time = dt
        while (time <= t):
            pk[0].lib.krome_tconst(self.narray,ctypes.byref(self.T),ctypes.byref(ctypes.c_double(dt)))
            time = time + dt
        if verbose is True:
            print('\n ============== Final abundances:')
            self.print_info()
        self.T = self.T.value

def cooling(n, ax=None, plot=False, save_data=False, return_data=False, show_tot=True,
            cool_list=cool_function.keys(), norm=1., data_path='./cooling.txt', **params):
    data = cool_heat('cooling', n, **params)
    if ax is None and not save_data and not return_data:
        plot=True
    ch_list, data_to_plot = plot_cool_heat('cooling', plot, data, ax, show_tot, cool_list, norm)
    if save_data:
        np.savetxt(data_path, data_to_plot)
    if return_data:
        return ch_list, data_to_plot


def heating(n, ax=None, plot=False, save_data=False, return_data=False, show_tot=True,
            heat_list=heat_function.keys(), norm=1., data_path='./heating.txt', **params):
    data = cool_heat('heating', n, **params)
    if ax is None and not save_data and not return_data:
        plot=True
    ch_list, data_to_plot = plot_cool_heat('heating', plot, data, ax, show_tot, heat_list, norm)
    if save_data:
        np.savetxt(data_path, data_to_plot)
    if return_data:
        return ch_list, data_to_plot


def cool_heat(kind, n, **params):

    assert kind in ['heating', 'cooling']

    params = krome_setup(**params)
    lnT = np.linspace(1, 8, 100)
    T = 10**lnT
    data = np.zeros((len(T), 1+len(cool_heat_function[kind])))
    data[:, 0]  = T
    for i in range(len(T)):
        array_cool = np.zeros(pk[0].krome_ncools)
        array_heat = np.zeros(pk[0].krome_nheats)
        array      = np.zeros(len(cool_heat_function[kind]))
        ntot = n / np.dot(np.array(params['x']), baryons_number)
        neq = ntot * params['x']
        pk[0].lib.krome_equilibrium(neq, ctypes.c_double(T[i]))
        pk[0].lib.krome_get_cooling_array(neq, T[i], array_cool) #krome gives cooling function per unit particle
        pk[0].lib.krome_get_heating_array(neq, T[i], array_heat)
        array = []
        for tag in cool_heat_function[kind].keys():
            array.append(cool_heat_function[kind][tag](array_cool, array_heat))
        data[i, 1:] = np.array(array) / np.sum(neq)

    return data


def plot_cool_heat(kind, plot, data, ax, show_tot, ch_list, norm):

    assert kind in ['heating', 'cooling']
    assert set(ch_list) <= set(cool_heat_function[kind].keys())

    show_plot=False

    if plot==True:
        print('Making plot...')
        show_plot = True
        fig, ax = plt.subplots()
    elif isinstance(ax, Axes):
        print('Adding plot to axes...')
    else:
        print('No axes to add the plot')
        ax = False

    data_to_plot = np.zeros((len(data[:, 0]), len(ch_list)+1))
    data_to_plot[:, 0] = data[:, 0]
    array_tot = np.zeros_like(data[:, 0])
    for tag in ch_list:
        index = cool_heat_function[kind].keys().index(tag)+1
        array_plot = norm * data[:, index]
        if ax:
            ax.plot(data[:, 0], array_plot, label=tag)
        data_to_plot[:, ch_list.index(tag)+1] = array_plot
        array_tot += array_plot

    if ax and show_tot:
        ax.plot(data[:, 0], array_tot, label='tot', linewidth=2, color='k', zorder=20)

    if ax:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin=10, xmax=1e8)
        ax.set_ylim(ymin=1.e-25, ymax=5.*np.max(array_tot))
        ax.set_xlabel('$T$' + ps.brack(ps.K))
        letter = {'heating': '$\Lambda$', 'cooling': '$\Gamma$'}
        ax.set_ylabel(letter[kind] + ps.brack('$\mathrm{erg}\,\mathrm{s}^{-1}$'))
        ax.legend()

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ['T'] + ch_list, np.array(data_to_plot)
