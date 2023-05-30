import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import ctypes
from pykrome_class import PyKROME

sys.path.append('..')
from utils.constants import *
import utils.plotstyle as ps

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

n_baryons = np.array([1., 0., 1., 4., 4., 4., 1., 2., 2.])

bins_default = np.array([0.7542, 2.65, 6.0, 11.2, 13.6, 14.159, 15.4, 24.59, 30, 54.42, 1000.])

ion_tag = ['H', 'e', 'H+', 'He', 'He+', 'He++', 'H-', 'H2', 'H2+']

pykrome_path = os.path.dirname(os.path.abspath(__file__))
pk = PyKROME(path_to_lib=pykrome_path)
pk.lib.krome_init()

cool_function = {'H2'       : (lambda c, h: c[pk.krome_idx_cool_h2])     ,
                 'atomic'   : (lambda c, h: c[pk.krome_idx_cool_atomic]) ,
                 'Compton'  : (lambda c, h: c[pk.krome_idx_cool_compton]),
                 'free-free': (lambda c, h: c[pk.krome_idx_cool_ff])     ,
                 'metals'   : (lambda c, h: c[pk.krome_idx_cool_zcie]
                                           -h[pk.krome_idx_heat_zcie])
		}

heat_function = {'photo'       : (lambda c, h: h[pk.krome_idx_heat_photo]),
                 'cosmic rays' : (lambda c, h: h[pk.krome_idx_heat_cr])   ,
                 'dust'        : (lambda c, h: h[pk.krome_idx_heat_dust]) ,
                 'chem'        : (lambda c, h: h[pk.krome_idx_heat_chem])  }

cool_heat_function = {'heating': heat_function ,
                      'cooling': cool_function }


def swap_ions(vec, flag='to_pykrome'):           #pykrome has different ions indexes!
    assert flag in ['to_pykrome', 'from_pykrome']
    if flag == 'to_pykrome':
        index = [1, 6, 0, 3, 7, 2, 4, 8, 5]
    else:
        index = [2, 0, 5, 3, 6, 8, 1, 4, 7]
    return vec[index]


def krome_setup(**params):
    params.setdefault('x'      , x_default)
    params.setdefault('zred'   , 0.       )
    params.setdefault('crate'  , 6.e-17   )
    params.setdefault('H2_diss', 1.38e9   )
    params.setdefault('Z'      , 1.       )
    params.setdefault('bins'   , bins_default)
    params.setdefault('F'      , np.zeros(10))
    params['bins'] = np.array(params['bins'])
    params['F'] = np.array(params['F'])
    params['x'] = swap_ions(np.array(params['x']), 'to_pykrome')
    Tcmb = 2.83*(params['zred'] + 1.)
    pk.lib.krome_set_zredshift(params['zred'])  #set redshift
    pk.lib.krome_set_tcmb(Tcmb)       #set cmb
    pk.lib.krome_set_tfloor(Tcmb)
    pk.lib.krome_set_clump(1.)       #set the clumping factor fior H2 formation on dust
    pk.lib.krome_set_user_crate(params['crate'])  # cosmic rays rate
    pk.lib.krome_set_user_myh2_dissociation(params['H2_diss'])
    pk.lib.krome_set_z(params['Z'])
    #pk.lib.krome_set_dust_to_gas(params['Z'])

    pk.lib.krome_set_photobine_lr(params['bins'][0:-1], params['bins'][1:])
    E_mid   = (params['bins'][0:-1] + params['bins'][1:]) / 2.
    E_delta = (params['bins'][1:] - params['bins'][0:-1])
    flux_for_krome = params['F'] * E_mid * h / eV / E_delta / 4 / np.pi
    pk.lib.krome_set_photobinj(flux_for_krome)
    fluxlw = params['F'][3] * 12.87 * h / eV / E_delta[3] / 4 / np.pi
    pk.lib.krome_set_user_myfluxlw(fluxlw)
    return params


class cell:

    def __init__(self, n, T, **params):   # n is the baryon density (rho / mp)
        params = krome_setup(**params)
        ntot = n / np.dot(np.array(params['x']), n_baryons) 
        self.n = ntot * params['x']
        self.T = T

    def print_info(self):
        n_to_print = swap_ions(self.n, 'from_pykrome')
        for i in range(len(ion_tag)):
            print ion_tag[i], '\t', '{0:1.5e}'.format(n_to_print[i]), '\tcm-3'
        print '\nntot', '\t', '{0:1.5e}'.format(np.sum(n_to_print)), '\tcm-3'
        print 'T', '\t', '{0:1.5e}'.format(self.T.value), '\tK'

    def equilibrium(self, verbose=True):
        self.T = ctypes.c_double(self.T)
        if verbose is True:
            print '\n ============== Initial abundances:'
            self.print_info()
        pk.lib.krome_equilibrium(self.n, self.T)
        if verbose is True:
            print '\n ============== Final abundances:'
            self.print_info()
        self.T = self.T.value
    
    def evolution(self, t, dt=None, verbose=True):
        self.T = ctypes.c_double(self.T)
        if verbose is True:
            print '\n ============== Initial abundances:'
            self.print_info()
        if dt is None:
            dt = t
        time = dt
        while (time <= t):
            pk.lib.krome(self.n,ctypes.byref(self.T),ctypes.byref(ctypes.c_double(dt)))
            time = time + dt
        if verbose is True:
            print '\n ============== Final abundances:'
            self.print_info()
        self.T = self.T.value


def cooling(n, ax='show', save_data=False, return_data=False, show_tot=True,
            cool_list=cool_function.keys(), norm=1., data_path='./cooling.txt', **params):
    data = cool_heat('cooling', n, **params)
    data_to_plot = plot_cool_heat('cooling', data, ax, show_tot, cool_list, norm)
    if save_data:
        np.savetxt(data_path, data_to_plot)
    if return_data:
        return data_to_plot


def heating(n, ax='show', save_data=False, return_data=False, show_tot=True,
            heat_list=heat_function.keys(), norm=1., data_path='./heating.txt', **params):
    data = cool_heat('heating', n, **params)
    data_to_plot = plot_cool_heat('heating', data, ax, show_tot, heat_list, norm)
    if save_data:
        np.savetxt(data_path, data_to_plot)
    if return_data:
        return data_to_plot


def cool_heat(kind, n, **params):

    assert kind in ['heating', 'cooling']

    params = krome_setup(**params)
    lnT = np.linspace(1, 8, 100)
    T = 10**lnT
    data = np.zeros((len(T), 1+len(cool_heat_function[kind])))
    data[:, 0]  = T
    for i in range(len(T)):
        array_cool = np.zeros(pk.krome_ncools)
        array_heat = np.zeros(pk.krome_nheats)
        array      = np.zeros(len(cool_heat_function[kind]))
        ntot = n / np.dot(np.array(params['x']), n_baryons) 
        neq = ntot * params['x']        
        pk.lib.krome_equilibrium(neq, ctypes.c_double(T[i]))
        pk.lib.krome_get_cooling_array(neq, T[i], array_cool) #krome gives cooling function per unit particle
        pk.lib.krome_get_heating_array(neq, T[i], array_heat)
        array = []
        for tag in cool_heat_function[kind].keys():
            array.append(cool_heat_function[kind][tag](array_cool, array_heat))
        data[i, 1:] = np.array(array) / np.sum(neq)

    return data


def plot_cool_heat(kind, data, ax, show_tot, ch_list, norm):

    assert kind in ['heating', 'cooling']
    assert set(ch_list) <= set(cool_heat_function[kind].keys())

    show_plot = False

    if ax == 'show':
        print 'Making plot...'
        show_plot = True
        fig, ax = plt.subplots()
    elif isinstance(ax, Axes):
        print 'Adding plot to axes...'
    else:
        print 'No axes to add the plot'
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

    return np.array(data_to_plot)
