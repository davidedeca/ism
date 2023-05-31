import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.constants import *

mass_ion = {
    'HI'    : 1.67353251819e-24,
    'e'     : 9.10938188e-28   ,
    'HII'   : 1.67262158e-24   ,
    'HeI'   : 6.69206503638e-24,
    'HeII'  : 6.69115409819e-24,
    'HeIII' : 6.69024316e-24   ,
    'H-'    : 1.67444345638e-24,
    'H2'    : 3.34706503638e-24,
    'H2+'   : 3.34615409819e-24
            }

#default_fields = ['lev', 'x', 'n', 'v', 'P_nt', 'P', 'xHI', 'xH2', 'xHII',
#                  'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10']

default_fields = ['lev', 'x', 'n', 'v', 'P', 
                  'xHI', 'xe', 'xHII','xHeI','xHeII','xHeIII','xH-', 'xH2', 'xH2+',
                  'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10'] 

unit_d = 1.50222845999999e-20
unit_v = pc / Myr
unit_P = unit_d * unit_v**2
unit_T = unit_P *mp / unit_d / kB
string = '==============================='

timestep = 0.01 # Myr

######### Define functions for physical quantities

def f_default(data, fields, var):
    return data[:, fields.index(var)]

def f_n(data, fields, var):
    var_x = 'x' + var[1::]
    return data[:, fields.index(var_x)] * data[:, fields.index('n')]

def f_P_tot(data, fields, var):
    return data[:, fields.index('P')] + data[:, fields.index('P_nt')]

def f_mu(data, fields, var):   # mu function to be written!
    out = 0
    for key in mass_ion.keys():
        out += data[:, fields.index('x'+key)] / mass_ion[key]
    out  = 1. / out / mass_ion['HI']
    return out 

def f_T(data, fields, var):
    P  = data[:, fields.index('P')]
    n  = data[:, fields.index('n')]
    mu = f_mu(data, fields, var)
    return P / n * mu 

######### Define dictionaries

list_n = ['n', 'nHI', 'nH2', 'nHII']
list_P = ['P', 'P_nt', 'P_tot']
list_x = ['xHI', 'xHII', 'xH2']
list_u = ['u' + str(j) for j in range(1,11)]

dict_var_base = {
    'lev'  : 'adim' ,
    'x'    : 'len'  ,
    'v'    : 'v'    ,
    'T'    : 'T'
    }

for n in list_n:
    dict_var_base[n] = 'n'

for p in list_P:
    dict_var_base[p] = 'P'

for xx in list_x:
    dict_var_base[xx] = 'adim'

for u in list_u:
    dict_var_base[u] = 'flux'

dict_norm_and_unit = {
    'adim' : [ 1                , ''        ],
    'len'  : [ 1                , 'pc'      ],
    'n'    : [ unit_d/mp        , 'cm-3'    ],
    'v'    : [ unit_v/1.e5      , 'Km/s'    ],
    'P'    : [ unit_P           , 'Ba'      ],
    'flux' : [ unit_v           , 'cm-2s-1' ],
    'T'    : [ unit_T           , 'K'       ]
     }

dict_function = {
    'P_tot': f_P_tot  ,
    'T'    : f_T
}

for var in ['nHI', 'nH2', 'nHII']:
    dict_function[var] = f_n

for var in default_fields:
    dict_function[var] = f_default


######### Define module functions

def log_to_data(filename, fields = default_fields):   # outdates, since now output are already printed to file

    log = open(filename, 'r')
    write = 0

    output_number = 1
    output_filename = ''
    output_file = None

    for line in log:

        if string in line and write == 0:
            write = 1
            output_filename = 'output_' + str(output_number).zfill(5)
            output_file = open(output_filename, 'w+')
            log.next()
            log.next()
            continue

        elif string in line:
            write = 0
            output_file.close()
            output_number += 1

        if write == 1:
            output_file.write(line)


def plot(var, num, fields=default_fields, log=0, path='.', show=True, store=False):

    norm, unit = dict_norm_and_unit[dict_var_base[var]]
    if (isinstance(num, int)): num = [num]

    plt.figure()

    for n in num:

        filename = path + '/output_' + str(n).zfill(5)
        data = np.loadtxt(filename)
        time = timestep * n
        label = 't = ' + str(time) + ' Myr'

        x = data[:, 1]

        y = dict_function[var](data, fields, var) * norm

        if log == 1: y = np.log10(y)

        plt.plot(x,y,label=label)

    plt.xlabel('x [pc]')
    plt.ylabel(var + ' [' + unit + ']')

    plt.legend()
    if show:
    	plt.show(block=False)
    else:
    	plt.close()

    if store:
    	return x, y
    else:
	return

def plot_abundances(num, fields = default_fields, log = 0, yrange=[None,None]):

    norm, unit = dict_norm_and_unit[dict_var_base['n']]
    if (isinstance(num, int)): num = [num]
    count = 0

    for n in num:

        count += 1
        print('Plot ' + count + ' of ' + str(len(num)))
        filename = 'output_' + str(n).zfill(5)
        data = np.loadtxt(filename)
        time = timestep * n
        title = 't = ' + str(time) + ' Myr'

        x = data[:, 1]
        n = dict_function['n'](data, fields, 'n') * norm
        HI = dict_function['nHI'](data, fields, 'nHI') * norm
        H2 = dict_function['nH2'](data, fields, 'nH2') * norm
        HII = dict_function['nHII'](data, fields, 'nHII') * norm

        if log == 1:
           n = np.log10(n)
           HI = np.log10(HI)
           H2 = np.log10(H2)
           HII = np.log10(HII)

        plt.figure()

        plt.plot(x, n, label=r'$n_{tot}$')
        plt.plot(x, HI, label=r'$n_{HI}$')
        plt.plot(x, H2, label=r'$n_{H2}$')
        plt.plot(x, HII, label=r'$n_{HII}$')

        plt.ylim(ymin=yrange[0], ymax=yrange[1])
        plt.title(title)
        plt.xlabel('x [pc]')
        plt.ylabel('n' + ' [' + unit + ']')
        plt.legend()
        plt.savefig('plot_abundances_' + str(count).zfill(5))
        plt.close()


def cell_mass(species, num, fields=default_fields):

    var = None
    if    species == 'HII' :  var = 'nHII'
    elif  species == 'H2'  :  var = 'nH2'
    elif  species == 'HI'  :  var = 'nHI'

    norm, unit = dict_norm_and_unit[dict_var_base['n']]
    filename = 'output_' + str(num).zfill(5)
    data = np.loadtxt(filename)
    time = timestep * num
    x = data[:, 1]
    n_var = dict_function[var](data, fields, var) * norm
    lev = dict_function['lev'](data, fields, 'lev')
    dx =  np.array([ x[-1] / 2**l for l in lev])
    M = np.sum( n_var * dx**3 ) * mp * pc**3

    return M
