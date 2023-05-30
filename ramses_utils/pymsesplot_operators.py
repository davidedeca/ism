import numpy as np
from pymses.analysis.visualization import ScalarOperator,FractionOperator,MaxLevelOperator
from pymses.utils import constants as C

NDIM = 3
NENER = 1
NRAD = 10
NION = 9

dict_axis = {
    'x'    : np.array([1, 0, 0]),
    'y'    : np.array([0, 1, 0]),
    'z'    : np.array([0, 0, 1])
            }

krome_ion = {
    'HI'    : [ 'ion1', 1.67353251819e-24 ],
    'e'     : [ 'ion2', 9.10938188e-28    ],
    'HII'   : [ 'ion3', 1.67262158e-24    ],
    'HeI'   : [ 'ion4', 6.69206503638e-24 ],
    'HeII'  : [ 'ion5', 6.69115409819e-24 ],
    'HeIII' : [ 'ion6', 6.69024316e-24    ],
    'H-'    : [ 'ion7', 1.67444345638e-24 ],
    'H2'    : [ 'ion8', 3.34706503638e-24 ],
    'H2+'   : [ 'ion9', 3.34615409819e-24 ]
            }


####### define lists (i.e. groups of variables with same unit and norm)

list_n    = ['n'] + ['n' + el for el in krome_ion.keys()]

list_P    = ['Pth', 'P']

if NENER==1 :
    list_P += ['Pnt']
else:
    list_P += ['Pnt' + str(i+1) for i in range(NENER)]

list_vel = ['vel', 'cs'] + ['vel' + ax for ax in dict_axis.keys()]
list_mom = ['mom'] + ['mom' + ax for ax in dict_axis.keys()]
list_g   = ['g']   + ['g'   + ax for ax in dict_axis.keys()]

list_x    = ['x' + el for el in krome_ion.keys()] + ['fHII']

list_N    = ['N'] + ['N' + el for el in krome_ion.keys()]

list_u  = ['utot', 'uion'] + ['u' + str(j) for j in range(1,NRAD+1)]

list_flux = []

for ax in dict_axis.keys():
    for i_rad in range(1,NRAD+1):
        list_flux.append('F' + ax + str(i_rad))

list_flux_r = ['F' + str(i_rad) + '_r' for i_rad in range(1,NRAD+1)]

list_M = ['M'] + ['M' + el for el in krome_ion.keys()]

list_other = ['rho', 'phi', 'Z', 'mu', 'T', 'T_mu', 'G0', 'mass', 'photons', 'Uion', 'lev']

list_var = list_n + list_P + list_vel + list_mom + list_g \
           + list_x + list_N + list_u + list_flux + list_M + list_other

list_extensive_var = list_N + list_M + ['photons']    ## to check variable which cannot
                                                       ## be used for slices or ray projections

####### define dictionaries, associationg evey variable with a 'base variable'

dict_var_base = {}

for n in list_n:
    dict_var_base[n] = 'n'

for p in list_P:
    dict_var_base[p] = 'P'

for v in list_vel:
    dict_var_base[v] = 'vel'

for mom in list_mom:
    dict_var_base[mom] = 'mom'

for g in list_g:
    dict_var_base[g] = 'g'

for xx in list_x:
    dict_var_base[xx] = 'x'

for NN in list_N:
    dict_var_base[NN] = 'N'

for u in list_u:
    dict_var_base[u] = 'flux'

for F in list_flux + list_flux_r:
    dict_var_base[F] = 'flux'

for M in list_M:
    dict_var_base[M] = 'M'

for var in list_other:
    dict_var_base[var] = var


def get_unit(var):

    var_base = dict_var_base[var]

    dict_unit = {
         'n'      : r'${\rm cm}^{-3}$'
        ,'P'      : r'${\rm Ba}$'
        ,'vel'    : r'${\rm km}/{\rm s}$'
        ,'mom'    : r'$\mathrm{km}/\mathrm{s}\,\mathrm{cm}^3$'
        ,'g'      : r'$\mathrm{cm}/\mathrm{s}^2$'
        ,'x'      : r' '
        ,'N'      : r'$\mathrm{cm}^{-2}$'
        ,'flux'   : r'$\mathrm{photons}\,{\rm cm}^{-2}{\rm s}^{-1}$'
        ,'rho'    : r'${\rm g}/{\rm cm}^3$'
        ,'phi'    : r'$\mathrm{cm}^2/\mathrm{s}^2$'
        ,'Z'      : r'${\rm Z}_{\odot}$'
        ,'mu'     : r' '
        ,'T'      : r'${\rm K}$'
        ,'T_mu'   : r'${\rm K}\mu$'
        ,'G0'     : r'${\rm Habing}$'
        ,'M'      : r'${\rm M}_\odot$'
        ,'photons': r'${\rm cm}^{-3}$'
        ,'lev'    : r' '
        ,'Uion'   : r' '
        }

    return dict_unit[var_base]


def get_norm(var, d):

    var_base = dict_var_base[var]

    dict_norm = {
         'n'      : d.info["unit_density"].express(C.g_cc)/C.mH.express(C.g)
        ,'P'      : d.info['unit_pressure'].express(C.barye)
        ,'vel'    : d.info["unit_velocity"].express(C.km/C.s)
        ,'mom'    : d.info['unit_density'].express(C.g_cc)/C.mH.express(C.g) \
                    * d.info['unit_velocity'].express(C.km / C.s)
        ,'g'      : d.info["unit_velocity"].express(C.cm/C.s) \
                    / d.info["unit_time"].express(C.s)
        ,'x'      : 1.
        ,'N'      : d.info["unit_density"].express(C.g_cc) / C.mH.express(C.g) \
                    * d.info["unit_length"].express(C.cm)
        ,'flux'   : d.info['unit_velocity'].express(C.cm/C.s)
        ,'rho'    : d.info['unit_density'].express(C.g_cc)
        ,'phi'    : d.info["unit_velocity"].express(C.cm/C.s)**2
        ,'Z'      : 1./0.02
        ,'mu'     : 1.
        ,'T'      : d.info["unit_temperature"].express(C.K)
        ,'T_mu'   : d.info["unit_temperature"].express(C.K)
        ,'G0'     : d.info['unit_velocity'].express(C.cm/C.s) / 1.6e-3
        ,'M'      : d.info['unit_density'].express(C.g_cc) \
                    * d.info["unit_length"].express(C.cm)**3 / C.Msun.express(C.g)
        ,'photons': d.info['unit_velocity'].express(C.cm/C.s) \
                           * d.info['unit_length'].express(C.cm)**3 \
                           / C.c.express(C.cm/C.s)
        ,'lev'    : 1.
        ,'Uion'   : d.info['unit_velocity'].express(C.cm/C.s) \
                           / C.c.express(C.cm/C.s) \
                           / (d.info["unit_density"].express(C.g_cc)/C.mH.express(C.g))
        }

    return dict_norm[var_base]


def get_function(var):

    if (var == "rho" or var == 'n'):           # n = rho / mp
        op = lambda dset: dset["rho"]
    elif var in list_n:                        # n_i = rho_i / mp (so this is not really the number density of a particular ion..)
        op = lambda dset: dset["rho"] * dset[krome_ion[var.lstrip("n")][0]]
    elif var == "P":
        if NENER==0: op = lambda dset: dset['P']
        elif NENER==1: op = lambda dset: dset['P'] + dset['P_nt']
        else:
           op = lambda dset: dset['P'] + np.sum([dset["P_nt" + str(j)] for j in range(1, NENER+1)])
    elif var == 'Pth':
        op = lambda dset: dset['P']
    elif var in list_P:
        op = lambda dset: dset["P_nt" + var.lstrip("P_nt")]
    elif var == "vel" or var=="g":
        op = lambda dset: np.sqrt(np.sum(dset[var]**2 ,axis = 1 ))
    elif var == "cs":
        op = lambda dset: np.sqrt(dset['P'] / dset['rho'])
    elif var in list_vel or var in list_g:
        var_base = dict_var_base[var]
        op = lambda dset: np.dot(dset[var_base], dict_axis[var.lstrip(var_base)])
    elif var == "mom":
        op = lambda dset: dset["rho"]*np.sqrt(np.sum(dset["vel"]**2 ,axis = 1 ))
    elif var in list_mom:
        op = lambda dset: dset["rho"]*np.dot(dset["vel"], dict_axis[var.lstrip("mom")])
    elif var == 'fHII':
        op = lambda dset: function_fHII(dset)
    elif var in list_x:            # should be mass fractions
        op = lambda dset: dset[krome_ion[var.lstrip("x")][0]]
    elif var == 'N':
        op = lambda dset: dset["rho"] * dset.get_sizes()
    elif var in list_N:
        op = lambda dset: dset["rho"] * dset[krome_ion[var.lstrip("N")][0]] * dset.get_sizes()
    elif var == "utot":
        op = lambda dset: function_utot(dset)
    elif var == "uion":
        op = lambda dset: function_uion(dset)
    elif var in list_u:
        op = lambda dset: dset["rad_density" + var.lstrip("u")]
    elif var in list_flux:
        op = lambda dset: np.dot(dset['rad_flux'+var[2:]], dict_axis[var[1:2]])
    elif var == "Z":
        op = lambda dset: dset['Z']
    elif var =='phi':
        op = lambda dset: dset['phi']
    elif var == 'mu':
        op = lambda dset: function_mu(dset)
    elif var == 'T':
        op = lambda dset: function_T(dset)
    elif var == 'T_mu':
        op = lambda dset: function_T_over_mu(dset)
    elif var == 'G0':
        op = lambda dset: function_G0(dset)
    elif var == 'M':
        op = lambda dset: dset['rho'] * dset.get_sizes()**3
    elif var in list_M:
        op = lambda dset: dset['rho'] * dset[krome_ion[var.lstrip("M")][0]] * dset.get_sizes()**3
    elif var == 'photons':
        op = lambda dset: function_utot(dset) * dset.get_sizes()**3
    elif var == 'lev':
        return MaxLevelOperator()
    elif var == 'Uion':
        op = lambda dset: function_Uion(dset)

    return op


def get_var_to_load(var):

    if (var == "rho" or var == 'n' or var=="M"):
        var_to_load = ["rho"]
    elif var in list_n:
        var_to_load = ["rho", krome_ion[var.lstrip("n")][0]]
    elif var == "P":
        var_to_load = ["P"]
        if   NENER==1 : var_to_load += ["P_nt"]
        elif NENER>1  : var_to_load += ["P_nt" + str(j) for j in range(1,NENER+1)]
    elif var == "Pth":
        var_to_load = ["P"]
    elif var in list_P:
        var_to_load = ["P_nt" + var.lstrip("P_nt")]
    elif var == 'cs':
        var_to_load = ["P", "rho"]
    elif var in list_vel:
        var_to_load = ["vel"]
    elif var in list_g:
        var_to_load = ["g"]
    elif var in list_mom:
        var_to_load = ["rho", "vel"]
    elif var == 'fHII':
        var_to_load = ["ion1", "ion3", "ion8"]
    elif var in list_x:
        var_to_load = [krome_ion[var.lstrip("x")][0]]
    elif var == 'N':
        var_to_load = ["rho"]
    elif var in list_N:
        var_to_load = ["rho", krome_ion[var.lstrip("N")][0]]
    elif var == "utot" or var=="photons":
        var_to_load = ["rad_density" + str(j) for j in range(1, NRAD+1)]
    elif var == "uion":
        var_to_load = ["rad_density" + str(j) for j in range(5, NRAD+1)]
    elif var in list_u:
        var_to_load = ["rad_density" + var.lstrip("u")]
    elif var in list_flux:
        var_to_load = ['rad_flux' + var[2:]]
    elif var in list_flux_r:
	var_to_load = ['rad_flux' + var[1]]
    elif var == "Z" or var == 'phi':
        var_to_load = [var]
    elif var == 'mu':
        var_to_load = ["rho"] + [krome_ion[el][0] for el in krome_ion.keys()]
    elif var == 'T':
        var_to_load = ["P", "rho"] + [krome_ion[el][0] for el in krome_ion.keys()]
    elif var == 'T_mu':
        var_to_load = ["P", "rho"]
    elif var == 'G0':
        var_to_load = ["rad_density3", "rad_density4"]
    elif var in list_M:
        var_to_load = ["rho", krome_ion[var.lstrip("M")][0]]
    elif var == 'vel_r':
        var_to_load = ['vel']
    elif var in ['G0_r', 'Nfuv_r']:
        var_to_load = ['rad_flux3', 'rad_flux4']
    elif var == 'Nion_r':
        var_to_load = ['rad_flux' + str(j) for j in range(5, NRAD+1)]
    elif var == 'lev':
        var_to_load = ['rho']
    elif var == 'Uion': 
        var_to_load  = ['rho', krome_ion['HII'][0], krome_ion['HI'][0], krome_ion['H2'][0]]
        var_to_load += ["rad_density" + str(j) for j in range(5, NRAD+1)]
    else:
        var_to_load = []

    return var_to_load

####### define functions

def function_fHII(cells):  # warning, KROME settings dependend
    return cells['ion3'] / (cells['ion1']+cells['ion3']+2*cells['ion8'])

def function_mu(dset):
    out = np.zeros_like(dset['rho'])
    for key in krome_ion.keys():
        out  +=  dset[krome_ion[key][0]] / krome_ion[key][1]
    out = 1. / out / krome_ion['HI'][1]
    return out

def function_T_over_mu(dset):
    return dset["P"]/dset["rho"]

def function_T(dset):
     return function_mu(dset) * function_T_over_mu(dset)

def function_uion(dset):  # warning, radiation bins dependent
    list_rad_ion  = ['rad_density' + str(j) for j in range(5,NRAD+1)]
    out           = np.zeros_like(dset['rad_density'+str(NRAD)])
    for campo in list_rad_ion:
        out += dset[campo]
    return out

def function_utot(dset):
    list_rad_tot  = ['rad_density'+str(j) for j in range(1,NRAD+1)]
    out           = np.zeros_like(dset['rad_density1'])
    for campo in list_rad_tot:
        out += dset[campo]
    return out

def function_Uion(dset): #ionization parameter
    uion = function_uion(dset)
    nH   = dset["rho"] * (dset[krome_ion['HI'][0]]+dset[krome_ion['HII'][0]]+dset[krome_ion['H2'][0]])
    return uion/nH

def function_G0(dset):
    dict_fuv = {}
    from astropy import units
    dict_fuv['rad_density3'] = 8.6* units.eV.to('erg')
    dict_fuv['rad_density4'] = 12.4*units.eV.to('erg')
    out = np.zeros_like(dset['rad_density3'])
    for rad_bin in dict_fuv.keys():
        out += dset[rad_bin] * dict_fuv[rad_bin]
    return out


####### OPERATOR - CELL

def operator(d, var, type_map, wg=None):

    assert wg  in list_var or wg==None

    if type_map != 'fft':
        error_string = "cannot make slices or ray-tracing with extensive variable"
        assert var not in list_extensive_var, error_string
        assert wg  not in list_extensive_var, error_string

    norm     = get_norm(var, d)
    unit     = get_unit(var)

    if isinstance(var, str):
        function = get_function(var)
    elif callable(var):
        function = var

    if wg is None:    # notice that wg is always None when type_map == 'slice'
        if var == 'lev':
            op = function
        else:
            op = ScalarOperator(lambda x: function(x) * norm)

    else:
        if isinstance(wg, str):
            wg_fun = get_function(wg)
        elif isinstance(wg, float) or isinstance(wg, int):
            wg_fun = lambda x: 1.
        elif callable(wg):
            wg_fun = wg

        op = FractionOperator(lambda x: function(x) * norm * wg_fun(x), wg_fun )

    return op, unit


def cell(d, cells, var, origin=None):

    assert var in list_var + ['vel_r', 'G0_r', 'M_r', 'Nfuv_r', 'Nion_r'] + list_flux_r

    if origin is None: origin = [0.5, 0.5, 0.5]

    if var in list_var:
        function = get_function(var)
        norm     = get_norm(var, d)
        unit     = get_unit(var)
        values   = function(cells) * norm

    else:

      if var == "vel_r":
          rr      = cells.points - origin
          r_unit  = rr / np.linalg.norm(rr, axis=1)[:, None]
          vel     = cells['vel']
          values  = np.einsum('ij,ij->i', vel, r_unit)  # elementwise dot product
          values *= get_norm('vel', d)
          unit    = get_unit('vel')

      elif var == 'G0_r':
          from astropy import units
          rr      = cells.points - origin
          r_unit  = rr / np.linalg.norm(rr, axis=1)[:, None]
          f3      = cells['rad_flux3']
          f4      = cells['rad_flux4']
          values  = np.einsum('ij,ij->i', f3, r_unit) * 8.6  * units.eV.to('erg')
          values += np.einsum('ij,ij->i', f4, r_unit) * 12.4 * units.eV.to('erg')
          values *= get_norm('G0', d)
          unit    = get_unit('G0')

      elif var == 'F3_r':
          from astropy import units
          rr      = cells.points - origin
          r_unit  = rr / np.linalg.norm(rr, axis=1)[:, None]
          flux    = cells['rad_flux' + var[1]]
          values  = np.einsum('ij,ij->i', flux, r_unit) * 8.6  * units.eV.to('erg')
          values *= get_norm('G0', d)
          unit    = get_unit('G0')

      elif var == 'F4_r':
          from astropy import units
          rr      = cells.points - origin
          r_unit  = rr / np.linalg.norm(rr, axis=1)[:, None]
          flux    = cells['rad_flux' + var[1]]
          values  = np.einsum('ij,ij->i', flux, r_unit) * 12.4 * units.eV.to('erg')
          values *= get_norm('G0', d)
          unit    = get_unit('G0')

      elif var == 'Nfuv_r':
          from astropy import units
          rr      = cells.points - origin
          r_unit  = rr / np.linalg.norm(rr, axis=1)[:, None]
          values = 0
          for i in [3, 4]:
              ff      = cells['rad_flux'+str(i)]
              values += np.einsum('ij,ij->i', ff, r_unit)
          values *= get_norm('u1', d)
          unit    = get_unit('u1')

      elif var == 'Nion_r':
          from astropy import units
          rr      = cells.points - origin
          r_unit  = rr / np.linalg.norm(rr, axis=1)[:, None]
          values = 0
          for i in range(5, 11):
              ff      = cells['rad_flux'+str(i)]
              values += np.einsum('ij,ij->i', ff, r_unit)
          values *= get_norm('u1', d)
          unit    = get_unit('u1')          

    return values, unit
