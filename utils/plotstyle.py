import matplotlib as mpl
import numpy as np
from matplotlib import rc
from cycler import cycler

#mpl.rcParams["font.family"] = "sans-serif"
#mpl.rcParams["font.sans-serif"] = "Helvetica"
#mpl.rcParams["text.usetex"] = True

#mpl.rcParams["font.size"] = 12
#mpl.rcParams["xtick.labelsize"] = 12
#mpl.rcParams["ytick.labelsize"] = 12
#mpl.rcParams["legend.fontsize"] = 12
#mpl.rcParams["legend.frameon"] = False
#mpl.rcParams["lines.linewidth"] = 1.4

# STRINGS
cc   = r'$\ \mathrm{cm}^{-3}$'
Msun = r'$\ \mathrm{M}_\odot$'
pc   = r'$\ \mathrm{pc}$'
kpc  = r'$\ \mathrm{kpc}$'
Myr  = r'$\ \mathrm{Myr}$'
Lsun = r'$\ \mathrm{L}_\odot$'
mag  = r'$\ \mathrm{mag}$'
K    = r'$\ \mathrm{K}$'
km_s = r'$\ \mathrm{km}\,\mathrm{s}^{-1}$'
Kcc  = r'$\ \mathrm{K}\,\mathrm{cm}^{-3}$'
Msun_yr = r'$\ \mathrm{M}_\odot\,\mathrm{yr}^{-1}$'

def brack(x):
    if x[0:3] == '$\\ ' and x[-1]=='$':
        x = x[3:-1]
        x = '  [$' + x + '$]'
    elif x[0] == '$' and x[-1] == '$':
        x = x[1:-1]
        x = '  [$' + x + '$]'
    else:
        x = '  [' + x + ']'
    return x


def exp_notation(x, dec=1, force_exp=False, only_exp=False, math=False):
    if np.abs(x) < 1e-100: #i.e. = 0, not sure 1e-100 is a good number
        x_string = '0'
    else:    
        x_exp = int(np.log10(x))
        x_val = x / 10**x_exp
        if only_exp:
            x_string = '10^{' + str(x_exp)   + '}'
        elif force_exp:
            string = '{:' + str(dec+1) + '.' + str(dec) + 'f}'
            x_string = string.format(x_val) + '\\times 10^{' + str(x_exp)   + '}'

        else:
            if x_exp == 0:
                string = '{:' + str(dec+1) + '.' + str(dec) + 'f}'
                x_string = string.format(x_val)
            elif x_exp == 1:
                string = '{:' + str(dec+2) + '.' + str(dec) + 'f}'
                x_string = string.format(x_val*10)
            elif x_exp == -1:
                string = '{:' + str(dec+1) + '.' + str(dec) + 'f}'
                x_string = string.format(x_val*0.1)
            else:
                string = '{:' + str(dec+1) + '.' + str(dec) + 'f}'
                x_string = '\\times 10^{' + str(x_exp) + '}'
                if x_string != '1.0':
                    x_string = string.format(x_val) + x_string
    if math:
        x_string= '$' + x_string + '$'
    return x_string   


def math(string):
    return '$' + str(string) + '$'
    
    

# COLORS
crimson = '#DC143C'
limegreen = '#32CD32'
navy = '#000080'
royalblue = '#4169E1'
deeppink = '#FF1493'
gold = '#FFD700'
coral = '#FF7F50'
orangered = '#FF4500'
rebecca = '#663399'
salmon = '#FA8072'
mediumseagreen = '#3CB371'
saddlebrown = '#8B4513'

colors = {'blue' : royalblue, 'red' : '#ff4c4c'}

#mpl.rcParams["axes.prop_cycle"] = cycler('color', [salmon, mediumseagreen, saddlebrown, rebecca])
