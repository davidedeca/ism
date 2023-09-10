import numpy as np
import scipy.integrate as integrate
import sys, os

from utils.constants import *

_, dim, max_radius = sys.argv

dim        = int(dim)
max_radius = float(max_radius)


def save_BEtable(dim, max_radius):

    # BEsphereDB contains the data in 3 rows: csi, psi, m (reduced mass)
    # dim = number of data (typical: 1000000)
    # max = maximum value of chi (typical: 1000.)

    def f(Y, x):      # f = 0 is the Lane - Emden equation
        y, z = Y      # unpack current values of y
        derivs = [z, -2*z/x + np.exp(-y)]
        return derivs

    x = np.linspace(1E-3, max_radius, dim)     # x is \chi in the Lane-Emden
    y0 = 1E-3
    z0 = 0.
    Y0 = [y0, z0]

    sol = integrate.odeint(f, Y0, x)
    y = sol[:, 0]                       # y is \psi in the Lane-Emden
    z = sol[:, 1]

    m = (4 * np.pi * np.exp(y))**(-0.5)  \
        * np.divide(np.gradient(y), np.gradient(x)) * np.square(x)

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(os.path.dirname(path))
    dir_path = os.path.join(dir_path, 'data_untracked')
    if not(os.path.isdir(dir_path)):
        os.mkdir(dir_path)
    np.save(os.path.join(dir_path,"BEsphereDB"), [x, y, m])


save_BEtable(dim, max_radius)