import numpy as np
import scipy.integrate as int
import sys

from utils.constants import *

# solution of Lane-Emden equation
# BEsphereDB contains the data in 3 rows: csi, psi, m (reduced mass)
# dim = number of data
# max = maximum value of chi

dim = 1000000
max = 1000.


def f(Y, x):            # f = 0 is the Lane - Emden equation
    y, z = Y      # unpack current values of y
    derivs = [z, -2*z/x + np.exp(-y)]
    return derivs

x = np.linspace(1E-3, max, dim)     # x is \chi in the Lane-Emden
y0 = 1E-3
z0 = 0.
Y0 = [y0, z0]

sol = int.odeint(f, Y0, x)
y = sol[:, 0]                       # y is \psi in the Lane-Emden
z = sol[:, 1]

m = (4 * np.pi * np.exp(y))**(-0.5)  \
    * np.divide(np.gradient(y), np.gradient(x)) * np.square(x)

np.save("data/BEsphereDB", [x, y, m])
