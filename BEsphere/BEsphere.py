import numpy as np
import matplotlib.pyplot as plt
import os

from utils.constants import *


print('... Loading Bonnor_Ebert profile database')


error_msg = 'create BEsphereDB.npy database first by running BEsphereDB.py'
error_msg_mag = 'create BEsphereDB_mag.npy database first by running BEsphereDB_mag.py'

path = os.path.abspath(__file__)
#dir_path = os.path.dirname(os.path.dirname(path))
#dir_path = os.path.join(dir_path, 'data')

dir_path = os.path.dirname(path)

path_db  = os.path.join(dir_path, 'BEsphereDB.npy')
assert(os.path.isdir(path_db)), error_msg
data = np.load(path_db)
data_x = data[0, :]
data_y = data[1, :]
data_m = data[2, :]
del(data)

#path_db  = os.path.join(dir_path, 'BEsphereDB_mag.npy')
#assert(os.path.isdir(path_db_mag)), error_msg_mag
#data_mag = np.load(database_mag)
#data_mag_x = data_mag[0, :]
#data_mag_y = data_mag[1, :]
#data_mag_m = data_mag[2, :]
#del(data_mag)

def BEmass(T, P0, particlemass):
    return 1.18 * (kB * T/ particlemass)**2 / np.sqrt(P0 * G**3)

def BEmass_magnetic(T, P0, B, R, particlemass):
    M_phi = 70 * (B / 10e-6) * (R / pc)**2 * Msun
    return BEmass(T, P0, particlemass) + M_phi

class BEsphere:

    def __init__(self, T, Pext, M, mu):
        if M > BEmass(T, Pext, mu * mp):
            raise ValueError("BE mass exceeded, collapse unavoidable")       
        self.T = T
        self.Pext = Pext
        self.M = M
        self.mu = mu
        self.csound = np.sqrt(kB * T / mu / mp)
        self.m = Pext ** 0.5 * G ** 1.5 * M / self.csound ** 4
        self.i_max = np.argmax(data_m)
        self.y0 = np.interp(self.m, data_m[0:self.i_max], data_y[0:self.i_max])
        self.x0 = np.interp(self.m, data_m[0:self.i_max], data_x[0:self.i_max])

    def radius(self):
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 * np.exp(self.y0)
        R = self.x0 / np.sqrt(4 * np.pi * G * rhoc) * self.csound
        return R

    def centraldensity(self):  # numberdensity
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 * np.exp(self.y0)
        return rhoc / self.mu / mp

    def density_at_radius(self, r):
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 * np.exp(self.y0)
        R = self.radius()
        x = r * np.sqrt(4 * np.pi * G * rhoc) / self.csound
        y = np.interp(x, data_x, data_y)
        n = rhoc * np.exp(-y) / self.mu / mp
        return n

    def densityprofile(self, N, log=0):     #numberdensity
        print('... Computing Bonnor-Ebert sphere')
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 * np.exp(self.y0)
        R = self.radius()
        if log:
            logr = np.linspace(np.log(R)-10., np.log(R), N)
            r = 10**logr
        else:
            r = np.linspace(0., R, N+1)
            r = r[::-1]
        x = r * np.sqrt(4 * np.pi * G * rhoc) / self.csound
        y = np.interp(x, data_x, data_y)
        n = rhoc * np.exp(-y) / self.mu / mp
        return r, n



class BEsphere_mag:

    def __init__(self, T, Pext, M, mu):
        if M > BEmass_magnetic(T, Pext, mu * mp):
            raise ValueError("BE mass exceeded, collapse unavoidable")       
        self.T = T
        self.Pext = Pext
        self.M = M
        self.mu = mu
        self.csound = np.sqrt(kB * T / mu / mp)
        self.m = Pext ** 0.5 * G ** 1.5 * M / self.csound ** 4
        self.i_max = np.argmax(data_mag_m)
        self.y0 = np.interp(self.m, data_mag_m[0:self.i_max], data_mag_y[0:self.i_max])
        self.x0 = np.interp(self.m, data_mag_m[0:self.i_max], data_mag_x[0:self.i_max])

    def radius(self):
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 / (1. - self.y0)**3
        R = self.x0 / np.sqrt(G) / rhoc**(1./3.)
        return R

    def centraldensity(self):  # numberdensity
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 / (1. - self.y0)**3
        return rhoc / self.mu / mp

    def density_at_radius(self, r):
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 / (1. - self.y0)**3
        R = self.radius()
        x = r * np.sqrt(4 * np.pi * G * rhoc) / self.csound
        y = np.interp(x, data_mag_x, data_mag_y)
        n = rhoc * (1. - y)**3 / self.mu / mp
        return n

    def densityprofile(self, N, log=0):     #numberdensity
        print('... Computing Bonnor-Ebert sphere')
        rho0 = self.Pext / self.csound ** 2
        rhoc = rho0 / (1. - self.y0)**3
        R = self.radius()
        if log:
            logr = np.linspace(np.log(R)-10., np.log(R), N)
            r = 10**logr
        else:
            r = np.linspace(0., R, N+1)
            r = r[::-1]
        x = r * np.sqrt(4 * np.pi * G * rhoc) / self.csound
        y = np.interp(x, data_mag_x, data_mag_y)
        n = rhoc * (1. - y)**3 / self.mu / mp
        return r, n




