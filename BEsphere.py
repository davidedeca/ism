import numpy as np
import matplotlib.pyplot as plt
import os

from utils.constants import *

print '... Loading Bonnor_Ebert profile database'
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
database = dir_path + '/data/BEsphereDB.npy'

data = np.load(database)

data_x = data[0, :]
data_y = data[1, :]
data_m = data[2, :]

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
        print '... Computing Bonnor-Ebert sphere'
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


