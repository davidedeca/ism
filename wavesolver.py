import numpy as np

from utils.constants import *

# The following convention is adopted: the flow velocity is positive if it is the same direction as the shock front

class State:    # to define a fluid state, set only two among rho, P, T..
                # the remaining one is computed by the constructor
                # if 'state' is given to the constructor, it works as a copy constructor

    def __init__(self, rho=None, P=None, v=0, gamma=5./3, T=None, m=mp, state=None):
        if state is None:
            if rho is None:
                if T == 0: rho = 0
                else: rho = m * P / kB / T
            elif P is None:
                P = rho * kB * T / m
            elif T is None:
                if rho == 0: T = 0
                else: T = m * P / kB / rho
            else:
                print "please, assign just two gas variables!"
                return
            self.rho = rho
            self.P = P
            self.v = v
            self.gamma = gamma
            self.T = T
            self.m = m
        else:
            self.rho = state.rho
            self.P = state.P
            self.v = state.v
            self.gamma = state.gamma
            self.T = state.T
            self.m = state.m

    def csound(self):
        return np.sqrt(self.gamma*self.P/self.rho)

    def isothermalcsound(self):
        return np.sqrt(self.P/self.rho)

    def show(self):
        print "n = ", self.rho/mp, "\tP = ", self.P, "\tT = ", self.T, "\tv = ", self.v, "\tgamma = ", self.gamma, "\tm = ", self.m


def shocksolver_Pgiven(state0, P1):     # returns post-shock state, if post-shock pressure is given
    gamma = state0.gamma
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    M0 = np.sqrt(((gamma+1.)*P1/P0+gamma-1.)/2./gamma)
    c0 = state0.csound()
    D = v0+c0*M0
    M1 = np.sqrt(((M0**2.)+2./(gamma-1.)) / ((2.*gamma*(M0**2.))/(gamma-1.)-1.))
    rho1 = rho0 * ((gamma-1)*P0+(gamma+1)*P1) / ((gamma-1)*P1+(gamma+1)*P0)
    c1 = np.sqrt(gamma*P1/rho1)
    v1 = D-c1*M1
    state1 = State(rho1, P1, v1, gamma, None, state0.m)
    return state1


def shocksolver_Vgiven(state0, v1):     # returns post-shock state, if post-shock velocity is given
    gamma = state0.gamma
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.csound()
    a = 2
    b = 2*(gamma-1)*v0-(gamma+1)*(v0+v1)
    c = (gamma+1)*v0*v1-(gamma-1)*v0**2-2*c0**2
    D = (-b + np.sqrt(b**2-4*a*c))/(2.*a)
    M0 = (D-v0)/c0
    M1 = np.sqrt(((M0**2.)+2./(gamma-1.)) / ((2.*gamma*(M0**2.))/(gamma-1.)-1.))
    P1 = P0 * (1+gamma*M0**2)/(1+gamma*M1**2)
    rho1 = rho0 * ((gamma - 1) * P0 + (gamma + 1) * P1) / ((gamma - 1) * P1 + (gamma + 1) * P0)
    state1 = State(rho1, P1, v1, gamma, None, state0.m)
    return state1


def shocksolver_Dgiven(state0, D):      # returns post-shock state, if shock velocity is given
    gamma = state0.gamma
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.csound()
    M0 = (D - v0) / c0
    M1 = np.sqrt(((M0 ** 2.) + 2. / (gamma - 1.)) / ((2. * gamma * (M0 ** 2.)) / (gamma - 1.) - 1.))
    P1 = P0 * (1 + gamma * M0 ** 2) / (1 + gamma * M1 ** 2)
    rho1 = rho0 * ((gamma - 1) * P0 + (gamma + 1) * P1) / ((gamma - 1) * P1 + (gamma + 1) * P0)
    c1 = np.sqrt(gamma * P1 / rho1)
    v1 = D - c1 * M1
    state1 = State(rho1, P1, v1, gamma, None, state0.m)
    return state1


def shockspeed(state0, state1):         # returns shock velocity, if pre-shock and post-shock states are given
    gamma = state0.gamma
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    P1 = state1.P
    M0 = np.sqrt(((gamma + 1.) * P1 / P0 + gamma - 1.) / 2. / gamma)
    c0 = state0.csound()
    D = v0 + c0 * M0
    return D


def isothermalshockspeed(state0, state1):         # returns shock velocity, if pre-shock and post-shock states are given
    return (state1.rho * state1.v - state0.rho * state0.v) / (state1.rho - state0.rho)


def isothermalmagneticshockspeed(state0, state1):
    return state0.v + np.sqrt((state0.P - state1.P)/(state0.rho - state1.rho) * state1.rho / state0.rho)


def rarefactionsolver_Pgiven(state0, P1):   # returns post-rarefaction state,
    gamma = state0.gamma                    # if post-rarefaction pressure is given
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.csound()
    v1 = v0 - c0*2/(gamma-1)*(1-(P1/P0)**((gamma-1)/2/gamma))
    rho1 = rho0 * (P1/P0)**(1/gamma)
    state1 = State(rho1, P1, v1, gamma, None, state0.m)
    return state1


def rarefactionsolver_Vgiven(state0, v1):   # returns post-rarefaction state,
    gamma = state0.gamma                    # if post-rarefaction velocity is given
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.csound()
    P1 = P0*(1+(gamma-1)*(v1-v0)/2/c0)**(2*gamma/(gamma-1))
    rho1 = rho0 * (P1/P0)**(1/gamma)
    state1 = State(rho1, P1, v1, gamma, None, state0.m)
    return state1


def isothermalshocksolver_Dgiven(state0, D):
    state1 = shocksolver_Dgiven(state0, D)
    gamma = state0.gamma
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.isothermalcsound()
    u0 = D - v0
    a = 1
    b = -(P0 + rho0 * u0**2)
    c = P0 * rho0 * u0**2
    P11 = (-b + np.sqrt(b**2-4*a*c))/(2.*a)
    T11 = state0.T
    state11 = State(None, P11, 0, gamma, T11, state0.m)
    state11.v = D - rho0 * u0 / state11.rho
    return state11


def isothermalshocksolver_Pgiven(state0, P1):
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    rho1 = rho0 * P1 / P0
    u0 = np.sqrt(rho1 / rho0 * (P1 - P0) / (rho1 - rho0))
    u1 = u0 * rho0 / rho1
    D = u0 + v0
    v1 = D - u1
    return State(rho1, P1, v1, state0.gamma, None, state0.m)


def isothermalshocksolver_Vgiven(state0, v1):
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    a = 1
    b = -(2 * P0 + (v1 - v0)**2 * rho0)
    c = P0**2
    P1 = (-b + np.sqrt(b**2-4*a*c))/(2.*a)
    rho1 = rho0 / P0 * P1
    return State(rho1, P1, v1, state0.gamma, None, state0.m)


def isothermalshocksolver_Dgiven(state0, D):
    gamma = state0.gamma
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.isothermalcsound()
    u0 = D - v0
    a = 1
    b = -(P0 + rho0 * u0**2)
    c = P0 * rho0 * u0**2
    P11 = (-b + np.sqrt(b**2-4*a*c))/(2.*a)
    T11 = state0.T
    state11 = State(None, P11, 0, gamma, T11, state0.m)
    state11.v = D - rho0 * u0 / state11.rho
    return state11


def isothermalrarefactionsolver_Pgiven(state0, P1):
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.isothermalcsound()
    v1 = v0 + c0 * np.log(P1 / P0)
    return State(None, P1, v1, state0.gamma, state0.T, state0.m)


def isothermalrarefactionsolver_Vgiven(state0, v1):
    rho0 = state0.rho
    P0 = state0.P
    v0 = state0.v
    c0 = state0.isothermalcsound()
    P1 = P0 * np.exp((v1 - v0) / c0)
    # rho1 = rho0 * np.exp((v1 - v0) / c0)
    return State(None, P1, v1, state0.gamma, state0.T, state0.m)


