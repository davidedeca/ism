import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from utils.constants import *
import wavesolver as ws

# R = gas on the right, L = gas on the left, PL>PR assumed in the case of resulting shock and rarefaction
# the positive sign of velocity is toward the right
# the function discontinuity_solver (stateR, stateL) must be invoked to solve the discontinuity.
# Which waves are going to form is printed, and the final equilibrium pressure across the separation interface is returned.


def discontinuity_chooser(stateR, stateL):      # decides the resulting waves and invokes the solver
    cR = stateR.csound()
    cL = stateL.csound()
    gammaR = stateR.gamma
    gammaL = stateL.gamma
    PR = stateR.P
    PL = stateL.P
    vR = stateR.v
    vL = stateL.v
    va = cR*np.sqrt(2/gammaR)*(PL/PR-1)/(np.sqrt((gammaR+1)*PL/PR+gammaR-1))
    vb = 2*cL*((PR/PL)**((gammaL-1)/2/gammaL)-1)/(gammaL-1)
    v = vL-vR
    if v > va:
        print "two SW"
        return 0
    elif v == va:
        print "one SW"
        return 1
    elif vb < v < va:
        print "RW + SW"
        return 2
    elif v == vb:
        print "one RW"
        return 3
    elif v < vb:
        print "two RW"
        return 4


def F(v, stateR, stateL):           # SL | SLL | SRR | SR
    SR = ws.State(stateR.rho, stateR.P, stateR.v, stateR.gamma, None, stateR.m)
    SL = ws.State(stateL.rho, stateL.P, -stateL.v, stateL.gamma, None, stateL.m)
    SRR = ws.shocksolver_Vgiven(SR, v)
    SLL = ws.shocksolver_Vgiven(SL, -v)
    return SRR.P - SLL.P


def G(v, stateR, stateL):           # SL | SLL | SRR | SR
    SR = ws.State(stateR.rho, stateR.P, stateR.v, stateR.gamma, None, stateR.m)
    SL = ws.State(stateL.rho, stateL.P, -stateL.v, stateL.gamma, None, stateR.m)
    SRR = ws.shocksolver_Vgiven(SR, v)
    SLL = ws.rarefactionsolver_Vgiven(SL, -v)
    return SRR.P-SLL.P


def H(v, stateR, stateL):           # SL | SLL | SRR | SR
    SR = ws.State(stateR.rho, stateR.P, stateR.v, stateR.gamma, None, stateR.m)
    SL = ws.State(stateL.rho, stateL.P, -stateL.v, stateL.gamma, None, stateL.m)
    SRR = ws.rarefactionsolver_Vgiven(SR, v)
    SLL = ws.rarefactionsolver_Vgiven(SL, -v)
    return SRR.P - SLL.P


def discontinuity_solver(stateR, stateL):
    x = discontinuity_chooser(stateR, stateL)
    if x == 0:
        v_eq = opt.brentq(F, stateR.v, stateL.v, args=(stateR, stateL))
        stateRR = ws.shocksolver_Vgiven(stateR, v_eq)
        return stateRR.P
    elif x == 1:
        return stateL.P
    elif x == 2:
        cL = stateL.csound()
        v_max = 2*cL/(stateL.gamma-1) - stateR.v
        v_eq = opt.brentq(G, stateR.v, v_max, args=(stateR, stateL))
        stateRR = ws.shocksolver_Vgiven(stateR, v_eq)
        return stateRR.P
    elif x == 3:
        return stateR.P
    elif x == 4:
        v_eq = opt.brentq(H, -1e6, 1e6, args=(stateR, stateL))
        stateRR = ws.rarefactionsolver_Vgiven(stateR, v_eq)
        return stateRR.P

for n in [1000, 1e4, 1e5, 1e6, 1e7]:
    print '*****', n
    sL = ws.State(n*mp, None, 0, T=1e5)
    sR = ws.State(1000*mp, None, -3e4, T=1e5)
    p = discontinuity_solver(sR, sL)
    sRsh = ws.shocksolver_Pgiven(sR, p)
    print sR.csound(), sRsh.csound(), sRsh.v, ws.shockspeed(sR, sRsh)
