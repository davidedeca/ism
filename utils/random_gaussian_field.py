import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.interpolate import interp1d

from constants import *

f_one = lambda x: 1.

def return_power_spectrum(power_index, kmin, kmax):
    def power_spectrum_function(k):
        k = np.array(k)
        ps = np.zeros_like(k)
        mask = np.logical_and((k > kmin), (k < kmax))
        ps[mask] = k[mask]**power_index
        return ps
    return power_spectrum_function


def get_k(input_array, box_dims):
    dim = len(input_array.shape)
    if dim == 1:
        x = np.arange(len(input_array))
        center = x.max()/2.
        kx = 2.*np.pi*(x-center)/box_dims[0]
        return [kx], kx
    elif dim == 2:
        x,y = np.indices(input_array.shape)
        center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        k = np.sqrt(kx**2 + ky**2)
        return [kx, ky], k
    elif dim == 3:
        x,y,z = np.indices(input_array.shape)
        center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2, \
                        (z.max()-z.min())/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        kz = 2.*np.pi * (z-center[2])/box_dims[2]
        k = np.sqrt(kx**2 + ky**2 + kz**2 )
        return [kx,ky,kz], k


def scalar_perturbs(level, boxlen, power_spectrum, scale_factor=f_one, random_seed=None):

    cells = 2**level
    Lcell = boxlen / cells
    dims = [cells, cells, cells]
    box_dims = [boxlen, boxlen, boxlen]
    random_seed = None


    if random_seed != None:
       np.random.seed(random_seed)

    map_ft_real = np.random.normal(loc=0., scale=1., size=dims)
    map_ft_imag = np.random.normal(loc=0., scale=1., size=dims)
    map_ft = map_ft_real + 1j*map_ft_imag
    k_comp, k_mod = get_k(map_ft_real, box_dims)

    del map_ft_real
    del map_ft_imag

    map_ft = map_ft * np.sqrt(power_spectrum(k_mod))

    map_ift = fftpack.ifftn(fftpack.fftshift(map_ft)).real

    map_ift *= scale_factor(map_ift)

    return map_ift


def isotropic_vector_perturbs(level, boxlen, power_spectrum, scale_factor=f_one, random_seed=None):
    velocity = scalar_perturbs(level, boxlen, power_spectrum, scale_factor, random_seed)
    theta = np.ones_like(velocity) * 2. * np.pi * np.random.uniform(size=velocity.shape)
    phi = np.ones_like(velocity) * np.arccos(1 - 2. * np.random.uniform(size=velocity.shape))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    comp_x = velocity * x
    comp_y = velocity * y
    comp_z = velocity * z
    return comp_x, comp_y, comp_z
