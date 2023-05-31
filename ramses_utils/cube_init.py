import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.io import FortranFile
import os, shutil

default_varlist = ['d', 'vx', 'vy', 'vz', 'P']
default_varlist = default_varlist + ['x' + str(i) for i in range(1,10) ]

################ used only by interp method, it's a very inefficient alternative to vectorialization
def apply_f(f, x, y, z):
  res = np.zeros((len(x), len(y), len(z)))
  for i in range(len(x)):
     for j in range(len(y)):
        for k in range(len(z)):
           res[k][j][i] = f([x[i], y[i], z[i]])
  return res


class cube:

  def __init__(self, ncell, boxlen, var_list=default_varlist):
     print('... creating a '+str(ncell)+'^3 cube')
     self.ncell    = ncell
     self.boxlen   = boxlen
     x = np.linspace(0.5, ncell-0.5, ncell) / ncell * boxlen
     self.centers  = np.meshgrid(x, x, x)
     self.data     = np.array([ np.zeros((ncell,ncell,ncell)) for var in var_list ])
     self.var_list = var_list

  def get_var(self, var):
     assert var in self.var_list
     index = self.var_list.index(var)
     return self.data[index]

  def load_var(self, var, matrix):
     assert var in self.var_list
     assert np.array(matrix).shape == self.data[0].shapes
     self.data[self.var_list.index(var)] = np.copy(matrix)

  def add_region(self, var, shape, value, **params):
     # value can be a float, a matrix or a function
     # (param 'radial' specifies if it's a function of radius)
     assert shape in ['fill', 'sphere', 'square']
     assert var in self.var_list
     print('... adding '+var+' field to the cube')
     params.setdefault('radial', False)
     if shape == 'sphere':
        params.setdefault('center', [0.,0.,0.])
        params.setdefault('r', 0.)
     elif shape == 'square':
        params.setdefault('size', [0., 0., 0.])
     index = self.var_list.index(var)
     matrix = np.copy(self.data[index])
     if shape == 'fill':
        mask = np.full(matrix.shape, True)
     elif shape == 'sphere':
        c = np.array(params['center'])
        r = np.sqrt((self.centers[0]-c[0])**2 + (self.centers[1]-c[1])**2 + (self.centers[2]-c[2])**2)
        mask = r <= params['r']
     elif shape == 'square':
        c = np.array(params['center'])
        mask = np.logical_and(
               np.logical_and( abs(self.centers[0]-c[0]) <= params['size']/2.  ,
                               abs(self.centers[1]-c[1]) <= params['size']/2. ),
                               abs(self.centers[2]-c[2]) <= params['size']/2. )
     if callable(value):
        if params['radial'] is True:
           matrix[mask] = value(r[mask])
        else:
           matrix[mask] = value(self.centers[0][mask], self.centers[1][mask], self.centers[2][mask])
     else:
        if isinstance(value,float) or isinstance(value,int): value = np.array([value])
        if len(value.shape) == 1:
           matrix[mask] = value
        elif len(value.shape) == 3:
           assert value.shape == matrix.shape
           matrix[mask] = value[mask]
        else:
           pass
     self.data[self.var_list.index(var)] = np.copy(matrix)

  def show_slice(self, var, **params):
     assert var in self.var_list
     params.setdefault('dir', 'z')
     params.setdefault('log', False)
     params.setdefault('factor', 1.)
     assert params['dir'] in ['x', 'y', 'z']
     params.setdefault('depth', self.boxlen/2.)
     depth = int(params['depth'] / self.boxlen * len(self.data[0]))
     index = self.var_list.index(var)
     if   params['dir'] == 'x': slicemap = self.data[index][:,:,depth]
     elif params['dir'] == 'y': slicemap = self.data[index][:,depth,:]
     elif params['dir'] == 'z': slicemap = self.data[index][depth,:,:]
     if params['log']: slicemap = np.log10(slicemap)
     slicemap *= params['factor']
     im = plt.imshow(slicemap, origin='lower', cmap='inferno',
                     extent=[-self.boxlen/2., self.boxlen/2., -self.boxlen/2., self.boxlen/2.])
     plt.colorbar(im)
     plt.show()

  def save(self, directory, var_list=None):
      if var_list is None:
          var_list=self.var_list
      if os.path.exists(directory):
          shutil.rmtree(directory)
      os.makedirs(directory)
      for var in var_list:
          filename = directory + '/ic_' + var + '.dat'
          f = FortranFile(filename, mode='w')
          f.write_record(self.get_var(var), dtype='=f')
          f.closefile()



# ------------------------------
