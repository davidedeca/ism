import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pymses as pms
from pymses.analysis.visualization import Camera, SliceMap, ScalarOperator,raytracing
from pymses.analysis.visualization import fft_projection as fft_proj
from pymses.utils import constants as C
from pymses.filters import CellsToPoints
from pymsesplot_operators import get_var_to_load, operator, cell, NION
import pymsesplot_operators
reload(pymsesplot_operators)

sys.path.append('..')

ionization_fractions = ['ion'+str(i+1) for i in xrange(NION)]

def var_list(wfd=False, path='.', num=1):
    if wfd:
        variable_list = write_field_descr(path, num)
    else:
        variable_list = read_field_descr(path)
    print variable_list


def is_variable(var_name, namelist):
    is_var = 0
    i_var = namelist.find(var_name)
    if i_var > -1:
        while (namelist[i_var] is not '.'):
            i_var += 1
        if namelist[i_var+1] in ['f', 'F']: is_var = 0
        else: is_var = 1
    return is_var


def variable_detector(path='.', nout=1):
    nstr = str(nout).zfill(5)
    namelist = open(os.path.join(path, 'output_'+nstr, 'namelist.txt'), 'r').read()
    descriptor = open(os.path.join(path, 'output_'+nstr, 'hydro_file_descriptor.txt'), 'r').read()
    info = open(os.path.join(path, 'output_'+nstr, 'info_'+nstr+'.txt'), 'r').read()

    n_passive = descriptor.count('scalar_') + descriptor.count('krome_')       # number of passive scalars

    n_nener = descriptor.count('non_thermal_')

    n_dim = ''                                   # find the number of dimensions
    digits = [str(j) for j in range(9)]
    i_ndim = info.find('ndim')
    while (info[i_ndim] not in digits):
        i_ndim += 1
    while (info[i_ndim] in digits):
        n_dim += (info[i_ndim])
        i_ndim += 1
    n_dim = int(n_dim)

    metal = is_variable('metal', namelist)      # find out if there are metals

    grav = is_variable('grav', namelist)        # find out if there is gravity

    rt = is_variable('rt', namelist)            # find out if rt is on

    n_groups = ''
    if rt:                                          # count the number of RT photon groups
        info_rt = open(path + '/output_00001/info_rt_00001.txt', 'r').read()
        i_ngroups = info_rt.find('nGroups')
        while (info_rt[i_ngroups] not in digits):
            i_ngroups += 1
        while (info_rt[i_ngroups] in digits):
            n_groups += (info_rt[i_ngroups])
            i_ngroups += 1
        n_groups = int(n_groups)

    variable_list = ['rho', 'vel']
    for i in range(n_nener):
        if n_nener==1: variable_list.append('P_nt')
        else: variable_list.append('P_nt%d' % (i+1))
    variable_list.append('P')
    if (metal):
        variable_list.append('Z')

    for i in range(n_passive):
        variable_list.append(ionization_fractions[i])
    if grav:
        variable_list.append('phi')
        variable_list.append('g')
    if rt:
        for i in range(n_groups):
            variable_list.append('rad_density%d' % (i+1))
            variable_list.append('rad_flux%d'% (i+1))
    return variable_list, n_dim, n_nener, metal, n_passive, grav, rt, n_groups


def read_field_descr(path='.'):
    descr = open(os.path.join(path, 'pymses_field_descrs.py'), 'r').read()
    variable_list = []

    i1 = descr.find(':')

    i2 = 0

    if i1 == -1:
        return variable_list

    while i2 != -1:

        i1 = descr.find('[', i1)
        i2 = descr.find(':', i1)

        j1 = i1
        j2 = i1

        while j1 != -1:
            j1 = j2 + 1
            j1 = descr.find('"', j1, i2)
            j2 = descr.find('"', j1+1, i2)
            if j1 != -1:
                variable_list.append(descr[j1+1:j2])

        if i2 != -1:
            del variable_list[-1]

        i1 = i2

    return variable_list


def write_field_descr(path='.', nout=1):
    descr = open(os.path.join(path, 'pymses_field_descrs.py'), 'w+')

    variable_list, n_dim, n_nener, metal, n_passive, grav, rt, n_groups = variable_detector(path, nout)

    descr.write('from pymses.sources.ramses import output\n')
    descr.write('self.amr_field_descrs_by_file = \\\n')
    descr.write('    {"%dD": {\n' % n_dim)
    descr.write('        "hydro" : [ output.Scalar("rho", 0),\n')
    descr.write('                    output.Vector("vel", %s),\n' % str(range(1, n_dim + 1)))
    for i in range(n_nener):
        if n_nener==1: descr.write('                    output.Scalar("P_nt", %d),\n' % (n_dim + 1))
        else: descr.write('                    output.Scalar("%s", %d),\n' % ("P_nt"+str(i+1), n_dim + 1 + i))
    descr.write('                    output.Scalar("P", %d)' % (n_dim+n_nener+1))
    if metal:
        descr.write(',\n                    output.Scalar("Z", %d)' % (n_dim+n_nener+2))
    for i in range(n_passive):
        descr.write(',\n                    output.Scalar("%s", %d)' % (ionization_fractions[i], n_dim+n_nener+2+metal+i))
    descr.write(']')

    if grav:
        descr.write(',\n')
        descr.write('        "grav"  : [ output.Scalar("phi", 0),\n')
        descr.write('                    output.Vector("g", %s)]\n' % str(range(1, n_dim + 1)))

    if rt:
        descr.write(',\n')
        descr.write('        "rt"    : [ ')
        for i in range(n_groups):
            if i > 0:
                descr.write(',\n')
                descr.write('                    ')
            descr.write('output.Scalar("rad_density%d", %d), output.Vector("rad_flux%d", [%d, %d, %d])'
                        % (i+1, 4*i, i+1, 4*i+1, 4*i+2, 4*i+3))
        descr.write(']\n')

    descr.write('          }\n')
    descr.write('    }')
    descr.close()

    return variable_list


def slice(var, num, show=True, path='.', path_out='.', wfd=False, **params):
    map = make_maps(var=var,num=num , type_map='slice', show=show,
                    path=path, path_out = path_out, wfd = wfd, **params)
    return map


def fft_projection(var, num, show=True, path='.', path_out='.', wfd=False, **params):
    map = make_maps(var=var,num=num , type_map='fft', show=show,
                    path=path, path_out = path_out, wfd = wfd, **params)
    return map


def ray_projection(var, num, show=True, path='.', path_out='.', wfd=False, **params):
    map = make_maps(var=var,num=num , type_map='ray', show=show,
                    path=path, path_out = path_out, wfd = wfd, **params)
    return map


def make_maps(var, num, type_map='slice', show=True, path='.',path_out='.', wfd=False, **params):

    params.setdefault('centr'     , [0.5, 0.5, 0.5])
    params.setdefault('los'       , 'z')
    params.setdefault('z'         , 0.)
    params.setdefault('size'      , [1., 1.])
    params.setdefault('up'        , 'y')
    params.setdefault('d_cam'     , 0.5)
    params.setdefault('far_cut_depth', 0.5)
    params.setdefault('log'       , False)
    params.setdefault('vec'       , False)
    params.setdefault('part'      , False)
    params.setdefault('part_offset', 0.1) #10% of the box
    params.setdefault('part_size', 1.)
    params.setdefault('part_color', 'red')
    params.setdefault('part_edgecolor', params['part_color'])
    params.setdefault('cmap'      , None)
    params.setdefault('dpi'       , None)
    params.setdefault('format_out', 'png')
    params.setdefault('weight'    , None)

    params.setdefault('cam_file', None)
    if params['cam_file']:
        with open(params['cam_file'], 'rb') as f:
            data = pickle.load(f)

    if (isinstance(num, int)): num = [num]
    if len(num) > 6: num = num[:6]
    n = len(num)

    if wfd:
        variable_list = write_field_descr(path, num[0])
    else:
        variable_list = read_field_descr(path)

    var_to_load = list(set(get_var_to_load(var) + get_var_to_load(params['weight'])))
    assert set(var_to_load).issubset(variable_list)

    k = 0
    if n is 1: k = [1, 1, 7, 5]
    elif n is 2: k = [1, 2, 12, 5]
    elif n is 3: k = [1, 3, 16, 5]
    elif n in [4, 5, 6]: k = [2, 3, 14, 7]

    fig, axes = plt.subplots(nrows=k[0], ncols=k[1], figsize=(k[2], k[3]))
    map = [None]*n

    for i in range(n):
        if k[1] is 1: ax = axes
        else: ax = axes.flat[i]
        d = pms.RamsesOutput(path, num[i])
        params.setdefault('res', 2 ** d.info['levelmax'])
        if params['cam_file'] is None:
            cam = Camera(center=params['centr'], line_of_sight_axis=params['los'], region_size=params['size'],
                         up_vector=params['up'], map_max_size=params['res'], distance=params['d_cam'],
                         far_cut_depth=params['far_cut_depth'])
        else:
            cam = Camera(center=data[num[i]]['center_init '], line_of_sight_axis=data[num[i]]['los_vec'],
                         region_size=params['size'], up_vector=params['up'], map_max_size=data[num[i]]['mms'],
                         distance=params['d_cam'], far_cut_depth=params['far_cut_depth'])

        if(type_map =='slice'):
          op, unit = operator(d, var, type_map)
        elif(type_map == 'fft'):
          op, unit = operator(d,var,type_map,wg=params['weight'])
        elif(type_map == 'ray'):
          op, unit = operator(d,var,type_map,wg=params['weight'])
        amr      = d.amr_source(var_to_load)

        if(type_map =='fft'):
          mapp    = fft_proj.MapFFTProcessor(amr, d.info)
        elif(type_map =='ray'):
          mapp    = raytracing.RayTracer(d, var_to_load)

        if(type_map =='slice'):
          map[i] = SliceMap(amr, cam, op, z=params['z'])
        elif(type_map =='fft' or type_map =='ray'):
          map[i] = mapp.process(op, cam,surf_qty=False)

        print 'min', np.min(map[i]), 'max', np.max(map[i])
        print 'mean', np.mean(map[i]), 'stdiv', np.std(map[i])
        #map[i][map[i]<1.e-20]=1.e-20
        if params['log']: 
            if map[i].min() <= 0.:
                print 'WARNING: <=0 values are masked to make log'
                mask = (map[i] <= 0.)
                map[i][mask] = map[i][np.logical_not(mask)].min()   
            map[i] = np.log10(abs(map[i]))
        X = np.linspace(0., 1., params['res']) * d.info['unit_length'].express(C.pc)
        Y = np.linspace(0., 1., params['res']) * d.info['unit_length'].express(C.pc)

        if params['vec']:
            if params['vec'] == 'mom':
                opx, unitx = operator(d, 'momy', type_map)
                opy, unity = operator(d, 'momx', type_map)
            else:
                opx, unitx = operator(d, 'vely', type_map)
                opy, unity = operator(d, 'velx', type_map)
            if(type_map =='slice'):
              mapx = SliceMap(amr, cam, opx, z=params['z'])
              mapy = SliceMap(amr, cam, opy, z=params['z'])
            elif(type_map =='fft'):
              mp_x  = fft_projection.MapFFTProcessor(amr, d.info)
              mapx  = mp_x.process(opx, cam,surf_qty=False)
              mp_y  = fft_projection.MapFFTProcessor(amr, d.info)
              mapy  = mp_y.process(opy, cam,surf_qty=False)

            sk = int(2 ** d.info['levelmin'] / 12)
            Q = plt.quiver(X[::sk], Y[::sk], mapx[::sk, ::sk], mapy[::sk, ::sk], color='white')
            qk = plt.quiverkey(Q, 0.9, 0.9, 10, r'$10 \frac{km}{s}$', labelpos='E', coordinates='figure')

        if params['part']:# and type_map=='slice' :
            print '.. adding particles'
            assert params['los'] == 'z', 'particles available only for los = z'
            part = d.particle_source(["mass"]).flatten()
            los_vector = [0., 0., 1.]
            mask = abs(np.einsum('ij,j', part.points, los_vector) - (params['z'] + 0.5)) < params['part_offset']/2.
            part_pos = part.points[mask]
            part_x   = [ppos[0]*d.info['boxlen'] for ppos in part_pos]
            part_y   = [ppos[1]*d.info['boxlen'] for ppos in part_pos]
            part_mass = part['mass'][mask]
            ax.scatter(part_x, part_y, s=params['part_size']*100*part_mass, 
                       color=params['part_color'], edgecolor=params['part_edgecolor'])

        ax.set_title('$t=$' + '%.3f' %(d.info['time'] * d.info['unit_time'].express(C.Myr)) + ' Myr')
        ax.set_xlabel('$\mathrm{pc}$')
        ax.set_ylabel('$\mathrm{pc}$')

    params.setdefault('clevels', [np.min([np.min(map[i]) for i in range(n)]), np.max([np.max(map[i]) for i in range(n)])])

    vmin = np.min(params['clevels'])
    vmax = np.max(params['clevels'])

    for i in range(n):
        if k[1] is 1: ax = axes
        else: ax = axes.flat[i]
        size = params['size']
        extent = np.array([1.-size[0], 1.+size[0], 1.-size[1], 1.+size[1]]) * d.info['unit_length'].express(C.pc) / 2.
        im = ax.imshow(map[i].T, cmap=params['cmap'], vmin=vmin, vmax=vmax, extent=extent, origin='lower')
	#transposition to correct for coordinate flip

    fig.subplots_adjust(left=0.05, right=0.8, hspace=0.3)
    r_edge = 0.85
    if n is 1: r_edge = 0.77
    cbar_ax = fig.add_axes([r_edge, 0.15, 0.03, 0.7])     # left side, bottom side, width, height
    ticks = np.linspace(vmin, vmax, 10)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
    cb.set_clim(vmin=vmin, vmax=vmax)
    cbar_title = var + ' [' + unit + ']'
    if params['log']: cbar_title = '$log\ $' + cbar_title
    cb.set_label(cbar_title)

    tag_out = type_map

    if (n==1):
        filename = tag_out+"_"+var+"_"+str(num[0]).zfill(5)
    else:
        filename = tag_out+"_"+var+"_"+str(num[0]).zfill(5)+"-"+str(num[-1]).zfill(5)

    if(show):
        fig.show()

    out_f = path_out+ '/' + filename+'.'+params['format_out']
    print 'printing file to'
    print '  ',out_f
    fig.savefig(out_f, dpi=params['dpi'])
    return map

class cube:

    def __init__(self, num, path='.', wfd=False, **params):
        params.setdefault('origin'    , [0.5, 0.5, 0.5])
        params.setdefault('var_list'  , None)
        if wfd:
            variable_list = write_field_descr(path)
        else:
            variable_list = read_field_descr(path)        
        if params['var_list'] is not None:
            assert set(params['var_list']) <= set(variable_list), 'Invalid variable(s) in var_list'
            variable_list = params['var_list']
        cube.params = params
        cube.data = pms.RamsesOutput(path, num)
        source = cube.data.amr_source(variable_list)
        cell_source = CellsToPoints(source)
        cube.cells = cell_source.flatten()
        print '--- Computing sizes..'
        cube.sizes = cube.cells.get_sizes() * cube.data.info["unit_length"].express(C.pc)
        print '--- Computing coordinates..'
        cube.centers = cube.cells.points * cube.data.info["unit_length"].express(C.pc)

    def var(self, var):
        values, unit = cell(cube.data, cube.cells, var, cube.params['origin'])
        return values

    def x(self):
        return self.centers[:, 0]

    def y(self):
        return self.centers[:, 1]

    def z(self):
        return self.centers[:, 2]

    def r(self, center):
        xx = self.x() - center[0]
        yy = self.y() - center[1]
        zz = self.z() - center[2]
        r2 = xx**2 + yy**2 + zz**2
        return np.sqrt(r2)


class stars:

    def __init__(self, num, path='.', **params):
        d = pms.RamsesOutput(path, num)
        self.variable_list = ['vel', 'mass', 'id', 'level', 'epoch']
        part = d.particle_source(self.variable_list)
        part_all   = part.flatten()
        unit_l = d.info['unit_length'].express(C.cm)
        unit_t = d.info['unit_time'].express(C.s)
        unit_d = d.info['unit_density'].express(C.g_cc)
        unit_v = d.info['unit_velocity'].express(C.cm / C.s)
        unit_m = d.info['unit_mass'].express(C.g) 
        self.x     = part_all.points * unit_l / 3.085678e18 #pc
        self.vel   = part_all['vel'] * unit_v / 1.e5 #km_s
        self.mass  = part_all['mass'] * unit_m / 1.989e33 #Msun
        self.id    = part_all['id']
        self.lev   = part_all['level']
        self.time  = part_all['epoch'] * unit_t / 3.1536e13 #Myr
        #self.metal = part_all['metal']

    def sort(self, string):
        assert string in ['mass', 'id', 'time']
        if   string == 'mass': index = np.argsort(self.mass)
        elif string == 'id'  : index = np.argsort(self.id)
        elif string == 'time': index = np.argsort(self.time)
        self.x     = self.x[index]
        self.vel   = self.vel[index]
        self.mass  = self.mass[index]
        self.id    = self.id[index]
        self.lev   = self.lev[index]
        self.time  = self.time[index]
        return self       


def radial_profile(var, num, show=True, path='.', path_out='.', wfd=False, **params):

    params.setdefault('origin'     , [0.5, 0.5, 0.5])
    params.setdefault('log'        , False)
    params.setdefault('bins'       , 30)
    params.setdefault('clevels'    , None)
    params.setdefault('dpi'        , None)
    params.setdefault('format_out' , 'png')
    params.setdefault('plot_origin', 'corner')
    params.setdefault('cumulative' , False)
    params['origin'] = np.array(params['origin'])
    params.setdefault('rmax', np.min( np.concatenate((np.abs(params['origin']), np.abs(params['origin']-1.) )) ))
    params['rmax'] = np.min( np.concatenate(([params['rmax']], np.abs(params['origin']), np.abs(params['origin']-1.) )))

    plt.figure()

    if wfd:
        variable_list = write_field_descr(path)
    else:
        variable_list = read_field_descr(path)

    var_to_load = get_var_to_load(var)
    print var_to_load
    print variable_list
    assert set(var_to_load).issubset(variable_list)

    if (isinstance(num, int)): num = [num]
    n = len(num)

    for i in range(n):
        d = pms.RamsesOutput(path, num[i])
        source = d.amr_source(var_to_load)
        cell_source = CellsToPoints(source)
        cells       = cell_source.flatten()
        pos         = cells.points

        centro = params['origin']
        radii  = np.sqrt(np.sum((pos[:,:] - centro[np.newaxis,:])**2,axis=1))
        mask = radii <= params['rmax']
        radii = radii[mask] * d.info['unit_length'].express(C.pc)

        # define histogram
        values, unit = cell(d, cells, var, params['origin'])
        values = values[mask]
        count, Bins  = np.histogram(radii, params['bins'])
        X = np.array([(Bins[i]+Bins[i+1])/2. for i in range(Bins.size-1)])

        if params['cumulative'] is False:
            sum, Bins    = np.histogram(radii, params['bins'], weights=values)
            ave          = sum / count
        else:
            ave = np.array([np.sum(values[radii < XX]) for XX in X])

        if (params['plot_origin'] == 'center'):
            X = np.concatenate(( d.info['unit_length'].express(C.pc)/2. \
                                     - X[::-1], d.info['unit_length'].express(C.pc)/2. + X  ))
            ave = np.concatenate((ave[::-1], ave))

        if params['log']: ave = np.log10(ave)

        if (show): plt.plot(X, ave, label='$t=%.1g \,\mathrm{Myr}$' % d.info['time'])

    if(params['clevels']): plt.ylim(ymin=params['clevels'][0], ymax=params['clevels'][1])

    if (show):
      plt.ylabel(var + '(' + unit + ')')
      plt.xlabel('$\mathrm{pc}$')
      plt.legend(loc='upper right')
      plt.tight_layout()
      if (n==1):
          filename = "profile_"+var+"_"+str(num[0]).zfill(5)
      else:
          filename = "profile_"+var+"_"+str(num[0]).zfill(5)+"-"+str(num[-1]).zfill(5)
      out_f = path_out + '/' + filename+'.'+params['format_out']

      print 'printing file to'
      print '  ',out_f
      plt.savefig(out_f, dpi=params['dpi'])

      plt.show()

    plt.close()

    return X,ave


def simple_projection(var, num, show=True, path='.', wfd=False, **params):

    params.setdefault('log', False)
    params.setdefault('dir', 'y')
    params.setdefault('bins', 30)
    params.setdefault('clevels', None)

    if wfd:
        variable_list = write_field_descr(path)
    else:
        variable_list = read_field_descr(path)

    var_to_load = get_var_to_load(var)
    assert set(var_to_load).issubset(variable_list)

    if (isinstance(num, int)): num = [num]
    if len(num) > 6: num = num[:6]
    n = len(num)

    for i in range(n):
        d = pms.RamsesOutput(path, num[i])
        source = d.amr_source(var_to_load)
        cell_source = CellsToPoints(source)
        cells = cell_source.flatten()
        ccenters = cells.points
        dirr = 1
        if params['dir'] is 'x': dirr = 0
        if params['dir'] is 'z': dirr = 2
        cy = [C[dirr] for C in ccenters]
        values, unit = cell(d, cells, var)
        params.setdefault('ymin', 0.)
        params.setdefault('ymax', d.info['boxlen'])
        count, Bins = np.histogram(cy, params['bins'])
        sum, Bins = np.histogram(cy, params['bins'], weights=values)
        ave = sum / count
        X = [(Bins[i]+Bins[i+1])/2. * (params['ymax']-params['ymin']) for i in range(Bins.size-1)]
        if params['log']: ave = np.log10(ave)
        if show:
            plt.plot(X, ave, label='$t=%.1g \,\mathrm{Myr}$' % d.info['time'])

    if not show:
        print 'Not showing plot'
        return X, ave

    if params['clevels']: plt.ylim(ymin=params['clevels'][0], ymax=params['clevels'][1])
    plt.ylabel(var + '(' + unit + ')')
    plt.xlabel('$\mathrm{pc}$')
    plt.legend(loc='upper right')

    if (n==1):
        filename = "profile_"+var+"_"+str(num[0]).zfill(5)
    else:
        filename = "profile_"+var+"_"+str(num[0]).zfill(5)+"-"+str(num[-1]).zfill(5)

    plt.savefig(path + '/' + filename, dpi=100)
    plt.show(block=False)
    return X, ave


def profiley(var, num, log=False, show=True, path='.', path_out='.', wfd=False, **params):

    params.setdefault('size', 1.)
    params.setdefault('clevels', None)

    if wfd:
        variable_list = write_field_descr(path, num[0])
    else:
        variable_list = read_field_descr(path)

    var_to_load = get_var_to_load(var)
    assert set(var_to_load).issubset(variable_list)

    X_out=list()
    Y_out=list()

    if (isinstance(num, int)): num = [num]
    n = len(num)

    plt.figure()

    for i in range(n):
        d = pms.RamsesOutput(path, num[i])
        op, unit = operator(d, var, type_map='slice')
        cam = Camera(line_of_sight_axis='z', up_vector='y')
        amr = d.amr_source(var_to_load)
        map = SliceMap(amr, cam, op, z=0.)
        print int(np.sqrt(map.size)) / 2
        extent = np.array([1. - params['size'], 1. + params['size']]) / 2.
        Y = map[int(np.sqrt(map.size))/2]
        Y = Y[ int(extent[0]*Y.size) : int(extent[1]*Y.size) ]
        if log: Y = np.log10(Y)
        extent = extent * d.info['unit_length'].express(C.pc)
        X = np.linspace(extent[0], extent[1], Y.size)
        # plot smoothing
        X_new = []
        Y_new = []
        ii= 0
        jj = 1
        while (ii < X.size):
           while (ii+jj < Y.size and Y[ii + jj] == Y[ii]):
              jj += 1
           X_new.append(np.average(X[ii:ii+jj+1]))
           Y_new.append(Y[ii])
           ii += jj+1
           jj = 1

        plt.plot(X_new, Y_new, label='$t=%.3f \,\mathrm{Myr}$' %d.info['time'])

        X_out.append(X_new)
        Y_out.append(Y_new)

    if params['clevels']: plt.ylim(ymin=params['clevels'][0], ymax=params['clevels'][1])
    plt.xlabel('$\mathrm{pc}$')
    plt.ylabel(var + ' (' + unit + ')')
    plt.legend(loc='best')
    plt.tight_layout()
    if (n==1):
        filename = "profiley_"+var+"_"+str(num[0]).zfill(5)
    else:
        filename = "profiley_"+var+"_"+str(num[0]).zfill(5)+"-"+str(num[-1]).zfill(5)
    out_f = path_out + '/' + filename+'.png'
    plt.savefig(out_f)
    if (show):
        plt.show()

    if(len(X_out) == 1):
      X_out,Y_out =X_out[0],Y_out[0]

    return X_out,Y_out

