import os
import numpy as np
import matplotlib.pyplot as plt
import pynbody
import pynbody.plot.sph as sph


available_global_props = ['nout', 't', 'Nstars', 'Mstars', 'MHII', 'MHI', 'MH2']
available_derive_props = ['SFR']

NBIG = 100000

class GMCsimulation:


    def __init__(self, path, datapath):

        self.path = path
        noutputs = [int(x[-5:]) for x in os.listdir(path) if 'output' in x]
        self.noutputs = np.sort(noutputs)
        self.datapath = datapath
        if not os.path.isdir(datapath):
            print('creating directory ' + datapath)
            os.mkdir(datapath)
        self.checkpoint = os.path.join(datapath, os.path.basename(path) + '_props.npz')
        print("GMC simulation object initialized")
        return 


    ####################

    def snap(self, nn):
        filename = os.path.join(self.path, 'output_'+str(nn).zfill(5))
        snap = pynbody.load(filename)
        return snap 


    ####################

    def save_props(self, props, nmax=NBIG):
        props = np.atleast_1d(props)
        global_props = []
        derive_props = []
        for prop in props:
            if prop in available_global_props:
                global_props.append(prop)
            elif prop in available_derive_props:
                derive_props.append(prop)
            else:
                raise ValueError("Property not known!")
        if len(global_props)>0:
            print("Starting to compute global cloud properties")
            self.__save_global_props(global_props, nmax)
        if len(derive_props)>0:
            print("Starting to save derived cloud properties")
            self.__save_derive_props(derive_props, nmax)
        return 


    ####################

    def __save_global_props(self, props, nmax=NBIG):

        props += ['nout', 't']

        nlast = dict()

        if os.path.isfile(self.checkpoint):
            data_old = np.load(self.checkpoint)
            for prop in props:
                if prop in data_old.files:
                    nlast[prop] = data_old['nout'][-1]
                else:
                    nlast[prop] = 1
        else:
            for prop in props:
                nlast[prop] = 1

        ncompleted = np.amin(list(nlast.values())) #min output for which all props are written        
        data_new = dict()

        for prop in props:
            data_new[prop] = np.zeros(len(self.noutputs))
        for prop in list(set(data_old.files) - set(props)):
            data_new[prop] = data_old[prop]
    
        nrange = self.noutputs[self.noutputs <= nmax]

        for i, nn in enumerate(nrange):

            if nn < ncompleted:

                for prop in props:
                    data_new[prop][i] = data_old[prop][i]

            else:

                filename = os.path.join(self.path, 'output_'+str(nn).zfill(5))
                snap = pynbody.load(filename)

                for prop in props:

                    if nn < nlast[prop]:
                        data_new[prop][i] = data_old[prop][i]

                    else:

                        if prop=='nout':
                            data_new['nout'][i] = nn

                        elif prop=='t':
                            data_new['t'][i] = snap.properties['time'].in_units('Myr')

                        elif prop=='Nstars':
                            x = len(snap.s)
                            data_new['Nstars'][i] = x

                        elif prop=='Mstars':
                            if not len(snap.s)==0:
                                x = np.sum(snap.s['mass'].in_units('Msol'))
                                data_new['Mstars'][i] = x
                            else:
                                data_new['Mstars'][i] = 0.

                        elif prop=='MH2':
                            x = np.sum(snap.g['xH2'] * snap.g['mass'].in_units('Msol'))
                            data_new['MH2'][i] = x

                        elif prop=='MHI':
                            x = np.sum(snap.g['xH'] * snap.g['mass'].in_units('Msol'))
                            data_new['MHI'][i] = x

                        elif prop=='MHII':
                            x = np.sum(snap.g['xHj'] * snap.g['mass'].in_units('Msol'))
                            data_new['MHII'][i] = x

            print(".. snapshot " + str(nn) + " done")

        np.savez(self.checkpoint, **data_new)

        return 


    ####################

    def __save_derive_props(self, props, nmax=NBIG):

        for prop in props:

            if prop=='SFR':
                self.__save_global_props(['t', 'Mstars'], nmax)
                data = dict(np.load(self.checkpoint))
                sfr = np.gradient(data['Mstars']) / np.gradient(data['t']) * 1e6
                data['sfr'] = sfr

        np.savez(self.checkpoint, **data)


    ####################

    def save_map(self, var, nout, tag=None, **kwargs):

        var  = np.atleast_1d(var)
        nout = np.atleast_1d(nout)

        kwargs['ret_im'] = True

        for nn in nout:

            filename = os.path.join(self.path, 'output_'+str(nn).zfill(5))
            snap = pynbody.load(filename)

            x = snap.g['x'].in_units('pc')
            size = abs(x.max() - x.min())
            snap['pos'] -= size/2.

            if 'width' not in kwargs.keys():
                kwargs['width'] = str(size) + ' pc'

            for vv in var:
                data = dict()
                im = sph.image(snap.g, vv, **kwargs)
                data['array']  = im.get_array().data
                data['extent'] = im.get_extent() 
                if 'units' in kwargs.keys():
                    data['var_units'] = kwargs['units']
                else:
                    str_units = str(snap.g[vv].units)
                    if str_units == 'NoUnit()':
                        data['var_units']  = ''
                    else:
                        data['var_units']  = str_units
                data['axis_units'] = kwargs['width'].split()[1]
                plt.close()
                filename = os.path.join(self.datapath, 'map_'+vv+'_'+str(nn).zfill(5))
                if tag is not None:
                    filename += tag
                np.savez(filename, **data)
                print('map saved for snapshot ' + str(nn) + ' and variable ' + vv)

        return 


    ####################

    def get_props(self, props):

        props = np.atleast_1d(props)

        if not os.path.isfile(self.checkpoint):
            raise ValueError("Save properties first!")
        else:
            data = np.load(self.checkpoint)

        for prop in props:
            if prop not in data.files:
                raise ValueError("You need to save first the property " + prop)

        if len(props) > 1:
            data_return = {prop : data[prop] for prop in props}
        else:
            data_return = data[props[0]]

        return data_return 


    ####################

    def get_all_props(self):
        return self.get_props(available_global_props+available_derive_props)


    ####################

    def get_map(self, var, nout, tag=None):       

        filename = os.path.join(self.datapath, 'map_'+var+'_'+str(nout).zfill(5))
        if tag is not None:
            filename += tag
        data = np.load(filename+'.npz')

        return data


    ####################

    def get_ax(self, ax, var, nout, tag=None, **kwargs):

        filename = os.path.join(self.datapath, 'map_'+var+'_'+str(nout).zfill(5))
        if tag is not None:
            filename += tag

        try:
            data = np.load(filename+'.npz')
        except:
            raise ValueError("Save map first with save_map")    

        if 'extent' not in kwargs.keys():
            kwargs['extent'] = data['extent']
        if 'aspect' not in kwargs.keys():
            extent = kwargs['extent']
            kwargs['aspect'] = (extent[1]-extent[0])/(extent[3]-extent[2])

        im = ax.imshow(data['array'], **kwargs)

        ax.set_xlabel(str(data['axis_units']))
        ax.set_ylabel(str(data['axis_units']))

        print("Image added to the ax")
        print("Returning also image, axis units and colormap units")

        return im, str(data['axis_units']), str(data['var_units'])


    ####################

    def plot_map(self, var, nout, tag=None, savefig=False, show=True, **kwargs):

        var  = np.atleast_1d(var)
        nout = np.atleast_1d(nout)

        for nn in nout:
            for vv in var:                
                
                fig, ax = plt.subplots()
                im, ax_units, var_units = self.get_ax(ax, vv, nn, tag, **kwargs)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel(vv+' ['+var_units+']')

                if savefig:
                    plt.savefig(savefig)

                if show:
                    plt.show()

        return 



















