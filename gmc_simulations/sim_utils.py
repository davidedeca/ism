import os
import numpy as np
import matplotlib.pyplot as plt
import pynbody


available_global_props = ['nout', 'tout', 'Nstars', 'Mstars', 'MHII', 'MHI', 'MH2']
available_derive_props = ['SFR']

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
        return 


    def save_props(self, props):
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
            self.__save_global_props(global_props)
        if len(derive_props)>0:
            self.__save_derive_props(derive_props)
        return 


    def __save_global_props(self, props):

        props += ['nout', 'tout']

        nlast = dict()
        if os.path.isfile(self.checkpoint):
            data_old = np.load(self.checkpoint)
            for prop in props:
                if prop in data_old.keys():
                    nlast[prop] = data['nout'][-1]
                else:
                    nlast[prop] = 1
        else:
            for prop in props:
                nlast[prop] = 1

        ncompleted = np.amin(nlast.values()) #min output for which all props are written 

        data_new = dict()

        for prop in props:
            data_new[prop] = np.zeros(len(self.noutputs))

        for i, nn in enumerate(self.noutputs):

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

                        elif prop=='tout':
                            data_new['tout'][i] = snap.properties['time'].in_units('Myr')

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

        np.savez(self.checkpoint, data_new)

        return 


    def __save_derive_props(self, props):

        for prop in props:

            if prop=='SFR':
                self.__save_global_props(['tout', 'Mstars'])
                data = dict(np.load(self.checkpoint))
                sfr = np.gradient(data['Mstars']) / np.gradient(data['tout']) * 1e6
                data['sfr'] = sfr

        np.save(self.checkpoint, data)


    def get_props_evol(self, props):

        props = np.atleast_1d(props)

        if not os.path.isfile(self.checkpoint):
            raise ValueError("Save properties first!")
        else:
            data = np.load(self.checkpoint)

        for prop in props:
            if prop not in data.keys():
                raise ValueError("You need to save first the property " + prop)

        data_return = {prop : data[prop] for prop in props}

        return data_return 





















