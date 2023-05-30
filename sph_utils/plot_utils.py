import pynbody
import numpy as np
import matplotlib.pyplot as plt
import pynbody.plot.sph as sph
import sph_utils.pynbody_operators

def load_centered(filename, *args, **kwargs):
    snap = pynbody.load(filename, *args, **kwargs)
    x = snap['x']
    dx = 0.5 * (x.max() + x.min())
    snap['pos'] -= dx
    return snap

def set_axis(ax, **params):

    params.setdefault('xlim', [0, 1])
    params.setdefault('ylim', [0, 1])
    params.setdefault('xlabel', '')
    params.setdefault('ylabel', '')
    params.setdefault('xlogscale', False)
    params.setdefault('ylogscale', False)
    params.setdefault('invert_xaxis', False)
    params.setdefault('invert_yaxis', False)
    params.setdefault('title', False)

    ax.set_xlim(params['xlim'])
    ax.set_ylim(params['ylim'])
    #ax.set_aspect(np.float(params['xlim'][1] - params['xlim'][0]) / \
    #              np.float(params['ylim'][1] - params['ylim'][0]))
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    if params['xlogscale']:
        ax.set_xscale('log')
    if params['ylogscale']:
        ax.set_yscale('log')

    if params['invert_xaxis']:
        ax.invert_xaxis()
    if params['invert_yaxis']:
        ax.invert_yaxis()
    
    if params['title']:
        ax.set_title(params['title'])

    return


def histogram2d(xdata, ydata, weight=None, **params):
                #show=True, bins=300, cmap='viridis', ret_data=False, savedir=None,
                #xlim=None, ylim=None, xlabel=None, ylabel=None, cbar_label=None,
                #title=None, **params):

    params.setdefault('logx', True)
    params.setdefault('logy', True)
    params.setdefault('show', True)
    params.setdefault('bins', 300)
    params.setdefault('cmap', 'viridis')
    params.setdefault('savedata', None)
    params.setdefault('saveplot', None)
    params.setdefault('xlabel', None)
    params.setdefault('ylabel', None)
    params.setdefault('cbar_label', 'log count')
    params.setdefault('title', None)
    params.setdefault('invert_xaxis', False)
    params.setdefault('cmap', 'viridis')
    params.setdefault('vmin', -5)
    params.setdefault('vmax', 0)
    params.setdefault('return_type', 'show') #show, data or ax
    params.setdefault('ax', None)
    params.setdefault('fig', None)

    if params['logx']:
        xdata = np.log10(xdata)
    if params['logy']:
        ydata = np.log10(ydata)

    hist, binsx, binsy = np.histogram2d(xdata, ydata, params['bins'], weights=weight)

    extent = [binsx[0], binsx[-1], binsy[0], binsy[-1]]
    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])

    if params['return_type'] == 'ax':
        #fig, ax = params['fig'], params['ax']
        ax = params['ax']
    else:
        fig, ax = plt.subplots()

    im = ax.imshow(np.log10(hist.T/hist.max()), origin='lower',
                            extent=extent, aspect=aspect,
                            vmin=params['vmin'], vmax=params['vmax'], cmap=params['cmap'])

    params.setdefault('xlim', [binsx[0], binsx[-1]])
    params.setdefault('ylim', [binsy[0], binsy[-1]])

    set_axis(ax, xlim=params['xlim'], ylim=params['ylim'],
             invert_xaxis=params['invert_xaxis'], xlabel=params['xlabel'], ylabel=params['ylabel'],
             title=params['title']
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(params['cbar_label'])

    if params['saveplot']:
        if params['saveplot'].split('.')[-1] not in ['png', 'jpg']:
            params['saveplot'] += '.png'
        plt.savefig(params['saveplot'], bbox_inches='tight', pad_inches=0.)

    if params['savedata']:
        np.save(params['savedata'], [hist, binsx, binsy])

    if params['return_type'] == 'show':
        plt.show()
    elif params['return_type'] == 'data':
        plt.close(fig)
        return hist, binsx, binsy
    elif params['return_type'] == 'ax':
        return ax

    return hist, binsx, binsy


def histogram1d(xdata, weight=None, **params):

    params.setdefault('logx', True)
    params.setdefault('logy', True)
    params.setdefault('bins', 300)
    params.setdefault('figsize', None)
    params.setdefault('save', None)
    params.setdefault('xlabel', None)
    params.setdefault('ylabel', 'log count')
    params.setdefault('color', None)
    params.setdefault('alpha', 1.)
    params.setdefault('ls', None)
    params.setdefault('label', None)
    params.setdefault('title', None)
    params.setdefault('invert_xaxis', False)
    params.setdefault('bar', False)
    params.setdefault('xlim', None)
    params.setdefault('ylim', None) 
    params.setdefault('norm', None)    #can be 'max' or 'density'
    params.setdefault('ax', None)
    params.setdefault('show', True)

    if weight is None:
        weight = np.ones_like(xdata) 

    if params['logx']:
        weight = weight[xdata>0.]
        xdata = xdata[xdata>0.]
        xdata = np.log10(xdata)

    assert len(xdata)>0, 'the data array is empty or all zero'

    hist, bins = np.histogram(xdata, bins=params['bins'], weights=weight)

    if params['norm'] == 'max':
        hist = hist.astype('float')/hist.max()
    elif params['norm'] == 'density':
        hist = hist.astype('float')/np.sum(hist)
    else:
        hist = hist.astype('float')/params['norm']

    bins = 0.5 * (bins[:-1] + bins[1:])

    if params['logy'] and not params['bar']:
        bins = bins[hist>0]
        hist = hist[hist>0]
        hist = np.log10(hist)

    if params['ax'] is not None:
        ax = params['ax']
    else:
        fig, ax = plt.subplots(figsize=params['figsize'])

        if params['xlim'] is None:
            params['xlim'] = [bins[0]-0.05*bins[-1], bins[-1]*1.05]

        if params['ylim'] is None:
            ymin = hist.min()
            ymax = hist.max() + np.abs(hist.max() - hist.min())*0.02
            params['ylim'] = [ymin, ymax] 

        set_axis(ax, xlim=params['xlim'], ylim=params['ylim'],
             invert_xaxis=params['invert_xaxis'], xlabel=params['xlabel'], 
             ylabel=params['ylabel'], title=params['title']
            )

    if params['bar']:
        bar_bottom = params['ylim'][0]
        bar_height = np.abs(hist - bar_bottom) 
        params.setdefault('bar_width', np.abs(bins[1]-bins[0]))
        ax.bar(bins, bar_height, bottom=bar_bottom, color=params['color'], 
               width=params['bar_width'], alpha=params['alpha'], label=params['label'], log=params['logy'])
    else:
        ax.plot(bins, hist, color=params['color'], ls=params['ls'], 
                label=params['label'], alpha=params['alpha'])

    if params['save']:
        filename = params['save']
        ext = filename.split('.')[-1] 
        if ext in ['png', 'jpg']:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.)
        elif ext in ['txt']:
            np.savetxt(filename, [hist, bins])
        elif ext in ['npy']:
            np.save(filename, [hist, bins])
        else: 
            assert False, 'Format of file to save not recognized'

    if params['show'] and params['ax'] is None:
        plt.show()
    elif params['ax'] is None:
        plt.close()
    
    return hist, bins 

def map(sim, quantity, width, integrated=False,**params): 

    #wrapper for pynbody.sph.image

    params.setdefault('res', 500)
    params.setdefault('log', True)
    params.setdefault('savedata', None)
    params.setdefault('saveplot', None)
    params.setdefault('xlabel', None)
    params.setdefault('ylabel', None)
    params.setdefault('cbar_label', None)
    params.setdefault('title', None)
    params.setdefault('cmap', 'viridis')
    params.setdefault('vmin', None)
    params.setdefault('vmax', None)
    params.setdefault('return_type', 'show') #show, data or ax
    params.setdefault('ax', None)
    params.setdefault('fig', None)

    if np.log2(params['res']).is_integer():
        raise Warning("There could be problems by setting the resolution to a power of 2")

    im = sph.image(sim, quantity, width=width, resolution=params['res'], 
                   av_z=integrated, log=params['log'], ret_im=True)

    plt.close()

    if params['return_type'] == 'ax':
        ax = params['ax']
    else:
        fig, ax = plt.subplots()

    Z = im.get_array()
    if params['log']:
        Z = np.log10(Z)

    params.setdefault('vmin', Z.min())
    params.setdefault('vmax', Z.max())

    X = [- float(width.split()[0])/2, float(width.split()[0])/2]
    Y = [- float(width.split()[0])/2, float(width.split()[0])/2]
    extent = [X[0], X[-1], Y[0], Y[-1]]
    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
    im = ax.imshow(Z, extent=extent, aspect=aspect,
               cmap=params['cmap'], vmin=params['vmin'], vmax=params['vmax'])    

    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cbar.ax.set_ylabel(params['cbar_label'])

    if params['saveplot']:
        if params['saveplot'].split('.')[-1] not in ['png', 'jpg']:
            params['saveplot'] += '.png'
        plt.savefig(params['saveplot'], bbox_inches='tight', pad_inches=0.)

    if params['savedata']:
        np.save(params['savedata'], Z)

    if params['return_type'] == 'show':
        plt.show()
    elif params['return_type'] == 'data':
        plt.close(fig)
        return hist, binsx, binsy
    elif params['return_type'] == 'ax':
        return ax

    return
