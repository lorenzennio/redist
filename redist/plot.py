import numpy as np
import matplotlib
matplotlib.style.use('redist.style')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from redist import modifier

def dists(cmod, alt_pars=(), lims=None, labels = [], plot_dists=True, plot_weights=False):
    if len(cmod.bins) == 1:
        fig, ax = _dists1d(cmod, alt_pars, lims, labels, plot_dists, plot_weights)
        
    elif len(cmod.bins) == 2:
        fig, ax = _dists2d(cmod, alt_pars, lims, labels, plot_dists, plot_weights)
    return fig, ax

        
def _dists1d(cmod, alt_pars, lims, labels, plot_dists, plot_weights):
    if not lims:
        lims = [cmod.bins[0][0], cmod.bins[0][-1]]
    x = np.linspace(*lims, 100)
    
    null = modifier.bintegrate(cmod.null_dist, cmod.bins, cutoff=cmod.cutoff)
    alt = modifier.bintegrate(cmod.alt_dist, cmod.bins, tuple(alt_pars), cutoff=cmod.cutoff)
        
    if plot_dists and plot_weights:
        fig, ax = plt.subplots(1,2, figsize=(14,5))
        axdist, axw = ax
    else:   
        fig, ax = plt.subplots(figsize=(7,5))
        if plot_dists:
            axdist = ax
        elif plot_weights:
            axw = ax

    if plot_dists:
        # axdist.plot(x, cmod.null_dist(x), 'C1',label='null')
        # axdist.plot(x, cmod.alt_dist(x, *alt_pars), 'C2', label='alternative')

        axdist.stairs(null, cmod.bins[0], label="null", color='C0', linewidth=1.5)
        axdist.stairs(alt, cmod.bins[0],  label="alt.", color='C1', linewidth=1.5)
        axdist.legend()
        
        if labels:
            axdist.set_xlabel(labels[0])
            axdist.set_ylabel(labels[1])
        
    
    if plot_weights:
        axw.plot(x, np.divide(cmod.alt_dist(x, *alt_pars),cmod.null_dist(x)), 'C3', label='weights')
        axw.stairs(alt/null, cmod.bins[0],   color='C3', linewidth=1.5)
        axw.set_ylim(0, 1.5*max(alt/null))
        axw.legend()
        
        if labels:
            axw.set_xlabel(labels[0])
            axw.set_ylabel('Weights')
            

    return fig, ax
    
def _dists2d(cmod, alt_pars, lims, labels, plot_dists, plot_weights):
    if not lims:
        lims = []
        lims.append([cmod.bins[0][0], cmod.bins[0][-1]])
        lims.append([cmod.bins[1][0], cmod.bins[1][-1]])
        
    x = np.linspace(*lims[0], 100)
    y = np.linspace(*lims[1], 100)
    extent = [min(x),max(x),min(y),max(y)]

    X,Y = np.meshgrid(x, y) # grid of point
    
    Znull = cmod.null_dist(x, y)
    Znull_bin = modifier.bintegrate(cmod.null_dist, cmod.bins)
    
    Zalt = cmod.alt_dist(x, y, *tuple(alt_pars))
    Zalt_bin = modifier.bintegrate(cmod.alt_dist, cmod.bins, tuple(alt_pars))
    
    
    if not plot_weights:
        fig, ax = plt.subplots(1,2, figsize=(14,5))
        axnull, axalt = ax
    elif plot_dists and plot_weights:
        fig, ax = plt.subplots(1,3, figsize=(21,5))
        axnull, axalt, axw = ax
    else:   
        fig, ax = plt.subplots(figsize=(7,5))
        axw = ax
        
    if plot_dists:
        axnull.set_title('null distribution')
        im = axnull.imshow(Znull_bin, cmap='viridis', extent=extent, interpolation=None, origin='lower', aspect='auto') 
        cset = axnull.contour(X, Y, Znull, 10, linewidths=2, cmap='Oranges', extent=extent)
        # axnull.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        fig.colorbar(im)
        
        axalt.set_title('alternative distribution')
        im = axalt.imshow(Zalt_bin, cmap='viridis', extent=extent, interpolation=None, origin='lower', aspect='auto') 
        cset = axalt.contour(X, Y, Zalt, 10, linewidths=2, cmap='Oranges', extent=extent)
        # axalt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        fig.colorbar(im)

    if plot_weights:
        axw.set_title('weights')
        im = axw.imshow(Zalt_bin/Znull_bin, cmap='viridis', extent=extent, interpolation=None, origin='lower', aspect='auto') 
        fig.colorbar(im)
        
    if labels:
        for a in ax:
            a.set_xlabel(labels[0])
            a.set_ylabel(labels[1])
    
    return fig, ax

def map(cmod, **imshow_kwargs):
    fig, ax = plt.subplots()
    
    im = ax.imshow(cmod.map, **imshow_kwargs)
    
    # Calculate (height_of_image / width_of_image)
    im_ratio = cmod.map.shape[0]/cmod.map.shape[1]
    
    # Plot vertical colorbar
    fig.colorbar(im, fraction=0.047*im_ratio)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.minorticks_off()
    
    ax.set_xlabel('Kinematic bins')
    ax.set_ylabel('Reconstruction\nbins')
    
    fig.tight_layout()
    
    return fig, ax