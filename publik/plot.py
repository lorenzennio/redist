import numpy as np
import matplotlib
matplotlib.style.use('publik.style')
import matplotlib.pyplot as plt
import seaborn as sns
from publik import modifier

def dists(cmod, alt_pars, lims=None, plot_dists=True, plot_weights=False):
    
    if not lims:
        lims = [cmod.bins[0], cmod.bins[-1]]
    x = np.linspace(*lims, 100)
    
    null = modifier.bintegrate(cmod.null_dist, cmod.bins)
    alt = modifier.bintegrate(cmod.alt_dist, cmod.bins, tuple(alt_pars))
    
    fig, ax = plt.subplots()
    
    if plot_dists:
        ax.plot(x, cmod.null_dist(x), 'C1',label='null')
        ax.plot(x, cmod.alt_dist(x, *alt_pars), 'C2', label='alt')

        ax.stairs(null, cmod.bins[0],       color='C1', linewidth=1.5)
        ax.stairs(alt, cmod.bins[0],        color='C2', linewidth=1.5)
    
    if plot_weights:
        ax.plot(x, cmod.alt_dist(x, *alt_pars)/cmod.null_dist(x), 'C3', label='weights')
        ax.stairs(alt/null, cmod.bins[0],   color='C3', linewidth=1.5)
        ax.set_ylim(0, 1.5*max(alt/null))
    
    plt.legend()
    plt.show()
    print('weights : ', alt/null)
    
def map(cmod):
    fig, ax = plt.subplots()
    
    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(cmod.map, cmap=cmap, annot=True, annot_kws={"fontsize":7},
                square=True, linewidths=.5, ax=ax)

    ax.set_xlabel('Kinematic bins')
    ax.set_ylabel('Fitting bins')
    
    plt.tight_layout()

    fig.tight_layout()
    plt.show()