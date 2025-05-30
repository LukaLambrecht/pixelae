# imports

# external modules
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# local modules
thisdir = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)

import plotting.plottools as plottools


def plot_cluster_occupancy(hist, fig=None, ax=None,
                           xaxtitlesize=None, yaxtitlesize=None,
                           **kwargs):
    
    # make base plot
    fig, ax = plottools.plot_hist_2d(hist,
                # fixed arguments
                extent=None, aspect=None, origin='lower',
                # modifiable arguments
                **kwargs)
    
    # automatically determine x-axis labels from number of bins
    nxbins = hist.shape[1]
    axvlines = np.arange(8-0.5, nxbins-0.5, 8)
    xaxticks = np.arange(4-0.5, nxbins-0.5, 8)
    if nxbins==72:
        # BPix, raw
        xaxtitle = 'Module'
        xaxticklabels = np.arange(-4, 5)
    elif nxbins==64:
        # BPix, preprocessed
        xaxtitle = 'Module'
        xaxticklabels = np.concatenate((np.arange(-4, 0), np.arange(1, 5)))
    elif nxbins==56:
        # FPix, raw (to check number)
        xaxtitle = 'Disk'
        xaxticklabels = np.arange(-3, 4)
    elif nxbins==48:
        # FPix, preprocessed (to check number)
        xaxtitle = 'Disk'
        xaxticklabels = np.concatenate((np.arange(-3, 0), np.arange(1, 4)))
    else: raise Exception('Number of x-bins not recognized.')
        
    # x-axis layout
    ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    for x in axvlines: ax.axvline(x, color='black', linestyle='dotted')
    ax.set_xticks(ticks=xaxticks, labels=xaxticklabels)
        
    # automatically determine y-axis labels from number of bins
    nybins = hist.shape[0]
    axhlines = np.arange(2-0.5, nybins-0.5, 2)
    yaxticks = np.arange(1-0.5, nybins-0.5, 2)
    if nybins%4==2:
        # BPix, raw
        yaxtitle = 'Ladder'
        nladders = int((nybins-2)/4)
        yaxticklabels = np.arange(-nladders, nladders+1)
    elif nybins%4==0:
        # BPix, preprocessed
        yaxtitle = 'Ladder'
        nladders = int(nybins/4)
        yaxticklabels = np.concatenate((np.arange(-nladders, 0), np.arange(1, nladders+1)))
    # FPix: to do later
    else: raise Exception('Number of y-bins not recognized.')
        
    # y-axis layout
    ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    for y in axhlines: ax.axhline(y, color='black', linestyle='dotted')
    ax.set_yticks(ticks=yaxticks, labels=yaxticklabels)
        
    return fig, ax