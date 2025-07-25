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


def plot_cluster_occupancy(hist, fig=None, ax=None, fast=False,
                           xaxtitlesize=None, yaxtitlesize=None,
                           **kwargs):
    
    # set suitable maximum excluding towers
    if 'caxrange' not in kwargs.keys():
        cmax = np.quantile(hist, 1-1/400)
        kwargs['caxrange'] = (1e-6, cmax)
    
    # make base plot
    fig, ax = plottools.plot_hist_2d(hist,
                # fixed arguments
                fig=fig, ax=ax, extent=None, aspect=None, origin='lower',
                # modifiable arguments
                **kwargs)
    
    # for faster plotting, don't bother with further aesthetics
    if fast: return fig, ax
    
    # automatically determine x-axis labels from number of bins
    nxbins = hist.shape[1]
    if nxbins==72:
        # barrel
        xaxtitle = r'Tower i$\phi$'
        xaxticks = np.arange(0, nxbins, 1)
        xaxticklabels = np.arange(1, nxbins+1, 1)
        axvlines = np.arange(3.5, nxbins-0.5, 4)
        
        # skip some ticks and labels for readability
        xaxticks = xaxticks[::4]
        xaxticklabels = xaxticklabels[::4]
        
    elif nxbins==20:
        # endcap
        xaxtitle = 'Tower ix'
        xaxticks = np.arange(0, nxbins, 1)
        xaxticklabels = np.arange(1, nxbins+1, 1)
        axvlines = None
        
        # skip some ticks and labels for readability
        xaxticks = xaxticks[::2]
        xaxticklabels = xaxticklabels[::2]
        
    else: raise Exception('Number of x-bins not recognized.')
        
    # x-axis layout
    ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if axvlines is not None:
        for x in axvlines: ax.axvline(x, color='black', linestyle='dotted')
    ax.set_xticks(ticks=xaxticks, labels=xaxticklabels)
        
    # automatically determine y-axis labels from number of bins
    nybins = hist.shape[0]
    if nxbins==72:
        # barrel
        yaxtitle = r'Tower i$\eta$'
        yaxticks = np.concatenate((np.arange(0, int(nybins/2), 4), np.arange(int(nybins/2), nybins, 4)))
        yaxticklabels = np.concatenate((np.arange(-int(nybins/2), 0, 4), np.arange(1, int(nybins/2)+1, 4)))
        axhlines = [nybins/2-0.5]
    elif nxbins==20:
        # endcap
        yaxtitle = 'Tower iy'
        yaxticks = np.arange(0, nybins, 1)
        yaxticklabels = np.arange(1, nybins+1, 1)
        axhlines = None
        
        # skip some ticks and labels for readability
        yaxticks = yaxticks[::2]
        yaxticklabels = yaxticklabels[::2]
        
    else: raise Exception('Number of y-bins not recognized.')
        
    # y-axis layout
    ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    if axhlines is not None:
        for y in axhlines: ax.axhline(y, color='black', linestyle='dotted')
    ax.set_yticks(ticks=yaxticks, labels=yaxticklabels)
        
    return fig, ax