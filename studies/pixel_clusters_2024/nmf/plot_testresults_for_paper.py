#!/usr/bin/env python3
"""
Make the pixel cluster occupancy example figures for the paper
(input, loss and cleaned loss).

Usage:
    python3 plot_testresults_for_paper.py
    python3 plot_testresults_for_paper.py --input <file.npz> --outputname <name> --year 2025 --layer BPix1

The input .npz file is expected to contain the arrays 'orig', 'loss' and 'binary'.
The figure is written to <outputname>.png and <outputname>.pdf.
"""

import argparse
from copy import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Plotting functions
# (copied and modified from elsewhere in the framework)

def add_colorbar(fig, ax, cobject, width=0.05, pad=0.03, **kwargs):
    # add a colorbar spanning the full height of ax;
    # width and pad are expressed as fractions of the axis width
    cax = ax.inset_axes([1 + pad, 0, width, 1])
    return fig.colorbar(cobject, cax=cax, **kwargs)


def plot_hist_2d(hist, fig=None, ax=None, figsize=None, title=None, titlesize=None,
                 xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
                 ticklabelsize=None, colorticklabelsize=None, extent=None, aspect=None, box_aspect=None,
                 caxrange=None, caxsoftmax=False,
                 docolorbar=True, caxtitle=None, caxtitlesize=None, caxtitleoffset=None,
                 origin='lower'):

    if fig is None or ax is None: fig, ax = plt.subplots(figsize=figsize)

    # settings
    histmin = np.amin(hist)
    histmax = np.amax(hist)
    histsoftmax = np.quantile(hist.astype(float), 0.99)
    hasnegative = histmin < -1e-6
    if aspect is None: aspect = 'equal'
    if extent is not None: aspect = 'auto'
    # if a box aspect is given (height/width of the plotted area, e.g. 1 for a square),
    # let the pixels stretch to fill a box of that shape, regardless of the input dimensions
    if box_aspect is not None:
        aspect = 'auto'
        ax.set_box_aspect(box_aspect)

    # make color object
    if not hasnegative:
        vmin = 1e-6
        vmax = max(vmin*2, histmax)
        if caxsoftmax: vmax = max(vmin*2, histsoftmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    else:
        extremum = max(abs(histmax), abs(histmin))
        norm = mpl.colors.Normalize(vmin=-extremum, vmax=extremum, clip=False)
    if caxrange is not None:
        norm = mpl.colors.Normalize(vmin=caxrange[0], vmax=caxrange[1], clip=False)
    cmap = copy(plt.get_cmap('jet'))
    cmap.set_under('w')
    cobject = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # needed for colorbar
    cobject.set_array([]) # ad-hoc bug fix

    # make the plot
    ax.imshow(hist, aspect=aspect, interpolation='none', norm=norm, cmap=cmap,
              extent=extent, origin=origin)

    # add the colorbar
    if docolorbar:
        cbar = add_colorbar(fig, ax, cobject)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)

    # add titles
    if ticklabelsize is not None: ax.tick_params(labelsize=ticklabelsize)
    if docolorbar and colorticklabelsize is not None: cbar.ax.tick_params(labelsize=colorticklabelsize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig, ax)


def plot_cluster_occupancy(hist, fig=None, ax=None, fast=False,
                           do_xaxtitle=True, xaxtitlesize=None,
                           do_yaxtitle=True, yaxtitlesize=None,
                           **kwargs):

    # default arguments
    default_kwargs = {
        'extent': None,
        'aspect': None,
        'box_aspect': 1,
        'origin': 'lower'
    }
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

    # make base plot
    fig, ax = plot_hist_2d(hist, fig=fig, ax=ax, **kwargs)

    # for faster plotting, don't bother with further aesthetics
    if fast: return fig, ax

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
        # FPix, raw
        xaxtitle = 'Disk'
        xaxticklabels = np.arange(-3, 4)
    elif nxbins==48:
        # FPix, preprocessed
        xaxtitle = 'Disk'
        xaxticklabels = np.concatenate((np.arange(-3, 0), np.arange(1, 4)))
    else: raise Exception('Number of x-bins not recognized.')

    # x-axis layout
    if do_xaxtitle: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    for x in axvlines: ax.axvline(x, color='black', linestyle='dotted')
    ax.set_xticks(xaxticks)
    ax.set_xticklabels(xaxticklabels)

    # automatically determine y-axis labels from number of bins
    nybins = hist.shape[0]
    if nxbins==72 or nxbins==64:
        # BPix
        axhlines = np.arange(2-0.5, nybins-0.5, 2)
        yaxticks = np.arange(1-0.5, nybins-0.5, 2)
    elif nxbins==56 or nxbins==48:
        # FPix
        axhlines = np.arange(4-0.5, nybins-0.5, 4)
        yaxticks = np.arange(2-0.5, nybins-0.5, 4)
    if nxbins==72:
        # BPix, raw
        yaxtitle = 'Ladder'
        nladders = int((nybins-2)/4)
        yaxticklabels = np.arange(-nladders, nladders+1)
    elif nxbins==64:
        # BPix, preprocessed
        yaxtitle = 'Ladder'
        nladders = int(nybins/4)
        yaxticklabels = np.concatenate((np.arange(-nladders, 0), np.arange(1, nladders+1)))
    elif nxbins==56:
        # FPix, raw
        yaxtitle = 'Panel'
        nladders = int((nybins-4)/8)
        yaxticklabels = np.arange(-nladders, nladders+1)
    elif nxbins==48:
        # FPix, preprocessed
        yaxtitle = 'Panel'
        nladders = int(nybins/8)
        yaxticklabels = np.concatenate((np.arange(-nladders, 0), np.arange(1, nladders+1)))
    else: raise Exception('Number of y-bins not recognized.')

    # subsample y-axis labels (else they are typically overlapping)
    stepdict = {6: 2, 14: 2, 22: 4, 32: 4, 17: 2}
    step = stepdict.get(nladders, 1)
    ids = np.sort(np.concatenate((np.arange(0, int((len(yaxticks)+1)/2), step),
                                  np.arange(len(yaxticks)-1, int(len(yaxticks)/2), -step))))
    yaxticks = yaxticks[ids]
    yaxticklabels = yaxticklabels[ids]

    # y-axis layout
    if do_yaxtitle: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    for y in axhlines: ax.axhline(y, color='black', linestyle='dotted')
    ax.set_yticks(yaxticks)
    ax.set_yticklabels(yaxticklabels)

    return fig, ax


def make_figure(orig, loss, binary, year, layer):

    # initialize figure
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(24, 8), squeeze=False)

    # plot raw data
    plot_cluster_occupancy(orig, fig=fig, ax=axs[0, 0],
                   title='Input', titlesize=25,
                   xaxtitlesize=25, yaxtitlesize=25,
                   ticklabelsize=20, colorticklabelsize=20,
                   caxsoftmax=True,
                   docolorbar=True, caxtitle='Number of clusters',
                   caxtitlesize=25, caxtitleoffset=30)
    # plot loss
    plot_cluster_occupancy(loss, fig=fig, ax=axs[0, 1],
                   title='Loss', titlesize=25,
                   xaxtitlesize=25, do_yaxtitle=False, yaxtitlesize=15,
                   ticklabelsize=20, colorticklabelsize=20,
                   docolorbar=True, caxtitle='Loss',
                   caxrange=(0, 0.1),
                   caxtitlesize=25, caxtitleoffset=25)
    # plot cleaned loss
    plot_cluster_occupancy(binary, fig=fig, ax=axs[0, 2],
                   title='Cleaned loss', titlesize=25,
                   xaxtitlesize=25, do_yaxtitle=False, yaxtitlesize=15,
                   ticklabelsize=20, colorticklabelsize=20,
                   docolorbar=True, caxtitle='Loss',
                   caxrange=(0, 1),
                   caxtitlesize=25, caxtitleoffset=25)

    # spacing
    fig.subplots_adjust(wspace=0.45)

    # title
    title = r'$\textbf{CMS}$'
    conditions = f'{year} (13.6 TeV)'
    axs[0, 0].text(0.01, 1.15, title, fontsize=25, transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.5, 1.15, layer, fontsize=25, ha='center', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.99, 1.15, conditions, fontsize=25, ha='right', transform=axs[0, 2].transAxes)

    return fig, axs


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Make pixel cluster occupancy example figure')
    parser.add_argument('--input', default='data_year2025_run392250_ls633_layerBPix1.npz',
                        help='Input .npz file with arrays orig, loss and binary')
    parser.add_argument('--outputname', default='pixel_example',
                        help='Output file name (without extension); .png and .pdf are written')
    parser.add_argument('--year', default='2025')
    parser.add_argument('--layer', default='BPix1')
    parser.add_argument('--nolatex', action='store_true',
                        help='Do not use LaTeX for text rendering (if no LaTeX installation is available)')
    parser.add_argument('--show', action='store_true', help='Show the figure interactively')
    args = parser.parse_args()

    # use latex for text formatting
    if not args.nolatex: plt.rc('text', usetex=True)

    # load data
    data = np.load(args.input)

    # make figure
    fig, axs = make_figure(data['orig'], data['loss'], data['binary'], args.year, args.layer)
    fig.savefig(f'{args.outputname}.png', bbox_inches='tight')
    fig.savefig(f'{args.outputname}.pdf', bbox_inches='tight')
    if args.show: plt.show()
    plt.close()
