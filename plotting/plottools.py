# Tools for making plots


# imports
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
# local imports
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import utils.plot_utils
importlib.reload(utils.plot_utils)
import utils.plot_utils as pu


def plot_lumisection( hists, 
                      title=None, titlesize=None,
                      caxtitle=None, caxtitlesize=None,
                      vmin=1e-6, vmax='auto',
                      **kwargs):
    ### make a plot of all pixel endcap disks for one lumisection
    # note: maybe to be extended to barrel layers later, for now only endcaps
    # input arguments
    # - hists: dictionary of names to np arrays;
    #          names must be chosen from 'fpix-3', ..., 'fpix+3'
    # - kwargs: passed down to plot_hist_2d in plot_utils from ML4DQM repo

    # initializations
    nrows = 2
    ncols = 3
    figsize = (12,8)
    position_map = {
      'fpix-1': (0,0),
      'fpix-2': (0,1),
      'fpix-3': (0,2),
      'fpix+1': (1,0),
      'fpix+2': (1,1),
      'fpix+3': (1,2), 
    }
    subtitle_map = {
      'fpix-1': 'FPIX $-1$',
      'fpix-2': 'FPIX $-2$',
      'fpix-3': 'FPIX $-3$',
      'fpix+1': 'FPIX $+1$',
      'fpix+2': 'FPIX $+2$',
      'fpix+3': 'FPIX $+3$',
    }
    
    # determine suitable ranges for color map
    if vmax=='auto':
        hvalues = np.array(list(hists.values()))
        vmax = np.quantile(hvalues, 0.9)*1.2
        if vmax<vmin: vmax = vmin*2

    # initalize the figure
    fig,axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # loop over individual plots
    for key, pos in position_map.items():
        if key not in hists.keys(): continue
        hist = hists[key]
        ax = axs[pos]
        pu.plot_hist_2d(hist, fig=fig, ax=ax, 
                        title=subtitle_map[key], titlesize=titlesize,
                        xaxtitle='ix', xaxtitlesize=titlesize,
                        yaxtitle='iy', yaxtitlesize=titlesize,
                        docolorbar=False, caxrange=(vmin,vmax), **kwargs)
        
    # add a colorbar
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axs[0,0].images[0], cax=cax)
    cbar.ax.tick_params(labelsize=titlesize)
    if caxtitle is not None:
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)

    # add a title
    if title is not None: fig.suptitle(title, fontsize=titlesize, y=0.97)
        
    # add other labels
    pu.add_cms_label( axs[0,0], pos=(0.05,1.2), extratext='Preliminary', fontsize=titlesize )
    pu.add_data_label( axs[0,2], '2023 (13.6 TeV)', pos=(0.95,1.2), fontsize=titlesize )
    
    # adjust subplot spacing
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # return the figure and axes
    return (fig,axs)

def plot_lumisections_gif( frames, figname, 
                           titles=None,
                           duration=300,
                           verbose=False,
                           mode='imageio',
                           **kwargs ):
    ### same as plot_lumisection but for a series of lumisections (make a gif image)
    # input arguments:
    # - frames: a list of dicts, one element per frame, see plot_lumisection
    # - figname: name of output file to create
    # - titles: a list of titles, same length as frames
    # - duration: duration of each frame in milliseconds
    # - mode: choose from 'imageio' or 'pillow' (for the backend)
    # manage backend
    if mode=='imageio':
        try: import imageio
        except: raise Exception('ERROR: could not import imageio')
    elif mode=='pillow':
        try: from PIL import Image
        except: raise Exception('ERROR: could not import PIL')
    else: raise Exception('ERROR: mode {} not recognized'.format(mode))
    # make individual images
    nframes = len(frames)
    filenames = []
    for i in range(nframes):
        if verbose: print('Processing frame {} of {}'.format(i+1, nframes))
        title = None
        if titles is not None: title = titles[i]
        fig,_ = plot_lumisection(frames[i], title=title, **kwargs)
        filename = os.path.splitext(figname)[0]+'_temp{}.png'.format(i)
        filenames.append(filename)
        fig.savefig(filename, facecolor='white', transparent=False)
        plt.close()
    # convert to gif
    if verbose: print('Converting to gif...')
    if mode=='imageio':
        # first approach with imageio
        with imageio.get_writer(figname, mode='I', duration=duration, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
    elif mode=='pillow':
        # second approach with Pillow
        frames = [Image.open(filename) for filename in filenames]
        frame_one = frames[0]
        frame_one.save(figname, format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=0)
    # remove individual images
    for filename in filenames:
        os.remove(filename)
    if verbose: print('Done')