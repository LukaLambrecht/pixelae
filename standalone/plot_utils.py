#!/usr/bin/env python
# coding: utf-8

# **A collection of useful basic functions for plotting.**  


### imports

# external modules
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy
try: import imageio
except: print('WARNING: could not import package "imageio".' 
              +' This is only used to create gif animations, so it should be safe to proceed without,'
              +' if you do not plan to do just that.')
try:
    from matplotlib import rc
    rc('text', usetex=True)
    plot_utils_latex_formatting = True
except: 
    print('WARNING: could not set LaTEX rendering for matplotlib.'
          +' Any TEX commands in figure labels might not work as expected.')
    plot_utils_latex_formatting = False
import importlib


##################
# help functions #
##################

def make_legend_opaque( leg ):
    ### set the transparency of all entries in a legend to zero
    for lh in leg.legendHandles: 
        try: lh.set_alpha(1)
        except: lh._legmarker.set_alpha(1)
            
def add_text( ax, text, pos, 
              fontsize=10,
              horizontalalignment='left',
              verticalalignment='bottom',
              background_facecolor=None, 
              background_alpha=None, 
              background_edgecolor=None,
              **kwargs ):
    ### add text to an axis at a specified position (in relative figure coordinates)
    # input arguments:
    # - ax: matplotlib axis object
    # - text: string, can contain latex syntax such as /textbf{} and /textit{}
    # - pos: tuple with relative x- and y-axis coordinates of bottom left corner
    txt = None
    if( isinstance(ax, plt.Axes) ): 
        txt = ax.text(pos[0], pos[1], text, fontsize=fontsize, 
                   horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, 
                   transform=ax.transAxes, **kwargs)
    else:
        txt = ax.text(pos[0], pos[1], text, fontsize=fontsize,
                   horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
                   **kwargs)
    if( background_facecolor is not None 
            or background_alpha is not None 
            or background_edgecolor is not None ):
        if background_facecolor is None: background_facecolor = 'white'
        if background_alpha is None: background_alpha = 1.
        if background_edgecolor is None: background_edgecolor = 'black'
        txt.set_bbox(dict(facecolor=background_facecolor, 
                            alpha=background_alpha, 
                            edgecolor=background_edgecolor))
        
def add_cms_label( ax, pos=(0.1,0.9), extratext=None, **kwargs ):
    ### add the CMS label and extra text (e.g. 'Preliminary') to a plot
    # special case of add_text, for convenience
    text = r'\textbf{CMS}'
    if extratext is not None: text += r' \textit{'+str(extratext)+r'}'
    add_text( ax, text, pos, **kwargs)
    
def add_data_label( ax, datalabel, pos=(0.9,0.9), **kwargs ):
    ### add the data label, e.g. '2023 (13.6 TeV)', to a plot
    # special case of add_text, for convenience
    add_text( ax, datalabel, pos, horizontalalignment='right', **kwargs)

def make_text_latex_safe( text ):
    ### make a string safe to process with matplotlib's latex parser in case no tex parsing is wanted
    # (e.g. escape underscores)
    # to be extended when the need arises!
    if not plot_utils_latex_formatting: return
    text = text.replace('_','\_')
    return text


########################################
# functions for plotting 1D histograms #
########################################
      
def plot_hists(histlist, fig=None, ax=None, colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1),
              title=None, titlesize=None, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
              ymaxfactor=None, legendsize=None, opaque_legend=False, ticksize=None,
              bkgcolor=None, bkgcmap='spring', bkgcolorbar=False, bkgrange=None, bkgtitle=None):
    ### plot some histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing colors (in string format), of length nhistograms
    #   note: it can also be a single string representing a color (in pyplot), then all histograms will take this color
    # - labellist is a list or array containing labels for in legend, of length nhistograms
    # - xlims is a tuple of min and max for the x-axis labels, defaults to (-0.5,nbins-0.5)
    # - title, xaxtitle, yaxtitle: strings for histogram title, x-axis title and y-axis title respectively
    # - bkgcolor: 1D array representing background color for the plot 
    #             (color axis will be scaled between min and max in bkgcolor)
    #   note: if bkgcolor does not have the same length as the x-axis, it will be compressed or stretched to fit the axis,
    #         but this might be meaningless, depending on what you are trying to visualize!
    # - bkgmap: name of valid pyplot color map for plotting the background color
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    if fig is None or ax is None: fig,ax = plt.subplots()
    dolabel = True; docolor = True
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    if isinstance(colorlist,str):
        colorlist = [colorlist]*len(histlist)
    if len(colorlist)==0:
        docolor = False
    if xlims[1]<xlims[0]: xlims = (-0.5,len(histlist[0])-0.5)
    xax = np.linspace(xlims[0],xlims[1],num=len(histlist[0]))
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,where='mid',color=colorlist[i],label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,where='mid',label=labellist[i],alpha=transparency)
    if bkgcolor is not None:
        # modify bkcolor so the automatic stretching matches the bin numbers correctly
        bkgcolor = [el for el in bkgcolor for _ in (0,1)][1:-1]
        bkgcolor = np.array(bkgcolor)
        if bkgrange is None: bkgrange=(np.min(bkgcolor),np.max(bkgcolor))
        ax.pcolorfast((xlims[0],xlims[1]), ax.get_ylim(),
              bkgcolor[np.newaxis],
              cmap=bkgcmap, alpha=0.1,
              vmin=bkgrange[0], vmax=bkgrange[1])
        # add a color bar
        if bkgcolorbar:
            norm = mpl.colors.Normalize(vmin=bkgrange[0], vmax=bkgrange[1])
            cobject = mpl.cm.ScalarMappable(norm=norm, cmap=bkgcmap)
            cobject.set_array([]) # ad-hoc bug fix
            cbar = fig.colorbar(cobject, ax=ax, alpha=0.1)
            if bkgtitle is not None:
                cbar.ax.set_ylabel(bkgtitle, fontsize=yaxtitlesize,
                                   rotation=270, labelpad=20.)
    if ymaxfactor is not None:
        ymin,ymax = ax.get_ylim()
        ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if dolabel: 
        leg = ax.legend(loc='upper right', fontsize=legendsize)
        if opaque_legend: make_legend_opaque(leg)
    if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig,ax)

def plot_hists_from_df(df, histtype, nhists):
    ### plot a number of histograms in a dataframe
    # - df is the dataframe from which to plot
    # - histtype is the name of the histogram type (e.g. 'chargeInner_PXLayer_1')
    # - nhists is the number of histograms to plot
    dfs = select_histnames(df,[histtype])
    nhists = min(len(dfs),nhists)
    dfs = dfs[0:nhists+1]
    val = get_hist_values(dfs)[0]
    plot_hists(val)
    
def plot_hists_multi(histlist, fig=None, ax=None, figsize=None,
                     colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1),
                     title=None, titlesize=None, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
                     caxtitle=None, caxtitlesize=None, caxtitleoffset=None, hidecaxis=False,
                     extratext=None, extratextsize=None,
                     remove_underflow=False, remove_overflow=False,
                     ylims=None, ymaxfactor=None, legendsize=None, opaque_legend=False,
                     ticksize=None ):
    ### plot many histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing numbers to be mapped to colors
    # - labellist is a list or array containing labels for in legend
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    if fig is None or ax is None: fig,ax = plt.subplots(figsize=figsize)
    dolabel = True; docolor = True
    # make label list for legend
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    # make color list
    if len(colorlist)==0:
        docolor = False
    # make x-axis
    nbins = len(histlist[0])
    if remove_underflow: nbins -= 1
    if remove_overflow: nbins -= 1
    if xlims[1]<xlims[0]: xlims = (0,nbins)
    xax = np.linspace(xlims[0],xlims[1],num=nbins)
    # remove under- and overflow
    if remove_underflow: histlist = histlist[:,1:]
    if remove_overflow: histlist = histlist[:,:-1]
    # make color map
    if docolor:
        norm = mpl.colors.Normalize(vmin=np.min(colorlist),vmax=np.max(colorlist))
        cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cobject.set_array([]) # ad-hoc bug fix
    # loop over histograms
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,where='mid',color=cobject.to_rgba(colorlist[i]),label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,where='mid',label=labellist[i],alpha=transparency)
    if( docolor and not hidecaxis ): 
        cbar = fig.colorbar(cobject, ax=ax)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)
    if ymaxfactor is not None:
        ymin,ymax = ax.get_ylim()
        ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if ylims is not None:
        ax.set_ylim( ylims )
    if dolabel: 
        leg = ax.legend(loc='upper right', fontsize=legendsize)
        if opaque_legend: make_legend_opaque(leg)
    if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if extratext is not None: 
        add_text( ax, extratext, (0.95,0.6), fontsize=extratextsize, horizontalalignment='right' )
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)      
    return (fig,ax)
    
def plot_sets(setlist, fig=None, ax=None, colorlist=[], labellist=[], transparencylist=[],
             title=None, titlesize=None,
             extratext=None, extratextsize=None,
             xaxtitle=None, xaxtitlesize=None, xlims=(-0.5,-1), 
             remove_underflow=False, remove_overflow=False,
             yaxtitle=None, yaxtitlesize=None, ylims=None, ymaxfactor=None, 
             legendsize=None, opaque_legend=False, ticksize=None):
    ### plot multiple sets of 1D histograms to compare the shapes
    # - setlist is a list of 2D numpy arrays containing histograms
    # - fig and ax: a pyplot figure and axis object (if one of both is none a new figure is created)
    # - title is a string that will be used as the title for the ax object
    # other parameters are lists of which each element applies to one list of histograms

    # check for empty arrays
    for i,hists in enumerate(setlist):
        if hists.shape[0]==0:
            raise Exception('ERROR in plot_utils.py / plot_sets:'
                    +' the {}th histogram set is empty, '.format(i)
                    +' this is currently not supported for plotting')
    # parse arguments
    dolabel = True
    if len(labellist)==0:
        labellist = ['']*len(setlist)
        dolabel = False
    if len(colorlist)==0:
        colorlist = ['red','blue','green','orange']
        if len(setlist)>4:
            raise Exception('ERROR in plot_utils.py / plot_sets: '
                    +'please specify the colors if you plot more than four sets.')
    if len(transparencylist)==0:
        transparencylist = [1.]*len(setlist)
    # make x axis
    nbins = len(setlist[0][0])
    if remove_underflow: nbins -= 1
    if remove_overflow: nbins -= 1
    if xlims[1]<xlims[0]: xlims = (0,nbins)
    xax = np.linspace(xlims[0],xlims[1],num=nbins)
    # create the figure
    if fig is None or ax is None: fig,ax = plt.subplots()
    # loop over sets
    for i,histlist in enumerate(setlist):
        if remove_underflow: histlist = histlist[:,1:]
        if remove_overflow: histlist = histlist[:,:-1]
        row = histlist[0]
        ax.step(xax,row,where='mid',color=colorlist[i],label=labellist[i],alpha=transparencylist[i])
        if len(histlist)<2: continue
        for j,row in enumerate(histlist[1:,:]):
            ax.step(xax,row,where='mid',color=colorlist[i],alpha=transparencylist[i])
    if ymaxfactor is not None:
        ymin,ymax = ax.get_ylim()
        ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if ylims is not None:
        ax.set_ylim( ylims )
    if dolabel: 
        leg = ax.legend(loc='upper right', fontsize=legendsize)
        if opaque_legend: make_legend_opaque(leg)
    if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if extratext is not None: 
        add_text( ax, extratext, (0.95,0.6), fontsize=extratextsize, horizontalalignment='right' )
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig,ax)

def plot_anomalous(histlist, ls, highlight=-1, hrange=-1):
    ### plot a range of 1D histograms and highlight one of them
    # input arguments:
    # - histlist and ls: a list of histograms and corresponding lumisection numbers
    # - highlight: the lumisection number of the histogram to highlight
    # - hrange: the number of histograms before and after lsnumber to plot (default: whole run)
    lshist = None
    if highlight >= 0:
        if not highlight in ls:
            print('WARNING in plot_utils.py / plot_anomalous: requested lumisection number not in list of lumisections')
            return None
        index = np.where(ls==highlight)[0][0]
        lshist = histlist[index]
    if hrange > 0:
        indexmax = min(index+hrange,len(ls))
        indexmin = max(index-hrange,0)
        histlist = histlist[indexmin:indexmax]
        ls = ls[indexmin:indexmax]
    # first plot all histograms in the run
    fig,ax = plot_hists_multi(histlist,colorlist=ls,transparency=0.1)
    # now plot a single histogram on top
    if lshist is not None: 
        xlims = (0,len(lshist))
        xax = np.linspace(xlims[0],xlims[1],num=len(lshist))
        ax.step(xax,lshist,where='mid',color='black',linewidth=2)
    return (fig,ax)


########################################
# functions for plotting 2D histograms #
########################################

def plot_hist_2d(hist, fig=None, ax=None, figsize=None, title=None, titlesize=None,
                xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
                ticklabelsize=None, colorticklabelsize=None, extent=None, caxrange=None,
                docolorbar=True, caxtitle=None, caxtitlesize=None, caxtitleoffset=None,
                 origin='lower'):
    ### plot a 2D histogram
    # - hist is a 2D numpy array of shape (nxbins, nybins)
    # notes:
    # - if the histogram contains only nonnegative values, values below 1e-6 will not be plotted
    #   (i.e. they will be shown as white spots in the plot) to discriminate zero from small but nonzero
    # - if the histogram contains negative values, the color axis will be symmetrized around zero.
    # - the default behaviour of imshow() is to flip the axes w.r.t. numpy convention
    #   (i.e. the first axis is the y-axis instead of the x-axis),
    #   and to have the y-axis pointing downwards;
    #   both effects are fixed by transposing the array and using the 'lower' origin keyword.
    if fig is None or ax is None: fig,ax = plt.subplots(figsize=figsize)
    
    # settings
    histmin = np.amin(hist)
    histmax = np.amax(hist)
    hasnegative = histmin<-1e-6
    aspect_ratio = hist.shape[0]/hist.shape[1]
    aspect = 'equal'
    if extent is not None: aspect = 'auto'   
        
    # make color object
    if not hasnegative:
        vmin = 1e-6
        vmax = max(vmin*2,histmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    else: 
        extremum = max(abs(histmax),abs(histmin))
        norm = mpl.colors.Normalize(vmin=-extremum, vmax=extremum, clip=False)
    if caxrange is not None:
        norm = mpl.colors.Normalize(vmin=caxrange[0], vmax=caxrange[1], clip=False)
    cmap = copy(mpl.cm.get_cmap('jet'))
    cmap.set_under('w')
    cobject = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # needed for colorbar
    cobject.set_array([]) # ad-hoc bug fix
    
    # make the plot
    ax.imshow(hist.T, aspect=aspect, interpolation='none', norm=norm, cmap=cmap,
              extent=extent, origin=origin)
    
    # add the colorbar
    # it is not straightforward to position it properly;
    # the 'magic values' are fraction=0.046 and pad=0.04, but have to be modified by aspect ratio;
    # for this, use the fact that imshow uses same scale for both axes, 
    # so can use array aspect ratio as proxy
    if docolorbar:
        fraction = 0.046; pad = 0.04
        fraction *= aspect_ratio
        pad *= aspect_ratio
        cbar = fig.colorbar(cobject, ax=ax, fraction=fraction, pad=pad)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)
    
    # add titles
    if ticklabelsize is not None: ax.tick_params(labelsize=ticklabelsize)
    if colorticklabelsize is not None: cbar.ax.tick_params(labelsize=colorticklabelsize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig,ax)

def plot_hists_2d(hists, ncols=4, axsize=5, title=None, titlesize=None,
                    subtitles=None, subtitlesize=None, xaxtitles=None, yaxtitles=None,
                    **kwargs):
    ### plot multiple 2D histograms next to each other
    # input arguments
    # - hists: list of 2D numpy arrays of shape (nxbins,nybins), or an equivalent 3D numpy array
    # - ncols: number of columns to use
    # - figsize: approximate size of a single axis in the figure
    #            (will be modified by aspect ratio)
    # - title, titlesize: properties of the super title for the entire figure
    # - subtitles, subtitlesize: properties of the individual histogram titles
    # - xaxtitles, yaxtitles: properties of axis titles of individual histograms
    # - kwargs: passed down to plot_hist_2d

    # check for empty array
    if len(hists)==0:
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                    +' the histogram set is empty, '
                    +' this is currently not supported for plotting')

    # check arugments
    if( subtitles is not None and len(subtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                +' subtitles must have same length as hists or be None')
    if( xaxtitles is not None and len(xaxtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                +' xaxtitles must have same length as hists or be None')
    if( yaxtitles is not None and len(yaxtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                +' yaxtitles must have same length as hists or be None')

    # initialize number of rows
    nrows = int(math.ceil(len(hists)/ncols))

    # calculate the aspect ratio of the plots and the size of the figure
    shapes = []
    for hist in hists: shapes.append( hist.shape )
    aspect_ratios = [el[0]/el[1] for el in shapes]
    aspect_ratio_max = max(aspect_ratios)
    aspect_ratio_min = min(aspect_ratios)
    aspect_ratio = 1
    if aspect_ratio_min > 1: aspect_ratio = aspect_ratio_min
    if aspect_ratio_max < 1: aspect_ratio = aspect_ratio_max
    figsize=None
    aspect_ratio *= 0.9 # correction for color bar
    if aspect_ratio>1: figsize = (axsize*ncols,axsize*nrows*aspect_ratio)
    if aspect_ratio<1: figsize = (axsize*ncols/aspect_ratio,axsize*nrows)

    # initalize the figure
    fig,axs = plt.subplots(nrows,ncols,figsize=figsize,squeeze=False)
    
    # loop over all histograms belonging to this lumisection and make the plots
    for i,hist in enumerate(hists):
        subtitle = None
        xaxtitle = None
        yaxtitle = None
        if subtitles is not None: subtitle = subtitles[i]
        if xaxtitles is not None: xaxtitle = xaxtitles[i]
        if yaxtitles is not None: yaxtitle = yaxtitles[i]
        # make the plot
        plot_hist_2d(hist, fig=fig,ax=axs[int(i/ncols),i%ncols],
                title=subtitle, titlesize=subtitlesize, xaxtitle=xaxtitle, yaxtitle=yaxtitle, 
                **kwargs)
    
    # add a title
    if title is not None: fig.suptitle(title, fontsize=titlesize)
    
    # return the figure and axes
    return (fig,axs)

def plot_hists_2d_gif( hists, 
                       titles=None, xaxtitle=None, yaxtitle=None,
                       duration=300, figname='temp_gif.gif',
                       mode='imageio',
                       **kwargs):
    # manage backend
    if mode=='imageio':
        try: import imageio
        except: raise Exception('ERROR: could not import imageio')
    elif mode=='pillow':
        try: from PIL import Image
        except: raise Exception('ERROR: could not import PIL')
    else: raise Exception('ERROR: mode {} not recognized'.format(mode))
    # make individual images
    nhists = len(hists)
    filenames = []
    for i in range(nhists):
        title = None
        if titles is not None: title = titles[i]
        fig,_ = plot_hist_2d(hists[i], title=title, xaxtitle=xaxtitle, yaxtitle=yaxtitle, **kwargs)
        filename = 'temp_gif_file_{}.png'.format(i)
        filenames.append(filename)
        fig.savefig(filename, facecolor='white', transparent=False)
        plt.close()
    # convert to gif
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
        
        
################################################################
# functions for plotting moments and distances in moment space #
################################################################

def plot_moments(moments, ls, dims=(0,1), 
                 fig=None, ax=None, markersize=10,
                 xaxtitle='auto', xaxtitlesize=12,
                 yaxtitle='auto', yaxtitlesize=12,
                 zaxtitle='auto', zaxtitlesize=12,
                 caxtitle=None, caxtitlesize=12, caxtitleoffset=15,
                 ticksize=None):
    ### plot the moments of a set of histograms
    # input arguments:
    # - moments: a numpy array of shape (nhists,nmoments)
    # - dims: a tuple of two or three values between 0 and nmoments-1
    from mpl_toolkits.mplot3d import Axes3D # specific import
    if fig==None: fig = plt.figure()
    if len(dims)==2:
        if ax==None: ax = fig.add_subplot(111)
        scpl = ax.scatter(moments[:,dims[0]],moments[:,dims[1]],s=markersize,c=ls,cmap='jet')
        cbar = plt.colorbar(scpl)
        if xaxtitle is not None:
            if xaxtitle=='auto': xaxtitle = 'Moment '+str(dims[0]+1)
            ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
        if yaxtitle is not None:
            if yaxtitle=='auto': yaxtitle = 'Moment '+str(dims[1]+1)
            ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)
        if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    elif len(dims)==3:
        if ax==None: ax = fig.add_subplot(111, projection='3d')
        scpl = ax.scatter(moments[:,dims[0]],moments[:,dims[1]],moments[:,dims[2]],s=markersize,c=ls,cmap='jet')
        plt.colorbar(scpl)
        if xaxtitle is not None:
            if xaxtitle=='auto': xaxtitle = 'Moment '+str(dims[0]+1)
            ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
        if yaxtitle is not None:
            if yaxtitle=='auto': yaxtitle = 'Moment '+str(dims[1]+1)
            ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
        if zaxtitle is not None:
            if zaxtitle=='auto': zaxtitle = 'Moment '+str(dims[2]+1)
            ax.set_zlabel(zaxtitle, fontsize=zaxtitlesize)
    return (fig,ax)
