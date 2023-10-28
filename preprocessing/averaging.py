import sys
import os
import numpy as np


def average_outer_ring(hists, refhists, radius=50):
    ### return histograms with the outer ring replaced by an average of the outer rings of refhists.
    # note: the inner ring (inside specified radius) is left unchanged.
    # note: the radius is given in number of bins.
    # note: refhists can be a fixed set of histograms (np array of shape (nhists, xdim, ydim))
    #       or an integer number (number of preceding histograms in hists to average over)
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return average_outer_ring(hists, refhists, radius=radius)[0]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
    
    # define a circular mask
    axis = np.arange(hists.shape[1])
    center = int(len(axis)/2)-0.5
    mask = (axis[np.newaxis,:]-center)**2 + (axis[:,np.newaxis]-center)**2 < radius**2
    
    # make result
    res = np.zeros(hists.shape).astype(float)
    # copy inner part
    res[:,mask] = hists[:,mask]
    # average outer part
    for i in range(len(hists)):
        thisrefhists = refhists
        if isinstance(thisrefhists, int):
            thisrefhists = hists[max(0,i-thisrefhists):i+1,:,:]
        res[i,~mask] = np.mean(np.array(thisrefhists), axis=0)[~mask]
    return res

def average_rings(hists, radii):
    ### generalization of average_outer_ring
    # note: radii is a dict of radii to numbers of preceding histograms to average over,
    #       in the convention that average_rings(hists, {radius: n})
    #       corresponds to average_outer_ring(hists, n, radius=radius)
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return average_rings(hists, radii)[0]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
    
    # define a ring shaped masks
    axis = np.arange(hists.shape[1])
    center = int(len(axis)/2)-0.5
    rings = {}
    r = sorted(list(radii.keys()))
    dist = (axis[np.newaxis,:]-center)**2 + (axis[:,np.newaxis]-center)**2
    for i in range(len(r)):
        mask = (dist > r[i]**2)
        if i<len(r)-1: mask = ((mask) & (dist < r[i+1]**2))
        rings[r[i]] = mask
    
    # make result
    res = np.copy(hists).astype(float)
    for i in range(len(hists)):
        for radius in r:
            mask = rings[radius]
            navg = radii[radius]
            thisrefhists = hists[max(0,i-navg):i+1,:,:]
            res[i,mask] = np.mean(np.array(thisrefhists), axis=0)[mask]
    return res