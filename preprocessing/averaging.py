import sys
import os
import numpy as np


def average_outer_ring(hist, refhists, radius=50):
    ### return a hist with the outer ring replaced by an average of the outer rings of refhists.
    # note: the inner ring (inside specified radius) is left unchanged.
    # note: refhists is usually the set of histograms in preceding/following lumisections.
    # note: the radius is given in number of bins.
    
    # define a circular mask
    axis = np.arange(hist.shape[0])
    center = int(len(axis)/2)-0.5
    mask = (axis[np.newaxis,:]-center)**2 + (axis[:,np.newaxis]-center)**2 < radius**2
    
    # make result
    res = np.zeros(hist.shape)
    res[mask] = hist[mask]
    res[~mask] = np.mean(np.array(refhists), axis=0)[~mask]
    return res