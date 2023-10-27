import sys
import os

from averaging import average_outer_ring
from rebinning import rebin_keep_zero


class PreProcessor(object):
    
    def __init__(self,
                 crop=None, # a tuple of slices (xdim and ydim)
                 n_time_averages=None, # number of preceding histograms to average over
                 time_average_radius=None, # radius used in time averaging
                 rebin_target=None # target shape for rebinning
                ):
        self.crop = crop
        self.n_time_averages = n_time_averages
        self.time_average_radius = time_average_radius
        if( self.n_time_averages is not None and self.time_average_radius is None ): self.time_average_radius = 0
        self.rebin_target = rebin_target
    
    def preprocess(self, histograms):
        ### preprocess a set of histograms
        # input arguments:
        # - histograms: 3D numpy array (nhistograms, xdim, ydim)
        
        # cropping
        if self.crop is not None:
            slicex = self.crop[0]
            slicey = self.crop[1]
            histograms = histograms[:, slicex, slicey]
            
        # average the outer part of the disk over some time steps
        if self.n_time_averages is not None:
            histograms = average_outer_ring(histograms, self.n_time_averages, self.time_average_radius)
    
        # rebinning
        if self.rebin_target is not None:
            histograms = rebin_keep_zero(histograms, self.rebin_target)
            
        return histograms