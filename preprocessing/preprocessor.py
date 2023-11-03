import sys
import os
import json
import numpy as np
import importlib

import averaging
import rebinning
import normalizing
importlib.reload(averaging)
importlib.reload(rebinning)
importlib.reload(normalizing)
from averaging import average_rings
from rebinning import rebin_keep_zero
from normalizing import normalize_by_omsjson


class PreProcessor(object):
    
    def __init__(self,
                 crop=None, # a tuple of slices (xdim and ydim)
                 time_average_radii=None, # dict of inner radii to number of time averages
                 rebin_target=None, # target shape for rebinning
                 omsjson=None, # json file or object with oms data
                 oms_normalization_attr=None, # oms attribute for normalization
                 spatial_normalization=None, # histogram for spatial normalization
                ):
        self.crop = crop
        self.time_average_radii = time_average_radii
        self.rebin_target = rebin_target
        self.omsjson = omsjson
        if isinstance(self.omsjson, str):
            with open(self.omsjson, 'r') as f:
                self.omsjson = json.load(f)
        self.oms_normalization_attr = oms_normalization_attr
        if self.oms_normalization_attr is not None:
            if self.omsjson is None: raise Exception('ERROR: omsjson must be specified if oms_normalization_attr is set.')
        self.spatial_normalization = spatial_normalization
        
    def __str__(self):
        ### get string representation
        # useful for documenting and keeping track what preprocessing was used
        res = 'PreProcessor instance with following attributes:\n'
        res += ' - crop: {}\n'.format(self.crop)
        res += ' - time_average_radii: {}\n'.format(self.time_average_radii)
        res += ' - rebin_target: {}\n'.format(self.rebin_target)
        omsjson_str = self.omsjson
        if omsjson_str is not None: omsjson_str = '<loaded>'
        res += ' - omsjson: {}\n'.format(omsjson_str)
        res += ' - oms_normalization_attr: {}\n'.format(self.oms_normalization_attr)
        spatial_normalization_str = self.spatial_normalization
        if isinstance(spatial_normalization_str, np.ndarray): spatial_normalization_str = '<ndarray>'
        res += ' - spatial_normalization: {}'.format(spatial_normalization_str)
        return res
    
    def preprocess(self, histograms, runs=None, lumis=None):
        ### preprocess a set of histograms
        # input arguments:
        # - histograms: 3D numpy array (nhistograms, xdim, ydim)
        
        # cropping
        if self.crop is not None:
            slicex = self.crop[0]
            slicey = self.crop[1]
            histograms = histograms[:, slicex, slicey]
            
        # average the outer part of the disk over some time steps
        if self.time_average_radii is not None:
            histograms = average_rings(histograms, self.time_average_radii)
    
        # rebinning
        if self.rebin_target is not None:
            histograms = rebin_keep_zero(histograms, self.rebin_target)
            
        # normalization
        if self.oms_normalization_attr is not None:
            if runs is None: raise Exception('ERROR: runs must be specified if oms_normalization_attr is set.')
            if lumis is None: raise Exception('ERROR: lumis must be specified if oms_normalization_attr is set.')
            histograms = normalize_by_omsjson(histograms, runs, lumis, self.omsjson, self.oms_normalization_attr)
        
        # spatial normalization
        if self.spatial_normalization is not None:
            avg_occupancy = self.spatial_normalization
            if( isinstance(self.spatial_normalization, str) and self.spatial_normalization=='auto' ):
                avg_occupancy = np.mean(histograms, axis=0)
            avg_occupancy[avg_occupancy==0] = 1
            histograms = np.divide(histograms[:], avg_occupancy)
            
        return histograms