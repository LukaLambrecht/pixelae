# external modules
import os
import sys
import json
import numpy as np
import pandas as pd

# local modules
thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
import tools.dftools as dftools
import tools.omstools as omstools


def make_default_preprocessor(era, layer):

    # set directory to normalization data
    normdata_dir = os.path.join(os.path.dirname(__file__), 'normdata')

    # load norm json
    metype = f'PXLayer_{layer}'
    normfile = os.path.join(normdata_dir, f'normdata_Run2024{era}_{metype}.json')
    with open(normfile, 'r') as f:
        norm_info = json.load(f)
    
    # divide the norm by the number of bins in order to normalize mean instead of sum
    if metype=='PXLayer_1': shape = (26, 72)
    elif metype=='PXLayer_2': shape = (58, 72)
    elif metype=='PXLayer_3': shape = (90, 72)
    elif metype=='PXLayer_4': shape = (130, 72)
    nbins = shape[0] * shape[1]
    norm_info['norm'] = [val / nbins for val in norm_info['norm']]

    # get the average occupancy map
    avgmefile = os.path.join(normdata_dir, f'avgme_Run2024{era}_{metype}.npy')
    avgme = np.load(avgmefile)

    # make a preprocessor
    preprocessor = PreProcessor(metype, global_norm=norm_info, local_norm=avgme)
    return preprocessor


class PreProcessor(object):
    '''
    Definition and application of preprocessing steps
    '''
    
    def __init__(self, metype, global_norm=None, local_norm=None, mask=None):
        '''
        Input arguments:
        - metype: type of ME.
          currently supported values: "PXLayer_[1, 2, 3, 4]".
        - global_norm: dict of the following form:
            {"run_number": [...],
             "lumisection_number": [...],
             "norm": [...]}
          used for global normalization by dividing each ME
          by the corresponding norm value.
        - local_norm: np array of the same shape as the ME,
          used for local normalization by dividing each ME
          bin-by-bin by this array.
        - mask: np array of the same shape as the ME,
          used for masking known problematic regions.
          note: not recommended for input data, better to mask loss directly.
        '''
        
        # copy metype attribute
        self.metype = metype
        metypes = ['PXLayer_1', 'PXLayer_2', 'PXLayer_3', 'PXLayer_4']
        if not self.metype in metypes:
            msg = f'ME type "{self.metype}" not recognized;'
            msg += f' choose from {metypes}.'
            raise Exception(msg)
            
        # set cropping properties based on type of ME
        self.anticrop = None
        if self.metype=='PXLayer_1': self.anticrop = (slice(12, 14), slice(32,40))
        elif self.metype=='PXLayer_2': self.anticrop = (slice(28, 30), slice(32,40))
        elif self.metype=='PXLayer_3': self.anticrop = (slice(44, 46), slice(32,40))
        elif self.metype=='PXLayer_4': self.anticrop = (slice(64, 66), slice(32,40))
        
        # copy global norm attribute
        self.global_norm = global_norm
        
        # set dimensions for checking
        required_dims = None
        if self.metype=='PXLayer_1': required_dims = (26, 72)
        elif self.metype=='PXLayer_2': required_dims = (58, 72)
        elif self.metype=='PXLayer_3': required_dims = (90, 72)
        elif self.metype=='PXLayer_4': required_dims = (130, 72)
        
        # parse local norm attribute
        self.local_norm = local_norm
        if self.local_norm is not None:
            
            # check dimensions
            if required_dims is not None and self.local_norm.shape != required_dims:
                msg = f'Local norm has shape {self.local_norm.shape} while {required_dims}'
                msg += f' was expected for ME type {metype}'
                raise Exception(msg)
                
            # do cropping
            if self.anticrop is not None:
                slicey = self.anticrop[0]
                slicex = self.anticrop[1]
                self.local_norm = np.delete(self.local_norm, slicey, axis=0)
                self.local_norm = np.delete(self.local_norm, slicex, axis=1)
                
            # safety for divide by zero
            # note: values are set to negative,
            # so they can later be recognized and set to zero.
            self.local_norm[self.local_norm==0] = -1
            
        # parse mask attribute
        self.mask = mask
        if self.mask is not None:
            
            # check dimensions
            if required_dims is not None and self.mask.shape != required_dims:
                msg = f'Mask has shape {self.mask.shape} while {required_dims}'
                msg += f' was expected for ME type {metype}'
                raise Exception(msg)
                
            # do cropping
            if self.anticrop is not None:
                slicey = self.anticrop[0]
                slicex = self.anticrop[1]
                self.mask = np.delete(self.mask, slicey, axis=0)
                self.mask = np.delete(self.mask, slicex, axis=1)
            
    
    def preprocess(self, df, **kwargs):
        '''
        Preprocess the data in a dataframe.
        Input arguments:
        - df: dataframe as read from input files.
        Returns:
        - np array of shape (number of histograms, number of y-bins, number of x-bins)
        '''
        
        # get histograms
        mes, runs, lumis = dftools.get_mes(df,
                            xbinscolumn='x_bin', ybinscolumn='y_bin',
                            runcolumn='run_number', lumicolumn='ls_number')
        
        # do preprocessing
        return self.preprocess_mes(mes, runs, lumis, **kwargs)

        
    def preprocess_mes(self, mes, runs, lumis, verbose=False):
        
        # initialize info for later printing
        preprocessing_info = {'run_number': runs, 'ls_number': lumis}
    
        # remove empty cross
        if self.anticrop is not None:
            slicey = self.anticrop[0]
            slicex = self.anticrop[1]
            mes = np.delete(mes, slicey, axis=1)
            mes = np.delete(mes, slicex, axis=2)
        
        # do global normalization
        if self.global_norm is not None:
            norm = omstools.find_oms_attr_for_lumisections(runs, lumis,
                     self.global_norm, 'norm',
                     run_key='run_number', lumi_key='lumisection_number',
                     verbose=False)
            norm[norm==0] = 1
            mes = np.divide(mes, norm[:, None, None])
            preprocessing_info['norm'] = norm
            
        # do local normalization
        if self.local_norm is not None:
            mes = np.divide(mes[:], self.local_norm)
            
        # do masking
        if self.mask is not None:
            mes[:, ~self.mask] = 0
        
        # set negative values to zero
        mes[mes < 0] = 0
        
        # printouts if requested
        if verbose:
            preprocessing_info = pd.DataFrame(preprocessing_info)
            print(preprocessing_info)
        
        # return the mes
        return mes
    
    
    def deprocess(self, mes, runs=None, lumis=None):
        '''
        Inverse operation of preprocess
        '''
        
        # local normalization
        if self.local_norm is not None:
            mes = np.multiply(mes[:], self.local_norm)
            
        # global normalization
        if self.global_norm is not None:
            if runs is None or lumis is None:
                msg = 'Cannot perform inverse preprocessing operation without run and lumisection numbers.'
                raise Exception(msg)
            norm = omstools.find_oms_attr_for_lumisections(runs, lumis,
                     self.global_norm, 'norm',
                     run_key='run_number', lumi_key='lumisection_number')
            norm[norm==0] = 1
            mes = np.multiply(mes, norm[:, None, None])
            
        # insert empty cross
        if self.anticrop is not None:
            slicey = self.anticrop[0]
            slicex = self.anticrop[1]
            mes = np.insert(mes, [int(mes.shape[1]/2)]*2, 0, axis=1)
            mes = np.insert(mes, [int(mes.shape[2]/2)]*8, 0, axis=2)
            
        return mes
