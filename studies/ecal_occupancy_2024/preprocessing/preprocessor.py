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



def make_default_preprocessor(era, metype,
                              global_normalization = None,
                              local_normalization = None):
    '''
    Input arguments:
    - global_normalization: choose from the following:
        - None: no global normalization.
        - "avg": set the average value to 1.
    - local_normalization: choose from the following:
        - None: no local normalization.
        - "avg": normalize by average ME.
    '''
    
    # set directory to normalization data
    normdata_dir = os.path.join(os.path.dirname(__file__), 'normdata')

    # settings for global normalization
    avgunity = False
    if global_normalization is None: pass
    elif global_normalization=='avg': avgunity = True
    else:
        raise Exception(f'Global normalization "{global_normalization}" not recognized.')

    # settings for local normalization
    avgme = None
    if local_normalization is None: pass
    elif local_normalization=='avg':
        
        # get the average occupancy map
        avgmefile = os.path.join(normdata_dir, f'avgme_Run2024{era}_{metype}.npy')
        avgme = np.load(avgmefile)
    
    else:
        raise Exception(f'Local normalization "{local_normalization}" not recognized.')

    # make a preprocessor
    preprocessor = PreProcessor(metype, global_norm=None, local_norm=avgme, avgunity=avgunity)
    return preprocessor


class PreProcessor(object):
    '''
    Definition and application of preprocessing steps
    '''
    
    def __init__(self, metype,
                 global_norm = None,
                 local_norm = None,
                 mask = None,
                 avgunity = False):
        '''
        Input arguments:
        - metype: type of ME.
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
        - avgunity: rescale average value to unity (after all other preprocessing steps).
        '''
        
        # copy metype attribute
        self.metype = metype
            
        # set cropping properties based on type of ME
        self.anticrop = None
        
        # copy global norm attribute
        self.global_norm = global_norm
        
        # set dimensions for checking
        self.required_dims = None
        if self.metype=='EB': self.required_dims = (34, 72)
        elif self.metype=='EE+': self.required_dims = (20, 20)
        elif self.metype=='EE-': self.required_dims = (20, 20)
        
        # parse local norm attribute
        self.local_norm = local_norm
        if self.local_norm is not None:
            
            # check dimensions
            if self.required_dims is not None and self.local_norm.shape != self.required_dims:
                msg = f'Local norm has shape {self.local_norm.shape} while {self.required_dims}'
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
            if self.required_dims is not None and self.mask.shape != self.required_dims:
                msg = f'Mask has shape {self.mask.shape} while {self.required_dims}'
                msg += f' was expected for ME type {metype}'
                raise Exception(msg)
                
            # do cropping
            if self.anticrop is not None:
                slicey = self.anticrop[0]
                slicex = self.anticrop[1]
                self.mask = np.delete(self.mask, slicey, axis=0)
                self.mask = np.delete(self.mask, slicex, axis=1)
                
        # set other properties
        self.avgunity = avgunity
            
    
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
        
        # set average value to unity
        if self.avgunity:
            mes[mes==0] = np.nan
            norm = np.nanmean(mes, axis=(1,2))
            mes = np.nan_to_num(mes, nan=0)
            mes = np.divide(mes, norm[:, None, None])
            preprocessing_info['avg'] = norm
        
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