import os
import sys
import cv2
import numpy as np


def rebin_keep_flag(hists, newshape, flag, mode='np'):
    '''
    Rebin histograms with exact flag value propagation.
    Mostly intended as helper function for different concrete cases (see below).
    Input arguments:
    - mode: choose from the following options:
        - "np": pure numpy implementation, exact and transparent,
          but only works for integer downscaling factors.
        - "cv2": a little black magic from the opencv (cv2) library,
          works with non-integer downscaling factors.
    '''
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return rebin_keep_flag(hists, newshape, flag, mode=mode)[0]
    
    # check dimension of input array
    if(len(hists.shape)!=3):
        raise Exception('A 3D array is expected, but found shape {}'.format(hists.shape))
        
    # check divisibility
    if mode=='np':
        if( hists.shape[1] % newshape[0] != 0
            or hists.shape[2] % newshape[1] != 0):
            msg = f'Incompatible shapes: input has shape {hists.shape}'
            msg += f' but requested new shape ({newshape}) is not a divisor,'
            msg += f' which is incompatible with mode "np"; try mode "cv2".'
            raise Exception(msg)
    
    # handle each mode
    if mode=='np':
    
        # define utility function
        def mean_keep_flag(a, flag, axis=None):
            ### calculate the mean value of an array but return flag if there are flags in the array
            # strategy: replace flag by NaN, take mean (will preserve NaN), convert NaN back to flag.
            a = a.astype(float) # need to convert since np.nan is of type float
            a[a==flag] = np.nan
            a = a.mean(axis=axis)
            np.nan_to_num(a, nan=flag, copy=False)
            # note: do not convert back to int type,
            # as input histogram might be of type float
            # (e.g. averaged or preprocessed)
            return a
        temp_shape = hists.shape[0], newshape[0], hists.shape[1]//newshape[0], newshape[1], hists.shape[2]//newshape[1]
        newhists = np.array(hists.reshape(temp_shape))
        newhists = mean_keep_flag(newhists, flag, axis=2)
        newhists = mean_keep_flag(newhists, flag, axis=3)
        return newhists
    
    elif mode=='cv2':
        # apparently cv2 switches the axis order in the provided newshape, so need to compensate for that
        newshape_cv2 = (newshape[1], newshape[0])
        hists = hists.astype(float)
        hists[hists==flag] = np.nan
        hists = np.array([cv2.resize(hist, newshape_cv2, interpolation=cv2.INTER_AREA) for hist in hists])
        np.nan_to_num(hists, nan=flag, copy=False)
        return hists
    
    
def rebin_keep_zero(hists, newshape, **kwargs):
    '''
    Rebin histograms with exact zero propagation.
    Bins are averaged out in groups to obtain the specified new shape,
    but if a group of bins contains a zero, the resulting average is set to zero as well.
    '''
    return rebin_keep_flag(hists, newshape, 0, **kwargs)


def rebin_keep_clip(hists, newshape, threshold, **kwargs):
    '''
    Rebin histograms with exact above-threshold propagation.
    Bins are averaged out in groups to obtain the specified new shape,
    but if a group of bins contains any value above the given threshold,
    the resulting average is set to that threshold as well.
    '''
    hists = np.minimum(hists, threshold)
    return rebin_keep_flag(hists, newshape, threshold, **kwargs)