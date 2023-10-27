import sys
import os
import numpy as np


def rebin_keep_zero(hists, newshape):
    ### rebin histograms with exact zero propagation.
    # bins are averaged out in groups to obtain the specified new shape,
    # but if a group of bins contains a zero, the resulting average is set to zero as well.
    # references:
    # efficient numpy rebinning: https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return rebin_keep_zero(hists, newshape)[0]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
    
    # define utility function
    def mean_keep_zero(a, axis=None):
        ### calculate the mean value of an array but return zero if there are zeros in array
        # strategy: replace 0 by NaN, take mean (will preserve NaN), convert NaN back to 0.
        a = a.astype(float) # need to convert since np.nan is of type float
        a[a==0] = np.nan
        a = a.mean(axis=axis)
        np.nan_to_num(a, copy=False)
        a = a.astype(int) # convert back to integer type
        return a
    temp_shape = hists.shape[0], newshape[0], hists.shape[1]//newshape[0], newshape[1], hists.shape[2]//newshape[1]
    newhists = np.array(hists.reshape(temp_shape))
    newhists = mean_keep_zero(newhists, axis=-1)
    newhists = mean_keep_zero(newhists, axis=2)
    return newhists