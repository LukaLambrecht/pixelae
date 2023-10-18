import sys
import os
import numpy as np


def rebin_keep_zero(hist, newshape):
    ### rebin a histogram with exact zero propagation.
    # bins are averaged out in groups to obtain the specified new shape,
    # but if a group of bins contains a zero, the resulting average is set to zero as well.
    # references:
    # efficient numpy rebinning: https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    def mean_keep_zero(a, axis=None):
        ### calculate the mean value of an array but return zero if there are zeros in array
        # strategy: replace 0 by NaN, take mean (will preserve NaN), convert NaN back to 0.
        a = a.astype(float) # need to convert since np.nan is of type float
        a[a==0] = np.nan
        a = a.mean(axis=axis)
        np.nan_to_num(a, copy=False)
        a = a.astype(int) # convert back to integer type
        return a
    temp_shape = newshape[0], hist.shape[0]//newshape[0], newshape[1], hist.shape[1]//newshape[1]
    newhist = np.array(hist.reshape(temp_shape))
    newhist = mean_keep_zero(newhist, axis=-1)
    newhist = mean_keep_zero(newhist, axis=1)
    return newhist