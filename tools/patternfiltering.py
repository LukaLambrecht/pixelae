import sys
import os
import numpy as np
import cv2

# note: cv2 (opencv-python) not installed on SWAN by default,
#       installed manually with 'pip install opencv-python'.


def contains_pattern(hists, pattern, mask=None, threshold=1e-3):
    """
    Check whether a given pattern is present in a (set of) histogram(s).
    Returns:
    - 1D boolean array (same length as hists).
    """
    
    # parse threshold
    if threshold is None: threshold = 1e-3
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return contains_pattern(hists, pattern, mask=mask)[0]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
    
    # set masked-out positions to not match the pattern
    if mask is not None:
        hists[:,~mask] = np.min(pattern)-1
    
    # perform pattern matching
    res = np.zeros(len(hists)).astype(bool)
    for i, hist in enumerate(hists):
        M = cv2.matchTemplate(hist.astype(np.float32), pattern.astype(np.float32), cv2.TM_SQDIFF)
        if np.any(M<threshold): res[i] = True
    return res


def filter_pattern(hists, pattern, threshold=1e-3):
    """
    Make a mask corresponding to given patterns in a (set of) histogram(s)
    Returns:
    - boolean array of the same shape as hists,
      with True for each pixel that belongs to the pattern,
      and False elsewhere.
    """
    
    # parse threshold
    if threshold is None: threshold = 1e-3
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return filter_pattern(hists, pattern)[0, :, :]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
    
    # perform pattern matching
    res = np.zeros(hists.shape).astype(bool)
    for i, hist in enumerate(hists):
        M = cv2.matchTemplate(hist.astype(np.float32), pattern.astype(np.float32), cv2.TM_SQDIFF)
        rows, columns = np.where(M<threshold)
        for row, col in zip(rows, columns): res[i, row:row+pattern.shape[0], col:col+pattern.shape[1]] = True
    return res


def contains_any_pattern(hists, patterns, **kwargs):
    """
    Extension of contains_pattern for multiple patterns.
    """
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return contains_any_pattern(hists, patterns, **kwargs)[0]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
        
    # loop over patterns
    res = np.zeros((len(hists), len(patterns)))
    for i, pattern in enumerate(patterns):
        res[:,i] = contains_pattern(hists, pattern, **kwargs)
    return (np.sum(res, axis=1)>0)


def filter_any_pattern(hists, patterns, **kwargs):
    """
    Extension of filter_pattern for multiple patterns.
    """
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return filter_any_pattern(hists, patterns, **kwargs)[0, :, :]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
        
    # loop over patterns
    res = np.zeros(hists.shape).astype(bool)
    for i, pattern in enumerate(patterns):
        res = ((res) | (filter_pattern(hists, pattern, **kwargs)))
    return res