import sys
import os
import numpy as np
import cv2

# note: cv2 (opencv-python) not installed on SWAN by default,
#       installed manually with 'pip install opencv-python'.


def contains_pattern(hists, pattern, mask=None):
    
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
        if np.any(M==0): res[i] = True
        rows, columns = np.where(M==0)
    return res

def contains_any_pattern(hists, patterns, mask=None):
    
    # convert to np array
    hists = np.array(hists)
    
    # handle case of single histogram
    if(len(hists.shape)==2):
        hists = np.expand_dims(hists, axis=0)
        return contains_any_pattern(hists, patterns, mask=mask)[0]
    
    # check dimension
    if(len(hists.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(hists.shape))
        
    # loop over patterns
    res = np.zeros((len(hists), len(patterns)))
    for i, pattern in enumerate(patterns):
        res[:,i] = contains_pattern(hists, pattern, mask=mask)
    return (np.sum(res, axis=1)>0)