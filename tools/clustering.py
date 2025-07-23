import sys
import os
import numpy as np
import cv2

# note: cv2 (opencv-python) not installed on SWAN by default,
#       installed manually with 'pip install opencv-python'.

sys.path.append(os.path.dirname(__file__))
from patternfiltering import filter_pattern


def cluster_loss_multithreshold(losses, thresholds, **kwargs):
    """
    Cluster loss using a multi-threshold approach.
    Apply the criterion "loss_pattern in (loss > loss_threshold)" for a range of thresholds and patterns. 
    Input arguments:
    - losses: 3D array of shape (number of instances, first axis, second axis)
    - thresholds: a list of dicts. Each dict must have the following content:
        - "loss_threshold": scalar, binary threshold to apply.
        - "pattern": 2D np array of integers (0 or 1), pattern to use in matching.
        - "filter_threshold": scalar, tolerance to use in matching.
    - kwargs: passed down to filter_pattern
    """
    
    # convert to np array
    losses = np.array(losses)
    
    # handle case of single histogram
    if(len(losses.shape)==2):
        losses = np.expand_dims(losses, axis=0)
        return cluster_loss_multithreshold(losses, thresholds, **kwargs)[0, :, :]
    
    # check dimension
    if(len(losses.shape)!=3):
        raise Exception('ERROR: a 3D array is expected, but found shape {}'.format(losses.shape))
        
    # loop over patterns
    res = np.zeros(losses.shape).astype(bool)
    for i, threshold in enumerate(thresholds):
        loss_threshold = threshold['loss_threshold']
        pattern = np.array(threshold['pattern'])
        filter_threshold = threshold['filter_threshold']
        binary_losses = (losses > loss_threshold).astype(int)
        res = ((res) | (filter_pattern(binary_losses, pattern, threshold=filter_threshold, **kwargs)))
    return res