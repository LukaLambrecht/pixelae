# define NMF model for 2D input arrays

import copy
import numpy as np
from sklearn.decomposition import MiniBatchNMF

class NMF2D(object):
    
    def __init__(self, **kwargs):
        self.nmf = MiniBatchNMF(**kwargs)
        self.xshape = None
        self.components = None
        
    def fit(self, X):
        # fit components
        self.xshape = list(X.shape[1:])
        X = np.reshape(X, (X.shape[0], -1))
        self.nmf.partial_fit(X)
        # post-processing: scaling
        means = np.mean(self.nmf.components_, axis=1)
        self.nmf.components_ = np.divide(self.nmf.components_, means[:, None])
        # post-processing: reshape back to input dimensions
        self.components = np.reshape(self.nmf.components_, (-1, *self.xshape))
        
    def predict(self, X):
        X = np.reshape(X, (X.shape[0], -1))
        Y = self.nmf.inverse_transform(self.nmf.transform(X))
        Y = np.reshape(Y, (-1, *self.xshape))
        return Y
    
    @staticmethod
    def from_other(other):
        new = NMF2D()
        new.nmf = copy.deepcopy(other.nmf)
        new.xshape = copy.deepcopy(other.xshape)
        new.components = copy.deepcopy(other.components)
        return new