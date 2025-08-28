# Model definition

# note: inside the container, only the files in the mlserver-model directory are visible.
#       importing local modules from outside this directory is not trivial
#       (although allegedly possible by making everything a package, but I did not test that yet),
#       and the simplest (though far from optimal) solution might be to duplicate all local dependencies
#       in this local model definition...


# import external modules
import os
import sys
import joblib
import numpy as np

# import local modules
import patternfiltering as patternfiltering
import rebinning as rebinning
import clustering as clustering
from nmf2d import NMF2D
from preprocessor import PreProcessor


class PixelNMF(object):
    # implementation of NMF model (including all pre- and post-processing steps)
    # for the pixel cluster occupancy monitoring elements.
    
    def __init__(self,
                 nmfs,
                 local_norms = None,
                 loss_masks = None,
                 threshold_patterns = None,
                 flagging_patterns = None
                 ):
        '''
        Initializer.
        Input arguments:
        - nmfs: dictionary of the following form: {monitoring element name: path to stored NMF model, ...}
        - local_norms: dictionary of the following form: {monitoring element name: local norm (2D np array), ...}
        - loss_masks: dictionary of the following form: {monitoring element name: loss mask (2D np array), ...}
        '''
        
        # get monitoring element names for later use
        self.menames = list(nmfs.keys())[:]
        
        # load NMF models
        # note: one model per monitoring element
        self.nmfs = {}
        for mename, nmffile in nmfs.items():
            nmf = joblib.load(nmffile)
            if not isinstance(nmf, NMF2D):
                msg = f'WARNING in PixelNMF.__init__: received object of type {type(nmf)},'
                msg += ' while an NMF2D model was expected; things might break further downstream.'
                print(msg)
            self.nmfs[mename] = nmf
            
        # make preprocessors
        self.preprocessors = {}
        for mename in self.menames:
            local_norm = None if local_norms is None else local_norms[mename]
            self.preprocessors[mename] = PreProcessor(mename, local_norm=local_norm, avgunity=True)
            
        # copy loss masks
        self.loss_mask = None
        if loss_masks is not None:
            self.loss_mask = self.make_combined_loss_mask(loss_masks)
            
        # copy thresholding and flagging patterns
        self.threshold_patterns = threshold_patterns
        if self.threshold_patterns is None:
            self.threshold_patterns = [
              {"loss_threshold": 0.04, "pattern": np.ones((2, 16)), "filter_threshold": 1.5}
            ]
        self.flagging_patterns = flagging_patterns
        if self.flagging_patterns is None:
            self.flagging_patterns = [np.ones((1, 8)), np.ones((2, 4))]
            
    def make_combined_loss_mask(self, loss_masks):
        '''
        Internal helper function to make combined loss mask
        from individual layer loss masks.
        '''
        
        # find target shape
        temp_preprocessor = PreProcessor(self.menames[0])
        temp_loss_mask = np.expand_dims(loss_masks[self.menames[0]], 0)
        temp_loss_mask = temp_preprocessor.preprocess_mes(temp_loss_mask, None, None)
        target_shape = temp_loss_mask.shape[1:3]
        
        # do loss masking
        loss_mask = np.zeros((1, target_shape[0], target_shape[1]))
        for mename in self.menames:
            this_loss_mask = loss_masks[mename]
            if this_loss_mask is None: continue
            # preprocess
            loss_mask_preprocessor = PreProcessor(mename)
            this_loss_mask = np.expand_dims(this_loss_mask, 0)
            this_loss_mask = loss_mask_preprocessor.preprocess_mes(this_loss_mask, None, None)
            # invert
            this_loss_mask = 1 - this_loss_mask
            # rescale
            this_loss_mask = rebinning.rebin_keep_clip(this_loss_mask, target_shape, 1, mode='cv2')
            # add to total
            loss_mask += this_loss_mask
        return loss_mask
    
    def preprocess(self, X, verbose=False):
        '''
        Do preprocessing (from raw MEs to input for the NMF model).
        Input arguments:
        - X: dictionary of the following form {monitoring element name: raw data (2D np array), ...}
        '''
        
        # loop over monitoring elements
        mes_preprocessed = {}
        if verbose: print('[INFO]: preprocessing...')
        for mename in self.menames:
            if verbose: print(f'  - {mename}')
            # do basic preprocessing using preprocessors defined earlier
            mes_preprocessed[mename] = self.preprocessors[mename].preprocess_mes(X[mename], None, None)
        return mes_preprocessed
            
    def infer(self, X, verbose=False):
        '''
        Run NMF inference/reconstruction.
        Input arguments:
        - X: dictionary of the following form {monitoring element name: preprocessed data (2D np array), ...}
        '''
        
        # loop over monitoring elements
        mes_reco = {}
        if verbose: print('[INFO]: running inference...')
        for mename in self.menames:
            if verbose: print(f'  - {mename}')
            X_input = np.copy(X[mename])
            # finetuning: clip very large values
            # to avoid the NMF model from compromising good agreement in most bins
            # for the sake of fitting slightly better a few spikes.
            # (note: this requires the preprocessing to contain some normalization,
            # otherwise the threshold is arbitrary.)
            threshold = 5
            X_input[X_input > threshold] = threshold
            # finetuning: clip zero-values
            # for exactly the same effect as above but in the opposite direction
            # (note: this requires the preprocessing to contain local normalization,
            # otherwise the replacement value is meaningless.)
            X_input[X_input == 0] = 1
            # finetuning: set NaN to zero
            # not needed before, probably because of filtering,
            # but need to explicitly build in that safety here
            # because filtering of input data is not guaranteed.
            X_input = np.nan_to_num(X_input, nan=0)
            # run NMF inference/reconstruction
            mes_reco[mename] = self.nmfs[mename].predict(X_input)
        return mes_reco
    
    def loss(self, X_input, X_reco, do_thresholding=True, verbose=False):
        '''
        Do loss calculation.
        - Input arguments:
        - X_input and X_reco: dictionary of the following form {monitoring element name: data (2D np array), ...}
          with input data (preprocessed) and NMF reconstruction respectively.
        '''
        
        # loop over monitoring elements
        losses = {}
        if verbose: print('[INFO]: calculating losses...')
        for mename in self.menames:
            if verbose: print(f'  - {mename}')
            # calculate losses
            losses[mename] = np.square(X_input[mename] - X_reco[mename])
            # thresholding
            if do_thresholding:
                losses[mename] = clustering.cluster_loss_multithreshold(losses[mename], self.threshold_patterns)
        return losses
    
    def combine(self, X_loss, do_masking=True, do_thresholding=True, verbose=False):
        '''
        Combine losses from different layers.
        - Input arguments:
        - X_loss: dictionary of the following form {monitoring element name: loss (2D np array), ...}
        '''
        
        # overlay different monitoring elements
        if verbose: print('[INFO]: overlaying layers...')
        target_shape = X_loss[self.menames[0]].shape[1:3]
        losses_combined = np.zeros(X_loss[self.menames[0]].shape)
        for mename in self.menames:
            losses_rebinned = rebinning.rebin_keep_clip(X_loss[mename], target_shape, 1, mode='cv2')
            losses_combined += losses_rebinned
    
        # do loss masking
        loss_mask = np.zeros((1, target_shape[0], target_shape[1]))
        if do_masking:
            if self.loss_mask is None:
                msg = '[WARNING]: requested loss masking, but loss mask is None.'
                print(msg)
            else: loss_mask = np.copy(self.loss_mask)
            
        # broadcast into correct shape
        loss_mask = np.repeat(loss_mask, len(losses_combined), axis=0)
  
        # apply threshold on combined binary loss
        if do_thresholding:
            if verbose: print('[INFO]: applying final threshold...')
            losses_combined = ((losses_combined >= 2) & (losses_combined > loss_mask)).astype(int)
            
        # return the result
        return losses_combined
    
    def flag(self, X_loss, verbose=False):
        '''
        Do final flagging of combined loss map.
        - Input arguments:
        - X_loss: combined loss (2D np array)
        '''
        
        # search for patterns in the combined loss
        flags = patternfiltering.contains_any_pattern(X_loss, self.flagging_patterns, threshold = 1e-3)
        return flags
        
    def predict(self, X, verbose=False):
        '''
        Run full chain on incoming data X
        '''
        if verbose:
            print('[INFO]: Running PixelNMF.predict on the following data:')
            for key, val in X.items():
                print(f'  - {key}: {val.shape}')
        mes_preprocessed = self.preprocess(X, verbose=verbose)
        mes_reco = self.infer(mes_preprocessed, verbose=verbose)
        losses = self.loss(mes_preprocessed, mes_reco, do_thresholding=True, verbose=verbose)
        losses_combined = self.combine(losses, do_masking=True, do_thresholding=True, verbose=verbose)
        flags = self.flag(losses_combined, verbose=verbose)
        return flags