# Model definition

# note: inside the container, only the files in the mlserver-model directory are visible.
#       importing local modules from outside this directory is not trivial
#       (although allegedly possible by making everything a package, but I did not test that yet),
#       and the simplest (though far from optimal) solution might be to duplicate all local dependencies
#       in this local model definition...


# import external modules
import os
import sys
import numpy as np

# import local modules
import dftools as dftools
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
        - nmfs: dictionary of the following form: {monitoring element name: NMF model, ...}
        - local_norms: dictionary of the following form: {monitoring element name: local norm (2D np array), ...}
        - loss_masks: dictionary of the following form: {monitoring element name: loss mask (2D np array), ...}
        '''
        
        # get monitoring element names for later use
        self.menames = list(nmfs.keys())[:]
        
        # check NMF models
        # note: one model per monitoring element
        self.nmfs = {}
        for mename, nmf in nmfs.items():
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
            
    def check_keys(self, X, verbose=False):
        '''
        Internal helper function to check keys of provided input data.
        '''
        if sorted(list(X.keys())) != sorted(self.menames):
            msg = 'ERROR: keys of provided data and self.menames do not agree;'
            msg += f' found {X.keys()} and {self.menames} respectively.'
            if verbose: print(msg)
            raise Exception(msg)
            
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
        #self.check_keys(X, verbose=verbose)
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
        #self.check_keys(X, verbose=verbose)
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
        Input arguments:
        - X_input and X_reco: dictionary of the following form {monitoring element name: data (2D np array), ...}
          with input data (preprocessed) and NMF reconstruction respectively.
        '''
        
        # loop over monitoring elements
        losses = {}
        if verbose: print('[INFO]: calculating losses...')
        #self.check_keys(X_input, verbose=verbose)
        #self.check_keys(X_reco, verbose=verbose)
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
        Input arguments:
        - X_loss: dictionary of the following form {monitoring element name: loss (2D np array), ...}
        '''
        
        # overlay different monitoring elements
        if verbose: print('[INFO]: overlaying layers...')
        #self.check_keys(X_loss, verbose=verbose)
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
        Input arguments:
        - X_loss: combined loss (2D np array)
        '''
        
        # search for patterns in the combined loss
        flags = patternfiltering.contains_any_pattern(X_loss, self.flagging_patterns, threshold = 1e-3)
        return flags
    
    def get_filter_mask(self, X_input, oms_data=None, verbose=False):
        '''
        Apply filters.
        Note: HLT rate filter not yet implemented.
        Note: filters hard-coded in this function for now, maybe generalize later.
        Input arguments:
        - X_input: dictionary of the following form {monitoring element name: data (3D np array), ...} with input data.
        - oms_data: dictionary of the following form {OMS attribute name: data (1D np array), ...} with OMS info.
                    note: for now, the lumisections in oms_data are assumed to exactly match those in X_input,
                          maybe generalize and make more robust later.
        '''
        
        # initializations
        if verbose: print('[INFO]: applying filter...')
        menames = sorted(list(X_input.keys()))
        mask = np.ones(len(X_input[menames[0]])).astype(bool)
        
        # define minimum entries filter
        min_entries_filter = {
            #'BPix1': 0.5e6,
            'BPix2': 0.5e6/2,
            'BPix3': 0.5e6/3,
            'BPix4': 0.5e6/4
        }

        # get number of entries per monitoring element
        entries = {}
        for mename in menames:
            entries[mename] = np.sum(X_input[mename], axis=(1,2))

        # define run numbers and lumisection numbers
        run_numbers = np.zeros(len(mask))
        ls_numbers = np.zeros(len(mask))
        if oms_data is not None:
            run_numbers = oms_data['run_number']
            ls_numbers = oms_data['lumisection_number']

        # define OMS filters
        oms_filters = None
        if oms_data is not None:
            oms_filters = [
              ["beams_stable"],
              ["cms_active"],
              ["bpix_ready"],
              ["fpix_ready"],
              ["tibtid_ready"],
              ["tob_ready"],
              ["tecp_ready"],
              ["tecm_ready"],
              ["pileup", '>', 25],
              ["hlt_zerobias_rate", '>', 5]
            ]
            for oms_filter in oms_filters:
                key = oms_filter[0]
                if key not in oms_data.keys():
                    msg = f'Provided OMS data does not contain the expected key {key}'
                    raise Exception(msg)

        # apply the filter
        mask, _ = dftools.filter_lumisections(run_numbers, ls_numbers,
                    entries = entries, min_entries_filter = min_entries_filter,
                    oms_info = oms_data, oms_filters = oms_filters,
                  )

        # total mask printouts
        if verbose:
            ntot = len(mask)
            npass = np.sum(mask.astype(int))
            print(f'[INFO]: total mask: {npass} / {ntot} passing lumisections.')
            
        # return mask
        return mask
        
    def predict(self, X, verbose=False):
        '''
        Run full chain on incoming data X
        '''

        # printouts for testing and debugging
        print('[INFO]: Running PixelNMF.predict on the following data:')
        for key, val in X.items():
            print(f'  - {key}: {val.shape}')

        # split actual input data from OMS data
        menames = ['BPix1', 'BPix2', 'BPix3', 'BPix4']
        X_input = {key: val for key, val in X.items() if key in menames}
        oms_input = {key: val for key, val in X.items() if key not in menames}
        # format the names of the meta-data fields
        oms_input_formatted = {}
        for key, val in oms_input.items():
            if key=='hlt_zerobias__rate': oms_input_formatted['hlt_zerobias_rate'] = val
            else: oms_input_formatted[key.split('__')[1]] = val
        oms_input = oms_input_formatted
        # printouts for testing and debugging
        print('Found following OMS keys for filtering:')
        print(oms_input.keys())
        if len(oms_input.keys())==0: oms_input = None

        # do preprocessing, inference, and loss calculation
        mes_preprocessed = self.preprocess(X_input, verbose=verbose)
        mes_reco = self.infer(mes_preprocessed, verbose=verbose)
        losses = self.loss(mes_preprocessed, mes_reco, do_thresholding=True, verbose=verbose)
        losses_combined = self.combine(losses, do_masking=True, do_thresholding=True, verbose=verbose)
        flags = self.flag(losses_combined, verbose=verbose)

        # printouts for testing and debugging
        nflags = np.sum(flags.astype(int))
        msg = '[INFO]: Flagging results (before filtering):'
        msg += f' flagged {nflags} out of {len(flags)} lumisections.'
        print(msg)
        
        # apply mask
        mask = self.get_filter_mask(X_input, oms_data=oms_input, verbose=verbose)
        flags[~mask] = False

        # printouts for testing and debugging
        nselected = np.sum(mask.astype(int))
        msg = f'[INFO]: Filtering: selected {nselected} out of {len(flags)} lumisections.'
        print(msg)
        nflags = np.sum(flags.astype(int))
        msg = '[INFO]: Flagging results (after filtering):'
        msg += f' flagged {nflags} out of {len(mask)} lumisections.'
        print(msg)
        sys.stdout.flush()
        sys.stderr.flush()
        
        # return final result
        return flags
