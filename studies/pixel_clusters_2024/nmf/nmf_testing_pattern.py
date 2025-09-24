# imports

# external modules
import os
import sys
import json
import time
import joblib
import importlib
import numpy as np
import scipy as sp
import pandas as pd

# local modules
thisdir = os.getcwd()
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)
from tools.dataloadertools import MEDataLoader
import tools.dftools as dftools
import tools.patternfiltering as patternfiltering
import tools.rebinning as rebinning
import tools.clustering as clustering
from automasking.tools.automaskreader import AutomaskReader
from studies.pixel_clusters_2024.preprocessing.preprocessor import PreProcessor
from studies.pixel_clusters_2024.preprocessing.preprocessor import make_default_preprocessor


def make_dataloaders(input_file_dict):
    # make a dataloader for each (set of) input file(s).
    # input and output are both 2-layer dicts of the form era -> layer -> input file / dataloader.
    dataloaders = {}
    for era, layers in input_file_dict.items():
        dataloaders[era] = {}
        for layer, files in layers.items(): dataloaders[era][layer] = MEDataLoader(files, verbose=False)
    print('Initiated the following data loaders:')
    for era, layers in dataloaders.items():
        print(f'  - Era {era}:')
        for layer, dataloader in layers.items():
            nfiles = len(dataloader.parquet_files)
            nrows = sum(dataloader.nrows)
            print(f'    - Layer {layer}: data loader with {nfiles} files and {nrows} rows.')
    return dataloaders
        
def make_preprocessors(eras, layers, local_normalization=None, **kwargs):
    # make a preprocessor for all eras and layers.
    # input are lists of eras and layers.
    # output is a 2-layer dict of the form era -> layer -> preprocessor.
    preprocessors = {}
    for era in eras:
        preprocessors[era] = {}
        preprocessor_era = era
        if '-part' in era: preprocessor_era = era.split('-part')[0]
        this_local_normalization = None
        if local_normalization is not None: this_local_normalization = local_normalization[era]
        for layer in layers: preprocessors[era][layer] = make_default_preprocessor(preprocessor_era, layer,
                                                           local_normalization=this_local_normalization,
                                                           **kwargs)
    return preprocessors    

def load_nmfs(nmf_file_dict):
    # load and NMF model from each input file.
    # input and output are both 2-layer dicts of the form era -> layer -> stored nmf model file / loaded nmf model.
    nmfs = {}
    for era, layers in nmf_file_dict.items():
        nmfs[era] = {}
        for layer, modelfile in layers.items(): nmfs[era][layer] = joblib.load(modelfile)
    return nmfs

def concatenate_output(outputs):
    # concatenate the outputs of run_evaluation
    
    # make lists
    batch_filter_results = []
    batch_flagged_run_numbers = []
    batch_flagged_ls_numbers = []
    for output in outputs:
        if output is None: continue
        batch_filter_results.append(output['filter_results'])
        if len(output['flagged_run_numbers'])>0:
            batch_flagged_run_numbers.append(output['flagged_run_numbers'])
            batch_flagged_ls_numbers.append(output['flagged_ls_numbers'])

    # contatenate the lists
    filter_results = {}
    if len(batch_filter_results)>0:
        for key in batch_filter_results[0].keys():
            filter_results[key] = sum([batch_filter_result[key] for batch_filter_result in batch_filter_results], [])
    if len(batch_flagged_run_numbers) > 0:
        flagged_run_numbers = np.concatenate(batch_flagged_run_numbers)
        flagged_ls_numbers = np.concatenate(batch_flagged_ls_numbers)
    else:
        flagged_run_numbers = np.array([])
        flagged_ls_numbers = np.array([])
    
    # make final result
    res = {
        'flagged_run_numbers': flagged_run_numbers,
        'flagged_ls_numbers': flagged_ls_numbers,
        'filter_results': filter_results
    }
    return res
        
def run_evaluation_batch(batch_paramset, dataloaders, nmfs, **kwargs):
    # run evaluation on a single batch.
    
    # initializations
    start_time = time.time()
    
    # load the batch
    # (i.e. read the dataframes for this batch from file)
    dfs = {}
    layers = list(dataloaders.keys())
    for layer in layers:
        dfs[layer] = dataloaders[layer].read_batch(batch_paramset)
        
    # determine runs in this batch
    # note: this is not strictly needed, but is used to split evaluation per run.
    # note: alternatively, one could define and read the batches as individual runs,
    #       but that is much slower.
    # note: this makes most sense if the batches are defined as groups of runs rather than fixed-size batches,
    #       else runs might be split in half and be present in multiple batches;
    #       for the most part, it doesn't matter as lumisections are evaluated independently,
    #       but one exception is dynamic masking.
    run_column = 'run_number'
    runs = np.unique(dfs[layers[0]][run_column].values)
    
    # loop over runs
    run_outputs = []
    for run in runs:
        print(f'    Now running on run {run}...')
        this_dfs = {}
        for layer in layers:
            this_dfs[layer] = dftools.select_runs(dfs[layer], [run], runcolumn=run_column)
    
        # run the evaluation on this data
        output = run_evaluation(this_dfs, nmfs, **kwargs)
        run_outputs.append(output)
        
    # concatenate outputs
    output = concatenate_output(run_outputs)
    
    # calculate time spent for this batch
    end_time = time.time()
    print('    Time for this batch: {:.2f}s'.format(end_time-start_time))
    
    # return the result
    return output

def run_evaluation(dfs, nmfs,
                     preprocessors = None,
                     min_entries_filter = None,
                     oms_info = None,
                     oms_filters = None,
                     hltrate_info = None,
                     hltrate_filters = None,
                     flagging_patterns = None,
                     flagging_threshold = None,
                     pattern_thresholds = None,
                     do_automasking = False,
                     automask_reader = None,
                     automask_map_preprocessors = None,
                     do_loss_masking = False,
                     loss_masks = None,
                     loss_mask_preprocessors = None,
                     do_dynamic_loss_masking = False,
                     dynamic_loss_masking_window = None):
    
    # initializations
    layers = list(dfs.keys())
    
    # filtering
    ndf = len(dfs[layers[0]])
    mask, filter_results = dftools.filter_dfs(dfs,
                             min_entries_filter = min_entries_filter,
                             oms_info = oms_info,
                             oms_filters = oms_filters,
                             hltrate_info = hltrate_info,
                             hltrate_filters = hltrate_filters)
    for layer in layers:
        dfs[layer] = dfs[layer][mask]
    ndfnew = len(dfs[layers[0]])    
    print(f'      Found {ndfnew} / {ndf} instances passing filters.')
    
    # safety for 0 instances passing filters
    if ndfnew==0:
        res = {
            'flagged_run_numbers': [],
            'flagged_ls_numbers': [],
            'filter_results': filter_results
        }
        return res
        
    # do preprocessing
    mes_preprocessed = {}
    if preprocessors is not None:
        print('      Preprocessing...')
        for layer in layers: mes_preprocessed[layer] = preprocessors[layer].preprocess(dfs[layer])
    else:
        for layer in layers: mes_preprocessed[layer], _, _ = dftools.get_mes(df,
                                       xbinscolumn='x_bin', ybinscolumn='y_bin',
                                       runcolumn='run_number', lumicolumn='ls_number')
    
    # do evaluation
    print('      Evaluating...')
    losses = {}
    for layer in layers:
        # make a copy for some additional processing before inference
        # that should not be reflected in the original to compare with
        this_mes_preprocessed = np.copy(mes_preprocessed[layer])
        # clip very large values
        # to avoid the NMF model from compromising good agreement in most bins
        # for the sake of fitting slightly better a few spikes.
        # (but only when preprocessing is applied, otherwise 'very large' is more difficult to define).
        if preprocessors is not None:
            threshold = 5
            this_mes_preprocessed[this_mes_preprocessed > threshold] = threshold
        # clip zero-values
        # for exactly the same effect as above but in the opposite direction
        if preprocessors is not None:
            this_mes_preprocessed[this_mes_preprocessed == 0] = 1
        mes_pred = nmfs[layer].predict(this_mes_preprocessed)
        losses[layer] = np.square(mes_preprocessed[layer] - mes_pred)
    
    # optional: do automasking
    if do_automasking:
        print('      Applying automask...')
        runs = dfs[layer[0]]['run_number'].values
        lumis = dfs[layer[0]]['ls_number'].values
        for layer in layers:
            subsystem = f'BPix{layer}'
            automask_maps = amreader.get_automask_maps_for_ls(runs, lumis, subsystem, invert=True)
            automask_maps = automask_map_preprocessors[layer].preprocess_mes(automask_maps, None, None)
            losses[layer] *= automask_maps
        
    # thresholding and clustering
    losses_thresholded = {}
    print('      Thresholding and clustering...')
    if pattern_thresholds is not None:
        for layer in layers:
            losses_thresholded[layer] = clustering.cluster_loss_multithreshold(losses[layer], pattern_thresholds)
    else: raise Exception('Default behaviour for this case not yet implemented.')

    # overlay different layers
    # strategy: sum the (rebinned) binary loss maps over different layers.
    print('      Combining layers...')
    target_shape = losses[layers[0]].shape[1:3]
    losses_combined = np.zeros(losses[layers[0]].shape)
    for layer in layers:
        losses_rebinned = rebinning.rebin_keep_clip(losses_thresholded[layer], target_shape, 1, mode='cv2')
        losses_combined += losses_rebinned

    # optional: do loss masking
    loss_mask = np.zeros(losses_combined.shape)
    preprocessed_loss_masks_per_layer = {}
    if do_loss_masking:
        print('      Applying loss mask...')
        loss_mask = np.zeros((1, target_shape[0], target_shape[1]))
        for layer in layers:
            this_loss_mask = loss_masks[layer]
            # preprocess
            this_loss_mask = np.expand_dims(this_loss_mask, 0)
            this_loss_mask = loss_mask_preprocessors[layer].preprocess_mes(this_loss_mask, None, None)
            # invert
            this_loss_mask = 1 - this_loss_mask
            # store at this stage for later use
            preprocessed_loss_masks_per_layer[layer] = this_loss_mask
            # rescale
            this_loss_mask = rebinning.rebin_keep_clip(this_loss_mask, target_shape, 1, mode='cv2')
            # add to total
            loss_mask += this_loss_mask
        loss_mask = np.repeat(loss_mask, len(losses_combined), axis=0)

    # optional: do dynamic loss masking
    dynamic_loss_mask = np.zeros(losses_combined.shape)
    if do_dynamic_loss_masking:
        print('      Calculating and applying dynamic loss mask...')
        window = np.concatenate((np.zeros(dynamic_loss_masking_window+1),
                                 np.ones(dynamic_loss_masking_window))).astype(float)
        window /= np.sum(window)
        for layer in layers:
            # re-calculate binary threshold on loss
            # note: cannot use the thresholded loss calculated above since it is implicitly clustered,
            #       while here we need bin-per-bin values.
            # note: for the thresholding, an arbitrary choice has to be made
            #       (since multiple thresholds with corresponding clustering patterns can be provided);
            #       here we just pick the first element in the provided list.
            threshold = pattern_thresholds[0]['loss_threshold']
            loss_binary = (losses[layer] > threshold).astype(int)
            # exclude static loss mask for dynamic mask calculation
            this_static_loss_mask = 1 - preprocessed_loss_masks_per_layer[layer]
            loss_binary = np.multiply(loss_binary, np.repeat(this_static_loss_mask, len(loss_binary), axis=0))
            # calculate dynamic loss mask for this layer
            this_dynamic_loss_mask = sp.ndimage.convolve1d(loss_binary.astype(float), window, axis=0, mode='constant')
            this_dynamic_loss_mask = (this_dynamic_loss_mask >= 0.999).astype(int)
            # rescale and add to total
            this_dynamic_loss_mask = rebinning.rebin_keep_clip(this_dynamic_loss_mask, target_shape, 1, mode='cv2')
            dynamic_loss_mask += this_dynamic_loss_mask

    # apply threshold on combined binary loss
    losses_combined = ((losses_combined >= 2) & (losses_combined > (loss_mask + dynamic_loss_mask))).astype(int)
    
    # search for patterns in the combined loss
    print('      Searching for patterns in the loss map...')
    flags = patternfiltering.contains_any_pattern(losses_combined, flagging_patterns,
              threshold = flagging_threshold)
    
    # store the flagged lumisections
    flagged_run_numbers = dfs[layers[0]]['run_number'].values[flags]
    flagged_ls_numbers = dfs[layers[0]]['ls_number'].values[flags]
    n_unique_runs = len(np.unique(dfs[layers[0]]['run_number'].values[flags]))
    print(f'      Found {np.sum(flags.astype(int))} flagged lumisections in {n_unique_runs} runs.')
    
    # explicitly delete some variables for memory management
    del dfs
    del mes_preprocessed
    del losses

    # return the result
    res = {
        'flagged_run_numbers': flagged_run_numbers,
        'flagged_ls_numbers': flagged_ls_numbers,
        'filter_results': filter_results
    }
    return res


def evaluate(config):
        
    # get eras and layers
    eras = config['eras']
    layers = config['layers']
        
    # make dataloaders, preprocessors and models
    dataloaders = make_dataloaders(config['input_files'])
    global_normalization = config.get('preprocessing_global_normalization', None)
    local_normalization = config.get('preprocessing_local_normalization', None)
    preprocessors = make_preprocessors(eras, layers,
                      global_normalization = global_normalization,
                      local_normalization = local_normalization)
    nmfs = load_nmfs(config['nmf_files'])
    
    # get evaluation settings
    batch_size = config['batch_size']
    target_batch_size = config['target_batch_size']
    flagging_patterns = [np.array(el) for el in config['flagging_patterns']]
    flagging_threshold = config['flagging_threshold']
    pattern_thresholds = config['pattern_thresholds']
    do_automasking = config['do_automasking']
    do_loss_masking = config['do_loss_masking']
    do_dynamic_loss_masking = config['do_dynamic_loss_masking']

    # read filters if needed
    min_entries_filter = config['min_entries_filter'] if 'min_entries_filter' in config.keys() else None
    oms_info = None
    oms_filters = None
    if 'oms_filters' in config.keys():
        oms_filters = config['oms_filters']
        oms_info = {}
        for era in eras:
            with open(config['oms_filter_files'][era], 'r') as f:
                oms_info[era] = json.load(f)
    hltrate_info = None
    hltrate_filters = None
    if 'hltrate_filters' in config.keys():
        hltrate_filters = config['hltrate_filters']
        hltrate_info = {}
        for era in eras:
            with open(config['hltrate_filter_files'][era], 'r') as f:
                hltrate_info[era] = json.load(f)

    # make automask reader if needed
    automask_reader = None
    automask_map_preprocessors = None
    if do_automasking:
        automask_reader = AutomaskReader(config['automask_data_file'])
        automask_map_preprocessors = {}
        for layer in layers: automask_map_preprocessors[layer] = PreProcessor(f'PXLayer_{layer}')

    # make loss mask if needed
    loss_masks = None
    loss_mask_preprocessors = None
    if do_loss_masking:
        loss_masks = {}
        loss_mask_preprocessors = {}
        for era in eras:
            loss_masks[era] = {}
            for layer in layers:
                loss_mask_file = config['loss_masking_zero_frac_files'][era][layer]
                loss_mask = np.load(loss_mask_file)
                loss_mask = (loss_mask < config['loss_masking_zero_frac_threshold'])
                loss_masks[era][layer] = loss_mask
        for layer in layers: loss_mask_preprocessors[layer] = PreProcessor(f'PXLayer_{layer}')
            
    # read dynamic loss masking settings if needed
    dynamic_loss_masking_window = None
    if do_dynamic_loss_masking:
        key = 'dynamic_loss_masking_window'
        if key in config.keys():
            dynamic_loss_masking_window = int(config[key])
        else:
            dynamic_loss_masking_window = 10
            msg = 'WARNING: dynamic loss masking was requested,'
            msg += f' but no window is set; using default of {dynamic_loss_masking_window}.'
            print(msg)

    # initialize result
    batch_outputs = []
    
    # loop over eras
    for era in eras:
        print(f'Now running on era {era}...')
        
        # prepare batch parameters
        # (common to all layers)
        kwargs = {}
        if batch_size == 'run': kwargs['target_size'] = target_batch_size
        batch_params = dataloaders[era][layers[0]].prepare_sequential_batches(batch_size=batch_size, **kwargs)
        nbatches = len(batch_params)
        print(f'Prepared {nbatches} batches of size {batch_size}.')

        # run over batches
        for batchidx, batch_paramset in enumerate(batch_params):
            print(f'  Now running on batch {batchidx+1} / {nbatches}...')
            batch_output = run_evaluation_batch(batch_paramset, dataloaders[era], nmfs[era],
                                                 preprocessors = preprocessors[era],
                                                 min_entries_filter = min_entries_filter,
                                                 oms_info = oms_info[era],
                                                 oms_filters = oms_filters,
                                                 hltrate_info = hltrate_info[era],
                                                 hltrate_filters = hltrate_filters,
                                                 flagging_patterns = flagging_patterns,
                                                 flagging_threshold = flagging_threshold,
                                                 pattern_thresholds = pattern_thresholds,
                                                 do_automasking = do_automasking,
                                                 automask_reader = automask_reader,
                                                 automask_map_preprocessors = automask_map_preprocessors,
                                                 do_loss_masking = do_loss_masking,
                                                 loss_masks = None if loss_masks is None else loss_masks[era],
                                                 loss_mask_preprocessors = loss_mask_preprocessors,
                                                 do_dynamic_loss_masking = do_dynamic_loss_masking,
                                                 dynamic_loss_masking_window = dynamic_loss_masking_window)
            batch_outputs.append(batch_output)

    # contatenate the result
    res = concatenate_output(batch_outputs)
    
    return res
        
        
if __name__=='__main__':
    
    # read job config
    configfile = sys.argv[1]
    print(f'Reading job config "{configfile}"...')
    with open(configfile, 'r') as f:
        config = json.load(f)
        
    # do evaluation
    output = evaluate(config)
    
    # parsing for json compatibility
    output['flagged_run_numbers'] = output['flagged_run_numbers'].tolist() 
    output['flagged_ls_numbers'] = output['flagged_ls_numbers'].tolist()
    for key, lslist in output['filter_results'].items():
        for idx, (run, lumi) in enumerate(lslist):
            lslist[idx] = (int(run), int(lumi))
            
    # printouts for checking and debugging
    nflags = len(output['flagged_run_numbers'])
    print(f'Summarized results from running over config {configfile}:')
    print(f'  - Number of flagged lumisections: {nflags}.')

    # write output file
    outputfile = config['outputfile']
    outputdir = os.path.dirname(outputfile)
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    with open(outputfile, 'w') as f:
        json.dump(output, f)
    print(f'Output file "{outputfile}" written.')
    print('Done.')
