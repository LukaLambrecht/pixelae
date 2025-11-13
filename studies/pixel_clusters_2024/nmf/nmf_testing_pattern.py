# imports

# external modules
import os
import sys
import json
import time
import joblib
import numpy as np
import scipy as sp
import pandas as pd
import argparse

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
from studies.pixel_clusters_2024.omstools.omstools import get_oms_data, get_hlt_data
from studies.pixel_clusters_2024.preprocessing.preprocessor import PreProcessor
from studies.pixel_clusters_2024.preprocessing.preprocessor import make_default_preprocessor
from studies.pixel_clusters_2024.nmf.concatenate_outputs import concatenate_output

# optional: dials for on-the-fly data retrieval
try:
    from cmsdials import Dials
    from cmsdials.auth.bearer import Credentials
    from cmsdials.filters import LumisectionFilters
    from cmsdials.filters import LumisectionHistogram2DFilters
    from cmsdials.filters import RunFilters
except:
    msg = 'WARNING: there was an error importing cmsdials;'
    msg += ' any downstream call to cmsdials will likely crash,'
    msg += ' but this should not be an issue when not using dials.'
    print(msg)


def get_dials_creds(max_attempts=5):
    ### helper function to get dials credentials

    authenticated = False
    auth_attempt_counter = 0
    while (auth_attempt_counter<max_attempts and not authenticated):
        auth_attempt_counter += 1
        try:
            creds = Credentials.from_creds_file()
            authenticated = True
        except: continue
    if not authenticated:
        # try one more time to trigger the original error again
        creds = Credentials.from_creds_file()
    sys.stdout.flush()
    sys.stderr.flush()
    return creds

def get_data_from_dials(dials, filters, max_attempts=5, max_pages=None):
    ### helper function to get dials data
  
    # make a wrapped call to cmsdials api
    data_retrieved = False
    attempt_counter = 0
    while (attempt_counter<max_attempts and not data_retrieved):
        attempt_counter += 1
        try:
            data = dials.h2d.list_all(filters, max_pages=max_pages, enable_progress=False)
            data_retrieved = True
        except: continue
    if not data_retrieved:
        # try one more time to trigger the original error
        data = dialsfunc(filters, max_pages=max_pages, enable_progress=False)
    sys.stdout.flush()
    sys.stderr.flush()
    return data

def make_dataloaders(input_file_dict):
    # make a dataloader for each (set of) input file(s).
    # input and output are both 2-layer dicts of the form era -> layer -> input file / dataloader.
    dataloaders = {}
    for era, layers in input_file_dict.items():
        dataloaders[era] = {}
        for layer, files in layers.items():
            # special case where files is actually a set of dials filters
            # to retrieve data on the fly
            if isinstance(files[0], dict): dataloaders[era][layer] = files[0]
            # default case where files is a list of files
            elif isinstance(files[0], str): dataloaders[era][layer] = MEDataLoader(files, verbose=False)
            else: raise Exception('Type not recognized.')
    print('Initiated the following data loaders:')
    for era, layers in dataloaders.items():
        print(f'  - Era {era}:')
        for layer, dataloader in layers.items():
            if isinstance(dataloader, MEDataLoader):
                nfiles = len(dataloader.parquet_files)
                nrows = sum(dataloader.nrows)
                print(f'    - Layer {layer}: data loader with {nfiles} files and {nrows} rows.')
            else:    
                print(f'    - Layer {layer}: [None] (will retrieve data online).')
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
        
def run_evaluation_batch(batch_paramset, dataloaders, nmfs, target_run=None, **kwargs):
    # run evaluation on a single batch.
    
    # initializations
    start_time = time.time()

    # read run numbers
    runs = None
    layers = list(dataloaders.keys())
    run_column = 'run_number'
    if len(batch_paramset)==2: runs = batch_paramset[1]
    elif isinstance(dataloaders[layers[0]], MEDataLoader):
        runs = np.unique(dataloaders[layers[0]].read_batch(batch_paramset, columns=[run_column])[run_column].values)
    else:
        # the remaining case, where the batch is defined in terms of absolute indices,
        # but the data is supposed to be retrieved on the fly, does not make sense.
        msg = 'Requested to retrieve data from DIALS on the fly,'
        msg += ' but the provided batch specification is incompatible with that.'
        raise Exception(msg)
 
    # skip this batch if the requested run is not present in this batch
    if target_run is not None and target_run not in runs: return

    # load the batch
    # (i.e. read the dataframes for this batch from file)
    dfs = {}
    layers = list(dataloaders.keys())
    for layer in layers:
        # default case where data are read from file
        if isinstance(dataloaders[layer], MEDataLoader):
            dfs[layer] = dataloaders[layer].read_batch(batch_paramset)
        # special case: retrieve data from dials on the fly
        else:
            this_dfs = []
            creds = get_dials_creds()
            dials = Dials(creds, workspace='tracker')
            for run in runs:
                dialsfilters = LumisectionHistogram2DFilters(
                  dataset = dataloaders[layer]["dataset"],
                  me = dataloaders[layer]["me"],
                  run_number = run
                )
                data = get_data_from_dials(dials, dialsfilters)
                this_dfs.append( data.to_pandas() )
            dfs[layer] = pd.concat(this_dfs, ignore_index=True)
            if len(dfs[layer])==0:
                msg = 'WARNING: empty dataframe returned...'
                print(msg)
        
    # split evaluation per run
    # note: this makes most sense if the batches are defined as groups of runs rather than fixed-size batches,
    #       else runs might be split in half and be present in multiple batches;
    #       for the most part, it doesn't matter as lumisections are evaluated independently,
    #       but one exception is dynamic masking.
    run_outputs = []
    for run in runs:

        if target_run is not None and run!=target_run: continue

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
                     dynamic_loss_masking_window = None,
                     debug=False):
    
    # initializations
    layers = list(dfs.keys())

    # get OMS info if not provided
    if oms_filters is not None and oms_info is None:
        print('      Retrieving OMS info...')
        attributes = [oms_filter[0] for oms_filter in oms_filters]
        run_numbers = dfs[layers[0]]['run_number'].values
        ls_numbers = dfs[layers[0]]['ls_number'].values
        oms_info = get_oms_data(attributes, run_numbers, ls_numbers).to_dict()
        oms_info = {key: np.array(list(val.values())) for key, val in oms_info.items()}
    if hltrate_filters is not None and hltrate_info is None:
        if len(hltrate_filters)>1: raise Exception('not yet implemented')
        path = hltrate_filters[0][0]
        run_numbers = dfs[layers[0]]['run_number'].values
        ls_numbers = dfs[layers[0]]['ls_number'].values
        hltrate_info = get_hlt_data(path, run_numbers, ls_numbers).to_dict()
        hltrate_info = {key: np.array(list(val.values())) for key, val in hltrate_info.items()}
        hltrate_info_formatted = {}
        for run in np.unique(run_numbers):
            mask = (run_numbers==run).astype(bool)
            this_ls_nbs = ls_numbers[mask]
            this_run_nbs = run_numbers[mask]
            this_rate = hltrate_info['hlt_rate'][mask]
            hltrate_info_formatted[str(run)] = {path: {
              'rate': this_rate,
              'run_number': this_run_nbs,
              'first_lumisection_number': this_ls_nbs
            }}
        hltrate_info = hltrate_info_formatted

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
            losses_thresholded[layer] = clustering.cluster_loss_multithreshold(losses[layer], pattern_thresholds[layer])
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
            # note: the provided loss masks are assumed to have the following values:
            #       1 = bin is good, not masked-away, should be considered.
            #       0 = bin is in permanent loss, masked-away, should be ignored.
            #       but for the downstream process, we want the other way round,
            #       where 0 means the bin is good and 1 means it should be ignored.
            this_loss_mask = 1 - this_loss_mask
            # store at this stage for later use
            preprocessed_loss_masks_per_layer[layer] = this_loss_mask
            if debug: np.save(f'static_loss_{layer}.npy', this_loss_mask)
            # rescale
            this_loss_mask = rebinning.rebin_keep_clip(this_loss_mask, target_shape, 1, mode='cv2')
            if debug: np.save(f'static_loss_rebinned_{layer}.npy', this_loss_mask)
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
            # get the loss
            loss_binary = losses_thresholded[layer]
            # exclude static loss mask for dynamic mask calculation
            # note: need to invert back to the convention with 1 for good bins and 0 for bins to be ignored.
            this_static_loss_mask = 1 - preprocessed_loss_masks_per_layer[layer]
            loss_binary = np.multiply(loss_binary, np.repeat(this_static_loss_mask, len(loss_binary), axis=0))
            # ignore very small regions (can occur after previous step)
            loss_binary = patternfiltering.filter_any_pattern(loss_binary, [np.ones((1,4)), np.ones((4,1)), np.ones((2,2))])
            # calculate dynamic loss mask for this layer
            this_dynamic_loss_mask = sp.ndimage.convolve1d(loss_binary.astype(float), window, axis=0, mode='constant')
            if debug: np.save(f'dynamic_loss_{layer}.npy', this_dynamic_loss_mask)
            this_dynamic_loss_mask = (this_dynamic_loss_mask >= 0.999).astype(int)
            # rescale and add to total
            this_dynamic_loss_mask = rebinning.rebin_keep_clip(this_dynamic_loss_mask, target_shape, 1, mode='cv2')
            if debug: np.save(f'dynamic_loss_rebinned_{layer}.npy', this_dynamic_loss_mask)
            dynamic_loss_mask += this_dynamic_loss_mask

    # store some extra intermediate output for debugging
    if debug:
        np.save('run_number.npy', dfs[layers[0]]['run_number'].values)
        np.save('ls_number.npy', dfs[layers[0]]['ls_number'].values)
        np.save('losses_combined.npy', losses_combined)
        np.save('static_loss_combined', loss_mask)
        np.save('dynamic_loss_combined', dynamic_loss_mask)

    # default criterion: at least 2 layers.
    # apply threshold on combined binary loss
    losses_combined_binary = ((losses_combined >= 2) & (losses_combined > (loss_mask + dynamic_loss_mask))).astype(int)
    # search for patterns in the combined loss
    print('      Searching for patterns in the loss map...')
    flags = patternfiltering.contains_any_pattern(losses_combined_binary, flagging_patterns,
              threshold = flagging_threshold)
    
    # alternative criterion: single layer (with larger size).
    # note: hard-coded for now, maybe extend later.
    # note: disabled, since it's probably better to only allow this for layer 2.
    # apply threshold on combined binary loss
    #losses_combined_binary_sl = ((losses_combined >= 1) & (losses_combined > (loss_mask + dynamic_loss_mask))).astype(int)
    # search for patterns in the combined loss
    #flagging_patterns_sl = [np.ones((2, 32))]
    #flagging_threshold_sl = 1e-3
    #flags_sl = patternfiltering.contains_any_pattern(losses_combined_binary_sl, flagging_patterns_sl,
    #             threshold = flagging_threshold_sl)
    
    # alternative criterion: single layer, but only layer 2 (with larger size).
    # note: hard-coded for now, maybe extend later.
    #layer = 'BPix2'
    #losses_binary_l2 = ((losses_thresholded[layer] == 1) & (preprocessed_loss_masks_per_layer[layer]==0))
    #flagging_patterns_l2 = [np.ones((2, 32))]
    #flagging_threshold_l2 = 1e-3
    #flags_l2 = patternfiltering.contains_any_pattern(losses_binary_l2, flagging_patterns_l2,
    #             threshold = flagging_threshold_l2)
   
    # combine criteria
    #flags = ((flags) | (flags_l2))
    
    # store the flagged lumisections
    flagged_run_numbers = dfs[layers[0]]['run_number'].values[flags]
    flagged_ls_numbers = dfs[layers[0]]['ls_number'].values[flags]
    n_unique_runs = len(np.unique(dfs[layers[0]]['run_number'].values[flags]))
    print(f'      Found {np.sum(flags.astype(int))} flagged lumisections in {n_unique_runs} runs.')
    if debug: print(flagged_ls_numbers)
 
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


def evaluate(config, target_run=None, debug=False):
        
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
            oms_info_file = config['oms_filter_files'][era]
            if oms_info_file is None:
                oms_info[era] = None # retrieve on the fly downstream
            else:
                with open(config['oms_filter_files'][era], 'r') as f:
                    oms_info[era] = json.load(f)
    hltrate_info = None
    hltrate_filters = None
    if 'hltrate_filters' in config.keys():
        hltrate_filters = config['hltrate_filters']
        hltrate_info = {}
        for era in eras:
            hltrate_info_file = config['hltrate_filter_files'][era]
            if hltrate_info_file is None:
                hltrate_info[era] = None # retrieve on the fly downstream
            else:
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
        batch_params = None
        dataloader = dataloaders[era][layers[0]]
        if isinstance(dataloader, MEDataLoader):
            kwargs = {}
            if batch_size == 'run': kwargs['target_size'] = target_batch_size
            batch_params = dataloader.prepare_sequential_batches(batch_size=batch_size, **kwargs)
            print('Found following batch parameters (with file-based input):')
            print(batch_params)
        else:
            # read available runs for the provided dataset from DIALS
            creds = get_dials_creds()
            dials = Dials(creds, workspace='tracker')
            runfilters = RunFilters(dataset=dataloader["dataset"])
            runs = dials.run.list_all(runfilters, enable_progress=False).results
            runs = sorted([el.run_number for el in runs])
            print('Found following runs (with dials-based input):')
            print(runs)
            # select runs based on partitioning
            if 'part' in dataloader.keys() and 'nparts' in dataloader.keys():
                part = dataloader['part']
                nparts = dataloader['nparts']
                runs = list(np.array_split(np.array(runs), nparts)[part])
                print(f'Limiting runs to following partition (part {part}):')
                print(runs)
            batch_params = [(0, [run]) for run in runs]
            print('Found following batch parameters (with dials-based input):')
            print(batch_params)
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
                                                 dynamic_loss_masking_window = dynamic_loss_masking_window,
                                                 target_run=target_run, debug=debug)
            batch_outputs.append(batch_output)

    # contatenate the result
    res = concatenate_output(batch_outputs)
    
    return res
        
        
if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True) 
    parser.add_argument('--run', default=None)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--no-output', default=False, action='store_true')
    args = parser.parse_args()

    # parse arguments
    target_run = None if args.run is None else int(args.run)

    # read job config
    configfile = args.config
    print(f'Reading job config "{configfile}"...')
    with open(configfile, 'r') as f:
        config = json.load(f)
        
    # do evaluation
    output = evaluate(config, target_run=target_run, debug=args.debug)
    
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
    if not args.no_output:
        outputfile = config['outputfile']
        outputdir = os.path.dirname(outputfile)
        if not os.path.exists(outputdir): os.makedirs(outputdir)
        with open(outputfile, 'w') as f:
            json.dump(output, f)
        print(f'Output file "{outputfile}" written.')
        print('Done.')
