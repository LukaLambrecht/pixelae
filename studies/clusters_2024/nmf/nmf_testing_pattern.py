# imports

# external modules
import os
import sys
import json
import time
import joblib
import importlib
import numpy as np
import pandas as pd

# local modules
thisdir = os.getcwd()
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)
from tools.dataloadertools import MEDataLoader
from tools.omstools import find_oms_attr_for_lumisections
import tools.patternfiltering as patternfiltering
import tools.rebinning as rebinning
from automasking.tools.automaskreader import AutomaskReader
from studies.clusters_2024.preprocessing.preprocessor import PreProcessor
from studies.clusters_2024.preprocessing.preprocessor import make_default_preprocessor


def make_dataloaders(input_file_dict):
    # make a dataloader for each (set of) input file(s).
    # input and output are both 2-layer dicts of the form era -> layer -> input file / dataloader.
    dataloaders = {}
    for era, layers in input_file_dict.items():
        dataloaders[era] = {}
        for layer, files in layers.items(): dataloaders[era][layer] = MEDataLoader(files)
    return dataloaders
        
def make_preprocessors(eras, layers, **kwargs):
    # make a preprocessor for all eras and layers.
    # input are lists of eras and layers.
    # output is a 2-layer dict of the form era -> layer -> preprocessor.
    preprocessors = {}
    for era in eras:
        preprocessors[era] = {}
        preprocessor_era = era
        if '-part' in era: preprocessor_era = era.split('-part')[0]
        for layer in layers: preprocessors[era][layer] = make_default_preprocessor(preprocessor_era, layer, **kwargs)
    return preprocessors    

def load_nmfs(nmf_file_dict):
    # load and NMF model from each input file.
    # input and output are both 2-layer dicts of the form era -> layer -> stored nmf model file / loaded nmf model.
    nmfs = {}
    for era, layers in nmf_file_dict.items():
        nmfs[era] = {}
        for layer, modelfile in layers.items(): nmfs[era][layer] = joblib.load(modelfile)
    return nmfs
        
def run_evaluation_batch(batch_paramset, dataloaders, nmfs, **kwargs):
    # run evaluation on a single batch.
    # helper function to evaluate(), see below.
    
    # initializations
    start_time = time.time()
    
    # get the dataframes
    dfs = {}
    layers = list(dataloaders.keys())
    for layer in layers:
        dfs[layer] = dataloaders[layer].read_sequential_batch(batch_paramset)
    
    # run the evaluation on this data
    output = run_evaluation(dfs, nmfs, **kwargs)
    
    # calculate time spent for this batch
    end_time = time.time()
    print('    Time for this batch: {:.2f}s'.format(end_time-start_time))
    
    # return the result
    return output

def filter_dfs(dfs,
               min_entries_filter = None,
               oms_filters = None):
    '''
    Filter a set of dataframes.
    Input arguments:
    - dfs: dict of dataframes of the form layer -> dataframe
    - min_entries_filter: dict of the form layer -> miminum number of entries per LS
      (requires each dataframe to have a column "entries" in it).
    - oms_filters: dict in OMS format, of the following form:
      {"run_number": [<run numbers>], "lumisection_number": [<lumisection numbers>],
       "<filter name 1>": [<booleans>], "<filter name 2>": [<booleans>], ...}
    '''
    
    # initializations
    filter_results = {}
    layers = list(dfs.keys())
    run_numbers = dfs[layers[0]]['run_number'].values
    ls_numbers = dfs[layers[0]]['ls_number'].values
    combined_mask = np.ones(len(dfs[layers[0]])).astype(bool)
    
    # minimum number of entries filter
    if min_entries_filter is not None:
        for layer in layers:
            threshold = min_entries_filter[layer]
            mask = (dfs[layer]['entries'] > threshold)
            # add to the total mask
            combined_mask = ((combined_mask) & (mask))
            # keep track of lumisections that fail
            fail = [(run, ls) for run, ls in zip(run_numbers[~mask], ls_numbers[~mask])]
            filter_results[f'min_entries_{layer}'] = fail
            
    # OMS attribute filters
    if oms_filters is not None:
        oms_filter_keys = [key for key in oms_filters.keys() if (key!='run_number' and key!='lumisection_number')]
        for key in oms_filter_keys:
            mask = find_oms_attr_for_lumisections(run_numbers, ls_numbers, oms_filters, key)
            # add to the total mask
            combined_mask = ((combined_mask) & (mask))
            # keep track of lumisections that fail
            fail = [(run, ls) for run, ls in zip(run_numbers[~mask], ls_numbers[~mask])]
            filter_results[key] = fail
    
    # return results
    return (combined_mask, filter_results)
    
def run_evaluation(dfs, nmfs,
                     preprocessors = None,
                     min_entries_filter = None,
                     oms_filters = None,
                     loss_threshold = 0.1,
                     flagging_patterns = None,
                     flagging_threshold = None,
                     do_per_layer_cleaning = False,
                     cleaning_patterns = None,
                     cleaning_threshold = None,
                     do_automasking = False,
                     automask_reader = None,
                     automask_map_preprocessors = None,
                     do_loss_masking = False,
                     loss_masks = None,
                     loss_mask_preprocessors = None):
    
    # initializations
    layers = list(dfs.keys())
    
    # filtering
    ndf = len(dfs[layers[0]])
    mask, filter_results = filter_dfs(dfs,
                             min_entries_filter=min_entries_filter,
                             oms_filters=oms_filters)
    for layer in layers:
        dfs[layer] = dfs[layer][mask]
    ndfnew = len(dfs[layers[0]])    
    print(f'    Found {ndfnew} / {ndf} instances passing filters.')
    
    # only for testing: print filter results
    # (decide how to handle more systematically later)
    #print(filter_results)
    
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
        print('    Preprocessing...')
        for layer in layers: mes_preprocessed[layer] = preprocessors[layer].preprocess(dfs[layer])
    else:
        for layer in layers: mes_preprocessed[layer], _, _ = dftools.get_mes(df,
                                       xbinscolumn='x_bin', ybinscolumn='y_bin',
                                       runcolumn='run_number', lumicolumn='ls_number')
    
    # do evaluation and apply threhold to loss map
    print('    Evaluating...')
    losses_binary = {}
    for layer in layers:
        mes_pred = nmfs[layer].predict(mes_preprocessed[layer])
        losses = np.square(mes_preprocessed[layer] - mes_pred)
        losses_binary[layer] = (losses > loss_threshold).astype(int)
    
    # optional: do automasking
    if do_automasking:
        print('    Applying automask...')
        runs = dfs[layer[0]]['run_number'].values
        lumis = dfs[layer[0]]['ls_number'].values
        for layer in layers:
            subsystem = f'BPix{layer}'
            automask_maps = amreader.get_automask_maps_for_ls(runs, lumis, subsystem, invert=True)
            automask_maps = automask_map_preprocessors[layer].preprocess_mes(automask_maps, None, None)
            losses_binary[layer] *= automask_maps

    # optional: do loss masking
    # update: now applied after combining layers instead of per-layer,
    # in order to be able to find cases where one layer is masked but another is not.
    '''
    if do_loss_masking:
        print('    Applying loss mask...')
        for layer in layers:
            mask = loss_masks[layer]
            mask = np.expand_dims(mask, 0)
            mask = loss_mask_preprocessors[layer].preprocess_mes(mask, None, None)
            losses_binary[layer] *= mask
    '''
        
    # optional: do filtering to keep only given patterns in the per-layer loss map
    if do_per_layer_cleaning:
        print('    Cleaning...')
        for layer in layers:
            losses_binary[layer] = patternfiltering.filter_any_pattern(
              losses_binary[layer],
              cleaning_patterns[layer],
              threshold = cleaning_threshold)
        
    # overlay different layers
    # strategy: sum the (rebinned) binary loss maps over different layers,
    # then apply the threshold >= 2 (i.e. overlapping anomaly in at least 2 layers)
    print('    Combining layers...')
    target_shape = losses_binary[layers[0]].shape[1:3]
    losses_binary_combined = np.zeros(losses_binary[layers[0]].shape)
    for layer in layers:
        losses_binary_rebinned = rebinning.rebin_keep_clip(losses_binary[layer], target_shape, 1, mode='cv2')
        losses_binary_combined += losses_binary_rebinned
    
    # optional: do loss masking
    loss_mask = np.zeros(losses_binary_combined.shape)
    if do_loss_masking:
        print('    Applying loss mask...')
        loss_mask = np.zeros((1, target_shape[0], target_shape[1]))
        for layer in layers:
            this_loss_mask = loss_masks[layer]
            # preprocess
            this_loss_mask = np.expand_dims(this_loss_mask, 0)
            this_loss_mask = loss_mask_preprocessors[layer].preprocess_mes(this_loss_mask, None, None)
            # invert
            this_loss_mask = 1 - this_loss_mask
            # rescale
            this_loss_mask = rebinning.rebin_keep_clip(this_loss_mask, target_shape, 1, mode='cv2')
            # add to total
            loss_mask += this_loss_mask
        loss_mask = np.repeat(loss_mask, len(losses_binary_combined), axis=0)
  
    # apply threshold on combined binary loss
    losses_binary_combined = ((losses_binary_combined >= 2) & (losses_binary_combined > loss_mask)).astype(int)
    
    # search for patterns in the combined loss
    print('    Searching for patterns in the loss map...')
    flags = patternfiltering.contains_any_pattern(losses_binary_combined, flagging_patterns,
              threshold = flagging_threshold)
    
    # store the flagged lumisections
    flagged_run_numbers = dfs[layers[0]]['run_number'].values[flags]
    flagged_ls_numbers = dfs[layers[0]]['ls_number'].values[flags]
    n_unique_runs = len(np.unique(dfs[layers[0]]['run_number'].values[flags]))
    print(f'    Found {np.sum(flags.astype(int))} flagged lumisections in {n_unique_runs} runs.')
    
    # explicitly delete some variables for memory management
    del dfs
    del mes_preprocessed
    del losses_binary

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
    loss_threshold = config['loss_threshold']
    flagging_patterns = [np.array(el) for el in config['flagging_patterns']]
    flagging_threshold = config['flagging_threshold']
    do_per_layer_cleaning = config['do_per_layer_cleaning']
    cleaning_patterns = {key: [np.array(el) for el in val] for key, val in config['cleaning_patterns'].items()}
    cleaning_threshold = config['cleaning_threshold']
    do_automasking = config['do_automasking']
    do_loss_masking = config['do_loss_masking']

    # read filters if needed
    min_entries_filter = config['min_entries_filter'] if 'min_entries_filter' in config.keys() else None
    oms_filters = None
    if 'oms_filter_keys' in config.keys():
        oms_filters = {}
        for era in eras:
            with open(config['oms_filter_files'][era], 'r') as f:
                oms_filters[era] = json.load(f)
            oms_filters[era] = {key: val for key, val in oms_filters[era].items() if key in config['oms_filter_keys']}

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

    # initialize result
    flagged_run_numbers = []
    flagged_ls_numbers = []
    batch_filter_results = []
    
    # loop over eras
    for era in eras:
        print(f'Era {era}...')
        
        # prepare batch parameters
        # (common to all layers)
        batch_params = dataloaders[era][layers[0]].prepare_sequential_batches(batch_size=batch_size)

        # run over batches
        for batchidx, batch_paramset in enumerate(batch_params):
            print(f'  Batch {batchidx+1}...')
            batch_results = run_evaluation_batch(batch_paramset, dataloaders[era], nmfs[era],
                                                 preprocessors = preprocessors[era],
                                                 min_entries_filter = min_entries_filter,
                                                 oms_filters = oms_filters[era],
                                                 loss_threshold = loss_threshold,
                                                 flagging_patterns = flagging_patterns,
                                                 flagging_threshold = flagging_threshold,
                                                 do_per_layer_cleaning = do_per_layer_cleaning,
                                                 cleaning_patterns = cleaning_patterns,
                                                 cleaning_threshold = cleaning_threshold,
                                                 do_automasking = do_automasking,
                                                 automask_reader = automask_reader,
                                                 automask_map_preprocessors = automask_map_preprocessors,
                                                 do_loss_masking = do_loss_masking,
                                                 loss_masks = None if loss_masks is None else loss_masks[era],
                                                 loss_mask_preprocessors = loss_mask_preprocessors)
            if batch_results is not None:
                batch_filter_results.append(batch_results['filter_results'])
                if len(batch_results['flagged_run_numbers'])>0:
                    flagged_run_numbers.append(batch_results['flagged_run_numbers'])
                    flagged_ls_numbers.append(batch_results['flagged_ls_numbers'])
                
            # break after one era for testing
            #break

    # contatenate the result
    filter_results = {}
    if len(batch_filter_results)>0:
        for key in batch_filter_results[0].keys():
            filter_results[key] = sum([batch_filter_result[key] for batch_filter_result in batch_filter_results], [])
    if len(flagged_run_numbers) > 0:
        flagged_run_numbers = np.concatenate(flagged_run_numbers)
        flagged_ls_numbers = np.concatenate(flagged_ls_numbers)
    else:
        flagged_run_numbers = np.array([])
        flagged_ls_numbers = np.array([])
        
    # return the result
    res = {
        'flagged_run_numbers': flagged_run_numbers,
        'flagged_ls_numbers': flagged_ls_numbers,
        'filter_results': filter_results
    }
    return res
        
        
if __name__=='__main__':
    
    # read job config
    configfile = sys.argv[1]
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

    # write output file
    outputfile = config['outputfile']
    outputdir = os.path.dirname(outputfile)
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    with open(outputfile, 'w') as f:
        json.dump(output, f)
