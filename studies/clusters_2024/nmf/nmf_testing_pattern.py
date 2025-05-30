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
import tools.patternfiltering as patternfiltering
import tools.rebinning as rebinning
from automasking.tools.automaskreader import AutomaskReader
from studies.clusters_2024.preprocessing.preprocessor import PreProcessor
from studies.clusters_2024.preprocessing.preprocessor import make_default_preprocessor


def make_dataloaders(input_file_dict):
    dataloaders = {}
    for era, layers in input_file_dict.items():
        dataloaders[era] = {}
        for layer, files in layers.items(): dataloaders[era][layer] = MEDataLoader(files)
    return dataloaders
        
def make_preprocessors(eras, layers):
    preprocessors = {}
    for era in eras:
        preprocessors[era] = {}
        preprocessor_era = era
        if '-part' in era: preprocessor_era = era.split('-part')[0]
        for layer in layers: preprocessors[era][layer] = make_default_preprocessor(preprocessor_era, layer)
    return preprocessors    

def load_nmfs(nmf_file_dict):
    nmfs = {}
    for era, layers in nmf_file_dict.items():
        nmfs[era] = {}
        for layer, modelfile in layers.items(): nmfs[era][layer] = joblib.load(modelfile)
    return nmfs
        
def run_evaluation_batch(batch_paramset, dataloaders, nmfs, **kwargs):
    
    # initializations
    start_tim = time.time()
    
    # get the dataframes
    dfs = {}
    layers = list(dataloaders.keys())
    for layer in layers:
        dfs[layer] = dataloaders[layer].read_sequential_batch(batch_paramset)
    
    # run the evaluation on this data
    flagged_run_numbers, flagged_ls_numbers = run_evaluation(dfs, nmfs, **kwargs)
    
    # calculate time spent for this batch
    end_time = time.time()
    print('    Time for this batch: {:.2f}s'.format(end_time-start_time))
    
    # return the result
    return flagged_run_numbers, flagged_ls_numbers
    
def run_evaluation(dfs, nmfs,
                     preprocessors = None,
                     threshold = 0.1,
                     flag_patterns = None,
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
    mask = np.ones(ndf).astype(bool)
    for layer in layers: mask = (mask & (dfs[layer]['entries'] > (0.5e6/int(layer))))
    for layer in layers:
        dfs[layer] = dfs[layer][mask]
    ndfnew = len(dfs[layers[0]])    
    print(f'    Found {ndfnew} / {ndf} instances passing filters.')
    if ndfnew==0: return ([], [])
        
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
        losses_binary[layer] = (losses > threshold).astype(int)
    
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
    if do_loss_masking:
        print('    Applying loss mask...')
        for layer in layers:
            mask = loss_masks[layer]
            mask = np.expand_dims(mask, 0)
            mask = loss_mask_preprocessors[layer].preprocess_mes(mask, None, None)
            losses_binary[layer] *= mask
            
    # optional: do filtering to keep only given patterns in the per-layer loss map
    if do_per_layer_cleaning:
        print('    Cleaning...')
        for layer in layers:
            losses_binary[layer] = patternfiltering.filter_any_pattern(
              losses_binary[layer],
              cleaning_patterns,
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
    losses_binary_combined = (losses_binary_combined >= 2).astype(int)
    
    # search for patterns in the combined loss
    print('    Searching for patterns in the loss map...')
    flags = patternfiltering.contains_any_pattern(losses_binary_combined, flag_patterns)
    
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
    return flagged_run_numbers, flagged_ls_numbers

def evaluate(config):
        
    # get eras and layers
    eras = config['eras']
    layers = config['layers']
        
    # make dataloaders, preprocessors and models
    dataloaders = make_dataloaders(config['input_files'])
    preprocessors = make_preprocessors(eras, layers)
    nmfs = load_nmfs(config['nmf_files'])
    
    # get evaluation settings
    batch_size = config['batch_size']
    threshold = config['threshold']
    flag_patterns = [np.array(el) for el in config['flag_patterns']]
    do_per_layer_cleaning = config['do_per_layer_cleaning']
    cleaning_patterns = [np.array(el) for el in config['cleaning_patterns']]
    cleaning_threshold = config['cleaning_threshold']
    do_automasking = config['do_automasking']
    do_loss_masking = config['do_loss_masking']
    
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
                loss_mask_file = config['loss_mask_files'][era][layer]
                loss_mask = np.load(loss_mask_file)
                loss_mask = (loss_mask < 0.9)
                loss_masks[era][layer] = loss_mask
        for layer in layers: loss_mask_preprocessors[layer] = PreProcessor(f'PXLayer_{layer}')

    # initialize result
    flagged_run_numbers = []
    flagged_ls_numbers = []
    
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
                                                 threshold = threshold,
                                                 flag_patterns = flag_patterns,
                                                 do_per_layer_cleaning = do_per_layer_cleaning,
                                                 cleaning_patterns = cleaning_patterns,
                                                 cleaning_threshold = cleaning_threshold,
                                                 do_automasking = do_automasking,
                                                 automask_reader = automask_reader,
                                                 automask_map_preprocessors = automask_map_preprocessors,
                                                 do_loss_masking = do_loss_masking,
                                                 loss_masks = loss_masks[era],
                                                 loss_mask_preprocessors = loss_mask_preprocessors)
            if batch_results is not None and len(batch_results[0]) > 0:
                flagged_run_numbers.append(batch_results[0])
                flagged_ls_numbers.append(batch_results[1])
                
            # break after one era for testing
            #break

    # contatenate the result
    if len(flagged_run_numbers) > 0:
        flagged_run_numbers = np.concatenate(flagged_run_numbers)
        flagged_ls_numbers = np.concatenate(flagged_ls_numbers)
        
    # return the result
    return flagged_run_numbers, flagged_ls_numbers
        
        
if __name__=='__main__':
    
    # read job config
    configfile = sys.argv[1]
    with open(configfile, 'r') as f:
        config = json.load(f)
        
    # do evaluation
    flagged_run_numbers, flagged_ls_numbers = evaluate(config)
    
    # write output
    output = {'flagged_run_numbers': flagged_run_numbers.tolist(), 
              'flagged_ls_numbers': flagged_ls_numbers.tolist()}
    outputfile = config['outputfile']
    outputdir = os.path.dirname(outputfile)
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    with open(outputfile, 'w') as f:
        json.dump(output, f)
