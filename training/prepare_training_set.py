import sys
import os
import numpy as np
import pandas as pd

def prepare_training_data_from_file( 
    parquet_file,
    verbose=False,
    entries_threshold=None,
    skip_first_lumisections=None ):
    
    # read the dataframe and get histograms
    if verbose: print('Loading file {}'.format(parquet_file))
    df = pd.read_parquet(parquet_file)
    nhists = len(df)
    xbins = df['Xbins'][0]
    ybins = df['Ybins'][0]
    hists = np.array([df['histo'][i].reshape(xbins,ybins) for i in range(nhists)])
    runs = np.array(df['fromrun'])
    lumis = np.array(df['fromlumi'])
    entries = np.array(df['entries'])
    masks = []
    if verbose:
        print('  Shape of hists array: {}'.format(hists.shape))
        print('  Runs: {}'.format(runs))
        print('  Lumis: {}'.format(lumis))
        print('  Entries: {}'.format(entries))

    # filter on number of entries
    if entries_threshold is not None:
        entries_mask = (entries > entries_threshold)
        masks.append(entries_mask)
        if verbose: 
            print('  Passing number of entries: {} ({:.2f} %)'.format(
                  np.sum(entries_mask), np.sum(entries_mask)/nhists*100))

    # filter on lumisection number
    if skip_first_lumisections is not None:
        lumisection_mask = (lumis > skip_first_lumisections)
        masks.append(lumisection_mask)
        if verbose:
            print('  Passing lumisection skip: {} ({:.2f} %)'.format(
                  np.sum(lumisection_mask), np.sum(lumisection_mask)/nhists*100))
        
    # other filters (to implement)
    
    # filter and format data
    training_mask = np.ones(nhists).astype(bool)
    for mask in masks: training_mask = (training_mask & mask)
    if verbose: print('  Training lumisections: {} ({:.2f} %)'.format(np.sum(training_mask), np.sum(training_mask)/nhists*100))
    training_data = hists[training_mask]
    training_runs = runs[training_mask]
    training_lumis = lumis[training_mask]
    training_data = np.expand_dims(training_data, 3)
    
    # return histograms and lumis
    return (training_data, training_runs, training_lumis)

def prepare_training_data_from_files( parquet_files, verbose=False, **kwargs ):
    res = [prepare_training_data_from_file(f, verbose=verbose, **kwargs) for f in parquet_files]
    training_data = np.concatenate(tuple([el[0] for el in res]))
    training_runs = np.concatenate(tuple([el[1] for el in res]))
    training_lumis = np.concatenate(tuple([el[2] for el in res]))
    if verbose:
        print('Shape of training data: {}'.format(training_data.shape))
        print('Shape of training runs: {}'.format(training_runs.shape))
        print('Shape of training lumis: {}'.format(training_lumis.shape))
    return (training_data, training_runs, training_lumis)