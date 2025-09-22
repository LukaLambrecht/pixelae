# **A collection of useful basic functions for manipulating pandas dataframes.**  
# 
# Functionality includes (among others):
# - selecting DCS-bit on data or golden json data.
# - selecting specific runs, lumisections, or types of histograms


### imports

# external modules
import os
import sys
import json
import numpy as np
import pandas as pd

# local modules
thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
import tools.jsontools as jsontools
from tools.omstools import find_oms_attr_for_lumisections
from tools.omstools import find_hlt_rate_for_lumisections


# getter and selector for histogram names 

def get_menames(df, menamecolumn='mename'):
    """
    Get a list of (unique) monitoring element names present in a df.
    """
    menamelist = sorted(list(set(df[menamecolumn].values)))
    return menamelist
    
def select_menames(df, menames, menamecolumn='mename'):
    """
    Keep only a subset of monitoring elements in a df.
    """
    df = df[df[menamecolumn].isin(menames)]
    df.reset_index(drop=True, inplace=True)
    return df


# getter and selector for run numbers

def get_runs(df, runcolumn='run'):
    """
    Return a list of (unique) run numbers present in a df.
    """
    runlist = sorted(list(set(df[runcolumn].values)))
    return runlist

def select_runs(df, runnbs, runcolumn='run'):
    """
    Keep only a subset of runs in a df.
    """
    df = df[df[runcolumn].isin(runnbs)]
    df.reset_index(drop=True, inplace=True)
    return df


# getter and selector for lumisection numbers

def get_ls(df, lumicolumn='lumi'):
    """
    Return a list of ls numbers present in a df.
    Note: the numbers are not required to be unique!
    Note: no check is done on the run number!
    """
    lslist = sorted(list(df[lumicolumn].values))
    return lslist

def select_ls(df, lsnbs, lumicolumn='lumi'):
    """
    Keep only a subset of lumisection numbers in a df.
    Note: no check is done on the run number!
    """
    df = df[df[lumicolumn].isin(lsnbs)]
    df.reset_index(drop=True, inplace=True)
    return df


### general getter and selector in json format

def get_runsls(df, runcolumn='run', lumicolumn='lumi'):
    """
    Return a dictionary with runs and lumisections in a dataframe,
    in the same format as e.g. golden json.
    """
    runslslist = get_runs(df, runcolum=runcolumn)
    for i,run in enumerate(runslslist):
        runslslist[i] = (run, get_ls( select_runs(df,[run],runcolumn=runcolumn), lumicolumn=lumicolumn))
    return jsontools.tuplelist_to_jsondict( runslslist )

def select_json(df, jsonfile, runcolumn='run', lumicolumn='lumi'):
    """
    Keep only lumisections that are in the given json file.
    """
    dfres = df[ jsontools.injson( df[runcolumn].values, df[runcolumn].values, jsonfile=jsonfile) ]
    dfres.reset_index(drop=True, inplace=True)
    return dfres

def select_runsls(df, jsondict, runcolumn='run', lumicolumn='lumi'):
    """
    Equivalent to select_json but using a pre-loaded json dict instead of a json file on disk.
    Input arguments:
    - jsondict can be either a dictionary in typical golden json format (see jsontools for more info)
      or a list of the following form: [(run number, lumisection number), ...]
    """
    if isinstance(jsondict, list):
        jsondict_new = {}
        for el in jsondict:
            run_number = str(el[0])
            ls_number = el[1]
            if run_number in jsondict_new: jsondict_new[run_number].append([ls_number, ls_number])
            else: jsondict_new[run_number] = [[ls_number, ls_number]]
        jsondict = jsondict_new
    dfres = df[ jsontools.injson( df[runcolumn].values, df[lumicolumn].values, jsondict=jsondict) ]
    dfres.reset_index(drop=True, inplace=True)
    return dfres


# getter and selector for sufficient statistics

def get_highstat(df, entriescolumn='entries', xbinscolumn='xbins', entries_to_bins_ratio=100):
    """
    Return a select object of runs and ls of histograms with high statistics.
    """
    return get_runsls(df[df[entriescolumn]/df[xbinscolumn]>entries_to_bins_ratio])

def select_highstat(df, entriescolumn='entries', xbinscolumn='xbins', entries_to_bins_ratio=100):
    """
    Keep only lumisection in df with high statistics.
    """
    return df[df[entriescolumn]/df[xbinscolumn]>entries_to_bins_ratio]
    
    
# functions to obtain histograms in np array format

def get_mes(df, datacolumn='data', xbinscolumn='xbins', ybinscolumn='ybins',
            runcolumn='run', lumicolumn='lumi',
            runs=None, lumis=None):
    """
    Get monitoring elements as a numpy array from a dataframe.
    Note: it is assumed that the df contains only one type of monitoring element!
    Note: for now only implemented for 2D monitoring elements!
    Input arguments:
    - df: dataframe
    - runs and lumis: 1D numpy arrays or lists (must be same length) with run and lumisection numbers to select
      (default: no selection is applied)
    Returns:
    - a tuple with the following elements:
      - numpy array of shape (number of instances, ybins, xbins) with the actual mes
      - numpy array of shape (number of instances) with run numbers
      - numpy array of shape (number of instances) with lumisection numbers
    """
    if runs is not None: df = select_runs(df, runs, runcolumn=runcolumn)
    if lumis is not None: df = select_ls(df, lumis, lumicolumn=lumicolumn)
    xbins = int(df[xbinscolumn].values[0])
    ybins = int(df[ybinscolumn].values[0])
    # note: df['data'][idx] yields an array of 1d arrays;
    # need to convert it to a 2d array with np.stack
    mes = np.array([np.stack(df[datacolumn].values[i]).reshape(ybins,xbins) for i in range(len(df))])
    runs = df[runcolumn].values
    lumis = df[lumicolumn].values
    return (mes, runs, lumis)


# advanced filtering

def filter_lumisections(run_numbers, ls_numbers,
        entries = None,
        min_entries_filter = None,
        oms_info = None,
        oms_filters = None,
        hltrate_info = None,
        hltrate_filters = None):
    '''
    Helper function to filter_dfs that does not rely on the actual dataframes.
    Can also be used standalone as an equivalent to filter_dfs,
    if the dataframes are not available but the equivalent information is.
    Input arguments:
    - run_numbers and ls_numbers: equally long 1D numpy arrays with run and lumisection numbers.
    - entries: dict of the form {<ME name>: <array with number of entries>, ...}.
               note: the array with number of entries is supposed to correspond to
                     run_numbers and ls_numbers; maybe generalize later.
    - min_entries_filter: dict of the form {<ME name>: <minimum number of entries for this ME>}.
    '''

    # initializations
    filter_results = {}
    combined_mask = np.ones(len(run_numbers)).astype(bool)

    # initialize ME names for min. entries filter
    menames = None
    if entries is not None and min_entries_filter is not None:
        menames = sorted(list(entries.keys()))
        testkeys = sorted(list(min_entries_filter.keys()))
        if menames != testkeys:
            msg = 'Keys in provided entries and min_entries_filter do not agree;'
            msg += f' found {menames} and {testkeys} respectively.'
            raise Exception(msg)

    # minimum number of entries filter
    if min_entries_filter is not None:
        for mename in menames:
            threshold = min_entries_filter[mename]
            mask = (entries[mename] > threshold)
            # add to the total mask
            combined_mask = ((combined_mask) & (mask))
            # keep track of lumisections that fail
            fail = [(run, ls) for run, ls in zip(run_numbers[~mask], ls_numbers[~mask])]
            filter_results[f'min_entries_{mename}'] = fail

    # OMS attribute filters
    if oms_filters is not None:
        for oms_filter in oms_filters:
            if len(oms_filter)==1:
                key = oms_filter[0]
                filterstr = key
                mask = find_oms_attr_for_lumisections(run_numbers, ls_numbers, oms_info, key).astype(bool)
            elif len(oms_filter)==3:
                key, operator, target = oms_filter
                filterstr = f'{key} {operator} {target}'
                values = find_oms_attr_for_lumisections(run_numbers, ls_numbers, oms_info, key)
                mask = eval(f'values {operator} {target}', {'values': values})
            else:
                raise Exception(f'Filter {oms_filter} not recognized.')
            # add to the total mask
            combined_mask = ((combined_mask) & (mask))
            # keep track of lumisections that fail
            fail = [(run, ls) for run, ls in zip(run_numbers[~mask], ls_numbers[~mask])]
            filter_results[filterstr] = fail

    # HLT rate filters
    if hltrate_filters is not None:
        for hltrate_filter in hltrate_filters:
            if len(hltrate_filter)==3:
                key, operator, target = hltrate_filter
                filterstr = f'{key} {operator} {target}'
                values = find_hlt_rate_for_lumisections(run_numbers, ls_numbers, hltrate_info, key)
                mask = eval(f'values {operator} {target}', {'values': values})
            else:
                raise Exception(f'Filter {hltrate_filter} not recognized.')
            # add to the total mask
            combined_mask = ((combined_mask) & (mask))
            # keep track of lumisections that fail
            fail = [(run, ls) for run, ls in zip(run_numbers[~mask], ls_numbers[~mask])]
            filter_results[filterstr] = fail

    # return results
    return (combined_mask, filter_results)

def filter_dfs(dfs, min_entries_filter=None, **kwargs):
    '''
    Filter a set of dataframes.
    Input arguments:
    - dfs: dict of dataframes of the form ME name -> dataframe
    '''

    # extract information from dataframes
    menames = sorted(list(dfs.keys()))
    run_numbers = dfs[menames[0]]['run_number'].values
    ls_numbers = dfs[menames[0]]['ls_number'].values
    entries = None
    if min_entreis_filters is not None:
        entries = {}
        for mename in menames:
            entries[mename] = dfs[mename]['entries'].values

    # return results
    return filter_data(run_numbers, ls_numbers,
            entries = entries,
            min_entries_filter = min_entries_filter,
            **kwargs)
