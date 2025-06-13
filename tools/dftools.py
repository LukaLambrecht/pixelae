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