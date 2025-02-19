import os
import sys
import json
import numpy as np


def find_oms_indices(runs, lumis, omsjson,
                     run_key='run_number', lumi_key='lumisection_number'):
    '''
    Find indices in and OMS dict for given run and lumisection numbers.
    Helper function for find_oms_attr_for_lumisections.
    '''
    # check if run_key and lumi_key are present in the omsjson
    if not run_key in omsjson.keys():
        msg = 'Run key "{}" not found in provided omsjson.'.format(run_key)
        msg += ' Available keys are: {}'.format(omsjson.keys())
        raise Exception(msg)
    if not lumi_key in omsjson.keys():
        msg = 'Lumi key "{}" not found in provided omsjson.'.format(lumi_key)
        msg += ' Available keys are: {}'.format(omsjson.keys())
        raise Exception(msg)
    # parse runs and lumis
    runs = np.array(runs).astype(int)
    lumis = np.array(lumis).astype(int)
    # combine run and lumisection number into a single unique number
    idfactor = 10000 # warning: do not use 1e4 instead of 10000 to avoid conversion from int to float
    ids = runs*idfactor + lumis
    ids = ids.astype(int)
    omsids = np.array(omsjson[run_key])*idfactor + np.array(omsjson[lumi_key])
    omsids = omsids.astype(int)
    # check if all ids are in omsids
    # note: reduce from error to warning,
    # since it seems some lumisections are intrinsically missing in OMS,
    # e.g. run 380147, LS 186.
    if np.any(np.isin(ids, omsids, invert=True)):
        missing_indices = np.isin(ids, omsids, invert=True).nonzero()[0]
        missing_ids = ids[missing_indices]
        msg = 'WARNING: not all provided lumisections could be found in the oms data.'
        msg += ' Missing lumisections are: {} ({} / {})'.format(missing_ids, len(missing_ids), len(ids))
        print(msg)
    # approach 1: using np.searchsorted
    # note: this will implicitly use neighbouring values for lumisections not found in the omsjson.
    omsids_sorted_inds = np.argsort(omsids)
    omsids_sorted = omsids[omsids_sorted_inds]
    indices = np.searchsorted(omsids_sorted, ids, side='left')
    indices = omsids_sorted_inds[indices]
    # approach 2: using np.isin
    # note: cleaner, but gives error on lumisections not found in the omsjson.
    #indices = np.isin(omsids, ids).nonzero()[0]
    return indices

def find_oms_attr_for_lumisections(runs, lumis, omsjson, omsattr, **kwargs):
    '''
    Retrieve an OMS attribute for given run and lumisection numbers.
    Input arguments:
    - runs and lumis: 1D arrays of the same length, with run and lumisection numbers.
    - omsjson: a dict with information from OMS, assumed to be of the form
               {"<attribute>": [<values>], ...}.
               the dict must have an attribute corresponding to run number,
               and an attribute corresponding to lumisection number.
               the names of these attributes can be passed in the kwargs
               (see find_oms_indices for more details).
    - omsattr: the name of the attribute to retrieve for the given lumisections.
    - kwargs: passed down to find_oms_indices
    Returns:
    - a 1D array of the same length as runs and lumis,
      with the values of the OMS attribute for the requested lumisections.
    '''
    # check if attribute is present
    if not omsattr in omsjson.keys():
        msg = 'Attribute "{}" not found in provided omsjson.'.format(omsattr)
        msg += ' Available keys are: {}'.format(omsjson.keys())
        raise Exception(msg)
    # retrieve indices for provided lumisections
    indices = find_oms_indices(runs, lumis, omsjson, **kwargs)
    return np.array(omsjson[omsattr])[indices]