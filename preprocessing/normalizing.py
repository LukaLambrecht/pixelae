import sys
import os
import json
import numpy as np


def normalize_by_external(histograms, normvalues):
    # check for zeros
    if not np.all(normvalues):
        print('WARNING in normalize_by_external: some normvalues are zero, setting to one.')
        normvalues[normvalues==0] = 1
    return np.divide(histograms, normvalues[:, None, None])

def normalize_by_omsjson(histograms, runs, lumis, omsjson, omsattr):
    # check initial size
    if( len(runs)!=len(histograms) or len(lumis)!=len(histograms) ):
        msg = 'ERROR: provided runs and/or lumis do not match the number of histograms.'
        raise Exception(msg)
    # get mask
    ids = runs*1e4 + lumis
    omsids = np.array(omsjson['run_number'])*1e4 + np.array(omsjson['lumisection_number'])
    mask = np.isin(omsids, ids)
    # check size of mask
    if( np.sum(mask)<len(histograms) ):
        missing = np.setdiff1d(ids, omsids)
        missing = [(int(el/1e4),el%1e4) for el in missing]
        msg = 'ERROR: not all provided lumisections could be found in the oms data;'
        msg += ' missing lumisections ({}): {}'.format(len(missing), missing)
        raise Exception(msg)
    # get values
    normvalues = np.array(omsjson[omsattr])[mask]
    # normalize
    return normalize_by_external(histograms, normvalues)

def normalize_by_omsfile(histograms, runs, lumis, omsfile, omsattr):
    # load the file
    with open(omsfile, 'r') as f:
        omsjson = json.load(f)
    # normalize by oms values
    return normalize_by_omsjson(histograms, runs, lumis, omsjson, omsattr)