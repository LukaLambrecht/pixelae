import os
import sys
import json
import numpy as np


def find_oms_indices(runs, lumis, omsjson):
    runs = np.array(runs).astype(int)
    lumis = np.array(lumis).astype(int)
    idfactor = 10000 # warning: do not use 1e4 instead of 10000 to avoid conversion from int to float
    ids = runs*idfactor + lumis
    omsids = np.array(omsjson['run_number'])*idfactor + np.array(omsjson['lumisection_number'])
    omsids_sorted_inds = np.argsort(omsids)
    omsids_sorted = omsids[omsids_sorted_inds]
    indices = np.searchsorted(omsids_sorted, ids, side='left')
    if( np.any(indices>=len(omsids_sorted)) ):
        msg = 'ERROR: not all provided lumisections could be found in the oms data.'
        raise Exception(msg)
    indices = omsids_sorted_inds[indices]
    return indices

def find_oms_attr_for_lumisections(runs, lumis, omsjson, omsattr):
    indices = find_oms_indices(runs, lumis, omsjson)
    return np.array(omsjson[omsattr])[indices]