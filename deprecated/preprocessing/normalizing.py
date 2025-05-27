import sys
import os
import json
import numpy as np

from omstools import find_oms_attr_for_lumisections


def normalize_by_external(histograms, normvalues, zerohandling='ones'):
    # check for zeros
    if not np.all(normvalues):
        msg = 'WARNING in normalize_by_external: some normvalues are zero,'
        if zerohandling=='ones':
            msg += ' will set zeros to one.'
            print(msg)
            normvalues[normvalues==0] = 1
        elif zerohandling=='avgneighbours':
            msg += ' will try to use average of neighbours'
            print(msg)
            for idx in np.nonzero(normvalues==0)[0]:
                try: normvalues[idx] = (normvalues[idx-1] + normvalues[idx+1])/2.
                except: pass
            return normalize_by_external(histograms, normvalues, zerohandling='ones')
        else: raise Exception('ERROR: zero handling {} not recognized.'.format(zerohandling))
    return np.divide(histograms, normvalues[:, None, None])

def normalize_by_omsjson(histograms, runs, lumis, omsjson, omsattr, **kwargs):
    # check initial size
    if( len(runs)!=len(histograms) or len(lumis)!=len(histograms) ):
        msg = 'ERROR: provided runs and/or lumis do not match the number of histograms.'
        raise Exception(msg)
    # get values
    normvalues = find_oms_attr_for_lumisections(runs, lumis, omsjson, omsattr)
    # normalize
    return normalize_by_external(histograms, normvalues, **kwargs)

def normalize_by_omsfile(histograms, runs, lumis, omsfile, omsattr, **kwargs):
    # load the file
    with open(omsfile, 'r') as f:
        omsjson = json.load(f)
    # normalize by oms values
    return normalize_by_omsjson(histograms, runs, lumis, omsjson, omsattr, **kwargs)