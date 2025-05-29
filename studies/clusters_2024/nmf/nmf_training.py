# imports

# external modules
import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd

# local modules
thisdir = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)

import tools.dftools as dftools
from tools.dataloadertools import MEDataLoader
from studies.clusters_2024.preprocessing.preprocessor import make_default_preprocessor
from studies.clusters_2024.nmf.modeldefs.nmf2d import NMF2D


def find_files(layer):
    # settings
    datadir = '/eos/user/l/llambrec/dialstools-output'
    year = '2024'
    eras = {
      'A': ['v1'],
      'B': ['v1'],
      'C': ['v1'],
      'D': ['v1'],
      'E': ['v1', 'v2'],
      'F': ['v1', 'v1-part1', 'v1-part2', 'v1-part3', 'v1-part4'],
      'G': ['v1', 'v1-part1', 'v1-part2', 'v1-part3', 'v1-part4'],
      'H': ['v1'],
      'I': ['v1', 'v2'],
      'J': ['v1']
    }
    dataset = 'ZeroBias'
    reco = 'PromptReco'
    mebase = 'PixelPhase1-Phase1_MechanicalView-PXBarrel-'
    mebase += 'clusters_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_{}'
    me = mebase.format(layer)

    # find files corresponding to settings
    files = {}
    for era, versions in eras.items():
        for version in versions:
            f = f'{dataset}-Run{year}{era}-{reco}-{version}-DQMIO-{me}.parquet'
            f = os.path.join(datadir, f)
            files[f'{era}-{version}'] = f

    # existence check
    missing = []
    for f in files.values():
        if not os.path.exists(f):
            missing.append(f)
    if len(missing) > 0:
        raise Exception(f'The following files do not exist: {missing}')
    
    # return result
    return files


def train(dataloader, nmf,
      nbatches=1, batch_size=1,
      min_entries=None, preprocessor=None,
      verbose=False):

    # loop over random batches
    for batchidx in range(nbatches):
        
        # load batch
        if verbose: print(f'Now processing batch {batchidx+1} / {nbatches}...')
        df = dataloader.read_random_batch(batch_size=batch_size, mode='subbatched', num_subbatches=100)
        ndf = len(df)
        
        # filtering
        if min_entries is not None: df = df[df['entries'] > min_entries]
        if verbose: print(f'  Found {len(df)} / {ndf} instances passing filters.')
        if len(df)==0: continue
        
        # do preprocessing
        if preprocessor is not None:
            if verbose: print('  Preprocessing...')
            mes_preprocessed = preprocessor.preprocess(df)
        else:
            mes_preprocessed, _, _ = dftools.get_mes(df,
                                       xbinscolumn='x_bin', ybinscolumn='y_bin',
                                       runcolumn='run_number', lumicolumn='ls_number')
        
        # experimental: set zero-occupancy to 1 (average expected value after preprocessing)
        mes_preprocessed[mes_preprocessed==0] = 1
        
        # fit NMF
        if verbose: print('  Training NMF...')
        nmf.fit(mes_preprocessed)

    return nmf


if __name__=='__main__':

    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--era', required=True)
    parser.add_argument('--layer', required=True)
    parser.add_argument('--outputfile', required=True)
    parser.add_argument('--min_entries', default=0, type=float)
    parser.add_argument('--n_components', default=5, type=int)
    parser.add_argument('--forget_factor', default=1, type=float)
    parser.add_argument('--tol', default=0.0, type=float)
    parser.add_argument('--max_no_improvement', default=100, type=int)
    parser.add_argument('--max_iter', default=1000, type=int)
    parser.add_argument('--alpha_H', default=0.1, type=float)
    parser.add_argument('--nbatches', default=1, type=int)
    parser.add_argument('--max_epochs', default=-1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()
    
    # find input files, make data loader and preprocessor
    files = find_files(args.layer)
    files = files[args.era]
    dataloader = MEDataLoader([files])
    preprocessor_era = args.era
    if '-part' in args.era: preprocessor_era = args.era.split('-part')[0]
    preprocessor = make_default_preprocessor(preprocessor_era, int(args.layer))

    # determine number of batches
    nbatches = args.nbatches
    if args.max_epochs > 0:
        nrows = sum(dataloader.nrows)
        batches_per_epoch = int(nrows/args.batch_size)
        nbatches = min(nbatches, args.max_epochs * batches_per_epoch)
    
    # make nmf
    nmf = NMF2D(
            n_components = args.n_components,
            forget_factor = args.forget_factor,
            batch_size = args.batch_size,
            verbose=True,
            tol = args.tol,
            max_no_improvement = args.max_no_improvement,
            max_iter = args.max_iter,
            alpha_H = args.alpha_H
    )

    # do training
    nmf = train(dataloader, nmf,
            nbatches=nbatches, batch_size=args.batch_size,
            min_entries=args.min_entries, preprocessor=preprocessor,
            verbose=True)

    # store model
    outputdir = os.path.dirname(args.outputfile)
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    joblib.dump(nmf, args.outputfile)
