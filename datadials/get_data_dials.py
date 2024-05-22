#!/usr/bin/env python3

# Get data with DIALS API
# Note: not sure how authentication is handled in job submission,
#       but it seems to work automagically if the credentials are already cached.
#       So first run a small test locally to do the authentication,
#       then jobs can be submitted without issues.


# general imports
import os
import sys
import json
import numpy as np
import pandas as pd
import argparse

# dials imports
from cmsdials import Dials
from cmsdials.auth.bearer import Credentials
from cmsdials.filters import LumisectionFilters
from cmsdials.filters import LumisectionHistogram1DFilters
from cmsdials.filters import LumisectionHistogram2DFilters
from cmsdials.filters import RunFilters

# local imports
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_14_0_4')


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Get data')
  parser.add_argument('-d', '--datasetnames', required=True)
  parser.add_argument('-m', '--menames', required=True)
  parser.add_argument('-o', '--outputdir', default='.')
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'])
  parser.add_argument('--test', default=False, action='store_true')
  args = parser.parse_args()
  
  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # handle job submission if requested
  if args.runmode=='condor':
    cmd = 'python3 get_data_dials.py'
    cmd += ' -d {}'.format(args.datasetnames)
    cmd += ' -m {}'.format(args.menames)
    cmd += ' -o {}'.format(args.outputdir)
    if args.test: cmd += ' --test'
    cmd += ' --runmode local'
    ct.submitCommandAsCondorJob('cjob_get_data', cmd,
      cmssw_version=CMSSW, home='auto')
    sys.exit()

  # print starting tag (for job completion checking)
  sys.stderr.write('###starting###\n')
  sys.stderr.flush()

  # do authentication
  # (wrap in while-try-except block because of observed apparently random json errors)
  authenticated = False
  auth_attempt_counter = 0
  while (auth_attempt_counter<5 and not authenticated):
    auth_attempt_counter += 1
    print('Retrieving cmsdials credentials from cache...')
    try:
      creds = Credentials.from_creds_file()
      authenticated = True
    except: continue
  if not authenticated:
    # try one more time to trigger the original error again
    creds = Credentials.from_creds_file()
  sys.stdout.flush()
  sys.stderr.flush()

  # create Dials object
  dials = Dials(creds, workspace='tracker')

  # make a list of datasets
  datasets = []
  with open(args.datasetnames, 'r') as f:
    datasets = json.load(f)

  # make a list of monitoring elements
  mes = []
  with open(args.menames, 'r') as f:
    mes = json.load(f)

  # loop over datasets
  for dataset in datasets:
    print('Now running on dataset {}...'.format(dataset))
    sys.stdout.flush()
    sys.stderr.flush()

    # retrieve run numbers
    # note: this is needed in an attempt to solve timeout errors;
    #       make separate calls per run instead of one giant call for a full dataset
    runfilters = RunFilters(dataset=dataset)
    runs = dials.run.list_all(runfilters).results
    runs = sorted([el.run_number for el in runs])
    print('Found {} runs'.format(len(runs)))

    # loop over mes
    for me in mes:
      print('Now running ME {}'.format(me))
      sys.stdout.flush()
      sys.stderr.flush()
      dfs = []

      # loop over runs
      for run in runs:

        # define filter
        h2dfilters = LumisectionHistogram2DFilters(
          dataset = dataset,
          me = me,
          run_number = run
        )
        max_pages = None
        if args.test: max_pages = 1

        # make the dials request
        # (wrap in while-try-except block in an attempt to solve timeout and other errors)
        data_retrieved = False
        attempt_counter = 0
        while (attempt_counter<5 and not data_retrieved):
          attempt_counter += 1
          try:
            data = dials.h2d.list_all(h2dfilters, max_pages=max_pages)
            data_retrieved = True
          except: continue
        if not data_retrieved:
          # try one more time to trigger the original error
          data = dials.h2d.list_all(h2dfilters, max_pages=max_pages)  

        # convert to dataframe
        df = data.to_pandas()
        dfs.append(df)

      # concatenate results for all runs
      df = pd.concat(dfs, ignore_index=True)

      # write to output file
      print('Writing output file...')
      if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)
      outputfile = (dataset+'-'+me).strip('/').replace('/','-')+'.parquet'
      outputfile = os.path.join(args.outputdir, outputfile)
      df.to_parquet(outputfile)

  # print finishing tag (for job completion checking)
  sys.stderr.write('###done###\n')
  sys.stderr.flush()
