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
  creds = Credentials.from_creds_file()

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
    print('Now running on dataset {}'.format(dataset))

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
