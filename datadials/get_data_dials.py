#!/usr/bin/env python3


# Get data with DIALS API


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
sys.path.append(os.path.abspath('../'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_14_0_4')


def get_creds(max_attempts=5):
  ### get dials credentials
  # the credential retrieval is essentially just a call to
  # Credentials.from_creds_file() (from the cmsdials api),
  # but wrapped in a while-try-except block,
  # to catch potential transient errors.
  authenticated = False
  auth_attempt_counter = 0
  while (auth_attempt_counter<max_attempts and not authenticated):
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
  return creds

def get_data(filters, max_attempts=5, max_pages=None):
  ### get dials data
  # the data retrieval is essentially just a call to
  # h2d.list_all (or similar for h1d, from the cmsdials api),
  # but wrapped in a while-try-except block,
  # to catch potential transient errors.
  
  # first define correct cmsdials api function
  # based on the type of filters
  if isinstance(filters, LumisectionHistogram1DFilters):
    dialsfunc = dials.h1d.list_all
  elif isinstance(filters, LumisectionHistogram2DFilters):
    dialsfunc = dials.h2d.list_all
  else:
    msg = 'ERROR: unrecognized type of DIALS filters: {}'.format(type(filters))
    raise Exception(msg)
  # make a wrapped call to cmsdials api
  data_retrieved = False
  attempt_counter = 0
  while (attempt_counter<max_attempts and not data_retrieved):
    attempt_counter += 1
    try:
      data = dialsfunc(filters, max_pages=max_pages, enable_progress=False)
      data_retrieved = True
    except: continue
  if not data_retrieved:
    # try one more time to trigger the original error
    data = dialsfunc(filters, max_pages=max_pages, enable_progress=False)
  sys.stdout.flush()
  sys.stderr.flush()
  return data


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Get data')
  parser.add_argument('-d', '--datasetnames', required=True,
    help='Path to a json file containing a list of dataset names,'
        +' may contain regex-style metacharacters or sets.')
  parser.add_argument('-m', '--menames', required=True,
    help='Path to a json file containing a list of monitoring elements,'
        +' may contain regex-style metacharacters or sets.')
  parser.add_argument('-t', '--metype', required=True, choices=['h1d', 'h2d'],
    help='Type of MEs (choose from "h1d" or "h2d"), needed for correct DIALS syntax.'
        +' Note: --menames json files with mixed 1D and 2D MEs are not supported,'
        +' they should be splitted and submitted separately.')
  parser.add_argument('-w', '--workspace', default='tracker',
    help='DIALS-workspace, see https://github.com/cms-DQM/dials-py?tab=readme-ov-file#workspace')
  parser.add_argument('-o', '--outputdir', default='.',
    help='Directory to store output parquet files into.')
  parser.add_argument('--resubmit', default=False, action='store_true',
    help='Only make output files that are not yet present in the output directory'
        +' (can be used if a small fraction of jobs failed because of transient errors).'
        +' Note: lines with regex-expressions will be used regardless.')
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'],
    help='Run directly in terminal ("local") or in HTCondor job ("condor").')
  parser.add_argument('--test', default=False, action='store_true',
    help='Truncate data for small and quick tests.')
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
    cmd += ' -t {}'.format(args.metype)
    cmd += ' -o {}'.format(args.outputdir)
    if args.resubmit: cmd += ' --resubmit'
    if args.test: cmd += ' --test'
    cmd += ' --runmode local'
    ct.submitCommandAsCondorJob('cjob_get_data', cmd,
      cmssw_version=CMSSW, home='auto')
    sys.exit()

  # print starting tag (for job completion checking)
  sys.stderr.write('###starting###\n')
  sys.stderr.flush()

  # make a list of datasets
  datasets = []
  print('Reading {}...'.format(args.datasetnames))
  with open(args.datasetnames, 'r') as f:
    datasets = json.load(f)
  print('Found following datasets:')
  for dataset in datasets: print('  - {}'.format(dataset))

  # make a list of monitoring elements
  mes = []
  print('Reading {}...'.format(args.menames))
  with open(args.menames, 'r') as f:
    mes = json.load(f)
  print('Found following MEs:')
  for me in mes: print('  - {}'.format(me))

  # do authentication
  creds = get_creds()

  # create Dials object
  dials = Dials(creds, workspace=args.workspace)

  # loop over datasets
  for datasetidx,dataset in enumerate(datasets):
    print('Now running on dataset {} ({}/{})'.format(dataset, datasetidx+1, len(datasets)))
    sys.stdout.flush()
    sys.stderr.flush()

    # retrieve run numbers
    # note: this is done to make separate calls per run 
    #       instead of one giant call for a full dataset
    runfilters = RunFilters(dataset__regex=dataset)
    runs = dials.run.list_all(runfilters, enable_progress=False).results
    runs = sorted([el.run_number for el in runs])
    print('Found {} runs'.format(len(runs)))

    # loop over mes
    for meidx,me in enumerate(mes):
      print('Now running on ME {} ({}/{})'.format(me, meidx+1, len(mes)))
      sys.stdout.flush()
      sys.stderr.flush()
      dfs = []

      # check if output file already exists and if so, skip this part
      # (if requested)
      if args.resubmit:
        outputfile = (dataset+'-'+me).strip('/').replace('/','-')+'.parquet'
        outputfile = outputfile.replace('\\','')
        outputfile = os.path.join(args.outputdir, outputfile)
        if os.path.exists(outputfile):
          print('Output file {} already exists, skipping this part.'.format(outputfile))
          continue

      # loop over runs
      for runidx,run in enumerate(runs):
        print('  - Run {} ({}/{})'.format(run, runidx+1, len(runs)))
        sys.stdout.flush()
        sys.stderr.flush()

        # define filter
        if args.metype=='h1d':
          dialsfilters = LumisectionHistogram1DFilters(
            dataset__regex = dataset,
            me__regex = me,
            run_number = run
          )
        elif args.metype=='h2d':
          dialsfilters = LumisectionHistogram2DFilters(
            dataset__regex = dataset,
            me__regex = me,
            run_number = run
          )
        else:
          msg = 'ERROR: metype {} not recognized'.format(args.metype)
          raise Exception(msg)
        max_pages = None
        if args.test: max_pages = 1

        # make the dials request
        data = get_data(dialsfilters, max_pages=max_pages)

        # convert to dataframe
        df = data.to_pandas()
        dfs.append(df)

      # concatenate results for all runs
      df = pd.concat(dfs, ignore_index=True)

      # check if the dataframe is empty
      if len(df)==0:
        msg = 'ERROR: resulting dataframe is empty, cannot write output file.'
        raise Exception(msg)

      # check if the dataframe contains multiple datasets and/or MEs
      # (can occur if the provided dataset/ME name is a regular expression),
      # and if so, split into one dataframe per dataset and ME.
      dfdict = {}
      datasetnames = list(set(df['dataset']))
      menames = list(set(df['me']))
      for datasetname in datasetnames:
        dfdict[datasetname] = {}
        for mename in menames:
          dfdict[datasetname][mename] = df[(df['me']==mename) & (df['dataset']==datasetname)]

      # write to output file(s)
      print('Writing output file(s)...')
      if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)
      for datasetname in dfdict.keys():
        for mename in dfdict[datasetname].keys():
          outputfile = (datasetname+'-'+mename).strip('/').replace('/','-')+'.parquet'
          outputfile = outputfile.replace('\\','')
          outputfile = os.path.join(args.outputdir, outputfile)
          dfdict[datasetname][mename].to_parquet(outputfile)

  # print finishing tag (for job completion checking)
  sys.stderr.write('###done###\n')
  sys.stderr.flush()
