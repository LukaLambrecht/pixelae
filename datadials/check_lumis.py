#!/usr/bin/env python3

# Check available lumisections in parquet files w.r.t. original datasets on DAS

# imports
import sys
import os
import json
import argparse
import pandas as pd
import numpy as np


def get_lumis_das(datasetname):
  ### get lumisections in a dataset from DAS
  # returns:
  # a tuple of equally long numpy arrays: (runs, lumisections)
  dascmd = "dasgoclient -query 'run lumi dataset={}' --limit 0".format(datasetname)
  dasstdout = os.popen(dascmd).read()
  runlumis = sorted([el.strip(' \t') for el in dasstdout.strip('\n').split('\n')])
  # format of runlumis is a list with strings of format '<run nb> [<ls nbs>]'
  if len(runlumis)==1 and runlumis[0]=='':
    runs = np.array([])
    lumis = np.array([])
    return (runs, lumis)
  allruns = []
  alllumis = []
  for runlumi in runlumis:
    run = int(runlumi.split(' ',1)[0])
    lumis = runlumi.split(' ',1)[1]
    lumis = lumis.strip('[] ')
    lumis = lumis.split(',')
    lumis = [int(lumi) for lumi in lumis]
    lumis = set(lumis)
    # (note: the above is to remove duplicates that are sometimes observed;
    #  not sure where it is coming from or what it means...)
    lumis = sorted(list(lumis))
    for lumi in lumis:
      allruns.append(run)
      alllumis.append(lumi)
  return (np.array(allruns), np.array(alllumis))

def get_lumis_parquet(pfile):
  ### get lumisections in a parquet file
  # returns:
  # a tuple of equally long numpy arrays: (runs, lumisections)
  df = pd.read_parquet(pfile, columns=['run_number', 'ls_number'])
  return (np.array(df['run_number']), np.array(df['ls_number']))
  
  
if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Check available lumisections')
  parser.add_argument('-d', '--datasets', required=True)
  parser.add_argument('-p', '--parquetfiles', required=True, nargs='+')
  parser.add_argument('--print_missing', default=False, action='store_true')
  args = parser.parse_args()
  
  # print arguments
  #print('Running with following configuration:')
  #for arg in vars(args):
  #  print('  - {}: {}'.format(arg,getattr(args,arg)))
    
  # internal parameters
  idfactor = 10000 # warning: do not use 1e4 instead of 10000 to avoid conversion from int to float

  # make a list of datasets
  datasets = []
  with open(args.datasets, 'r') as f:
    datasets = json.load(f)

  # find out which parquet files belong to which dataset
  dataset_files = {dataset.replace('-','').replace('/',''): [] for dataset in datasets}
  for parquetfile in args.parquetfiles:
    tag = (parquetfile.split('/')[-1].split('DQMIO')[0]+'DQMIO').replace('-','')
    if tag in dataset_files.keys(): dataset_files[tag].append(parquetfile)
    else:
      msg = 'WARNING: no matching dataset found for file {}'.format(parquetfile)
      print(msg)

  # check missing lumisections in files
  print('Checking missing lumisections in files:')
  for dataset in datasets:
    refruns, reflumis = get_lumis_das(dataset)
    refids = refruns*idfactor + reflumis
    print('  Dataset: {} ({} lumisections)'.format(dataset, len(refruns)))
    # loop over corresponding files
    parquetfiles = dataset_files[dataset.replace('-','').replace('/','')]
    print('  Corresponding files ({}):'.format(len(parquetfiles)))
    for parquetfile in parquetfiles:
      runs, lumis = get_lumis_parquet(parquetfile)
      ids = runs*idfactor + lumis
      missing = np.setdiff1d(refids, ids)
      print('    - File: {}: {} missing lumisections'.format(parquetfile, len(missing)))
      if( len(missing)>0 and args.print_missing ):
        missing_runs = (missing/idfactor).astype(int)
        missing_lumis = np.remainder(missing, idfactor).astype(int)
        missing_runlumis = [(run, lumi) for run, lumi in zip(missing_runs, missing_lumis)]
        print('            {}'.format(missing_runlumis))

  # check overlap between datasets
  print('Checking overlap between datasets:')
  allids = np.array([])
  for dataset in datasets:
    runs, lumis = get_lumis_das(dataset)
    ids = runs*idfactor + lumis
    unique = np.setdiff1d(ids, allids)
    allids = np.concatenate((allids, unique))
    print('  Dataset: {}: {} lumisections of which {} unique'.format(dataset, len(ids), len(unique)))

  # check overlap between files
  print('Checking overlap between files:')
  allids = np.array([])
  for dataset in datasets:
    parquetfile = dataset_files[dataset.replace('-','').replace('/','')]
    if len(parquetfile)<1:
      print('  (Checking dataset {}, found no corresponding files)'.format(dataset))
      continue
    parquetfile = parquetfile[0]
    runs, lumis = get_lumis_parquet(parquetfile)
    ids = runs*idfactor + lumis
    unique = np.setdiff1d(ids, allids)
    allids = np.concatenate((allids, unique))
    print('  File: {}: {} lumisections of which {} unique'.format(parquetfile, len(ids), len(unique)))
