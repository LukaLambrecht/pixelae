#!/usr/bin/env python3

# Get data with DIALS API
# Looper over datasets and monitoring elements

# imports
import os
import sys
import six
import json
import argparse
# local imports
sys.path.append(os.path.abspath('../'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_14_0_4')


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
  parser.add_argument('--splitdatasets', default=False, action='store_true',
    help='Submit separate jobs for each dataset in the provided list.'
        +' Note: lines with regex-expressions are kept in a single job.')
  parser.add_argument('--splitmes', default=False, action='store_true',
    help='Submit separate jobs for each monitoring element in the provided list'
        +' Note: lines with regex-expressions are kept in a single job.')
  parser.add_argument('--resubmit', default=False, action='store_true',
    help='Submit only jobs for output files that are not yet present in the output directory'
        +' (can be used if a small fraction of jobs failed because of transient errors).'
        +' Note: lines with regex-expressions will be resubmitted regardless.')
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'],
    help='Run directly in terminal ("local") or in HTCondor job ("condor").')
  parser.add_argument('--test', default=False, action='store_true',
    help='Truncate loop and data for small and quick tests.')
  args = parser.parse_args()
  
  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # make a list of datasets
  datasets = []
  with open(args.datasetnames, 'r') as f:
    datasets = json.load(f)

  # make a list of monitoring elements
  menames = []
  with open(args.menames, 'r') as f:
    menames = json.load(f)

  # handle splitting per dataset
  datasetfiles = [args.datasetnames]
  if args.splitdatasets:
    datasetfiles = []
    for i,dataset in enumerate(datasets):
      datasetfile = 'temp_datasets_{}.json'.format(i)
      datasetfiles.append(datasetfile)
      with open(datasetfile, 'w') as f:
        json.dump([dataset], f)

  # handle splitting per monitoring element
  mefiles = [args.menames]
  if args.splitmes:
    mefiles = []
    for i,mename in enumerate(menames):
      mefile = 'temp_menames_{}.json'.format(i)
      mefiles.append(mefile)
      with open(mefile, 'w') as f:
        json.dump([mename], f)

  # make output directory
  if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

  # check which output files are already present
  # to skip the corresponding jobs (if requested)
  # note: depends on naming convention in get_data_dials.py!
  # note: only relevant here if splitdatasets and splitmes are true,
  #       else just need to pass down to get_data_dials.py
  existing_output_files = []
  veto_jobs = []
  if( args.splitdatasets and args.splitmes and args.resubmit ):
    keep_jobs = []
    for dataset, datasetfile in zip(datasets,datasetfiles):
      for me, mefile in zip(menames, mefiles):
        outputfile = (dataset+'-'+me).strip('/').replace('/','-')+'.parquet'
        outputfile = outputfile.replace('\\','')
        outputfile = os.path.join(args.outputdir, outputfile)
        if os.path.exists(outputfile):
          existing_output_files.append(outputfile)
          veto_jobs.append((datasetfile,mefile))
        else: keep_jobs.append((dataset,me))
    print('Note: the following output files already exist ({}):'.format(len(existing_output_files)))
    for f in sorted(existing_output_files): print('  - {}'.format(f))
    print('Will submit only the remaining jobs ({}):'.format(len(keep_jobs)))
    for (dataset,me) in keep_jobs: print('  - {} / {}'.format(dataset,me))
    print('Continue? (y/n)')
    go = six.moves.input()
    if not go=='y': sys.exit()

  # loop over datasets and monitoring elements
  cmds = []
  for datasetfile in datasetfiles:
    for mefile in mefiles:
      if (datasetfile,mefile) in veto_jobs: continue
      cmd = 'python3 get_data_dials.py'
      cmd += ' -d {}'.format(datasetfile)
      cmd += ' -m {}'.format(mefile)
      cmd += ' -t {}'.format(args.metype)
      cmd += ' -w {}'.format(args.workspace)
      cmd += ' -o {}'.format(args.outputdir)
      if args.resubmit: cmd += ' --resubmit'
      if args.test: cmd += ' --test'
      cmd += ' --runmode local'
      cmds.append(cmd)

  # limit for testing
  if args.test: cmds = [cmds[0]]

  # run or submit commands
  if args.runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_get_data', cmds,
      cmssw_version=CMSSW, home='auto')
  else:
    for cmd in cmds:
      print(cmd)
      os.system(cmd)
