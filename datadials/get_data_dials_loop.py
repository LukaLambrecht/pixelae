#!/usr/bin/env python3

# Get data with DIALS API
# Looper over datasets and monitoring elements

# imports
import os
import sys
import json
import argparse
# local imports
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_14_0_4')


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Get data')
  parser.add_argument('-d', '--datasets', required=True)
  parser.add_argument('-m', '--menames', required=True)
  parser.add_argument('-o', '--outputdir', default='.')
  parser.add_argument('--splitdatasets', default=False, action='store_true')
  parser.add_argument('--splitmes', default=False, action='store_true')
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'])
  parser.add_argument('--test', default=False, action='store_true')
  args = parser.parse_args()
  
  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # make a list of datasets
  datasets = []
  with open(args.datasets, 'r') as f:
    datasets = json.load(f)

  # make a list of monitoring elements
  menames = []
  with open(args.menames, 'r') as f:
    menames = json.load(f)

  # handle splitting per dataset
  datasetfiles = [args.datasets]
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

  # loop over datasets and monitoring elements
  cmds = []
  for datasetfile in datasetfiles:
    for mefile in mefiles:
      cmd = 'python3 get_data_dials.py'
      cmd += ' -d {}'.format(datasetfile)
      cmd += ' -m {}'.format(mefile)
      cmd += ' -o {}'.format(args.outputdir)
      #if args.test: cmd += ' --test'
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
