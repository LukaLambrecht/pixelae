#!/usr/bin/env python3

# Get data from DAS
# Looper over datasets

# imports
import sys
import os
import json
import argparse
# local imports
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Get data')
  parser.add_argument('-d', '--datasets', required=True)
  parser.add_argument('-m', '--menames', required=True)
  parser.add_argument('-p', '--proxy', default=None)
  parser.add_argument('-r', '--redirector', default='root://cms-xrd-global.cern.ch/')
  parser.add_argument('-o', '--outputdir', default='.')
  parser.add_argument('--splitmes', default=False, action='store_true')
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'])
  parser.add_argument('--test', default=False, action='store_true')
  args = parser.parse_args()
  args.proxy = os.path.abspath(args.proxy) if args.proxy is not None else None
  
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

  # handle splitting per monitoring element
  mefiles = [args.menames]
  if args.splitmes:
    mefiles = []
    for i,mename in enumerate(menames):
      mefile = 'temp_menames_{}.json'.format(i)
      mefiles.append(mefile)
      with open(mefile, 'w') as f:
        json.dump([mename], f)

  # loop over datasets (and optionally monitoring elements)
  cmds = []
  for dataset in datasets:
    for mefile in mefiles:
      cmd = 'python3 get_data.py'
      cmd += ' -d {}'.format(dataset)
      cmd += ' -m {}'.format(mefile)
      if args.proxy is not None: cmd += ' -p {}'.format(args.proxy)
      cmd += ' -r {}'.format(args.redirector)
      cmd += ' -o {}'.format(args.outputdir)
      if args.test: cmd += ' --test'
      cmd += ' --runmode local'
      cmds.append(cmd)

  # limit for testing
  if args.test: cmds = [cmds[0]]

  # run or submit commands
  if args.runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_get_data', cmds,
      cmssw_version=CMSSW, proxy=args.proxy, home='auto')
  else:
    for cmd in cmds:
      print(cmd)
      os.system(cmd)
