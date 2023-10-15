#!/usr/bin/env python3

# Get data from DAS

# imports
import sys
import os
import json
import numpy as np
import argparse
import ROOT
# local imports
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
from dqmio.src.DQMIOReader import DQMIOReader
import dqmio.src.tools as dqmiotools
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Get data')
  parser.add_argument('-d', '--datasetname', required=True)
  parser.add_argument('-m', '--menames', required=True)
  parser.add_argument('-p', '--proxy', default=None)
  parser.add_argument('-r', '--redirector', default='root://cms-xrd-global.cern.ch/')
  parser.add_argument('-o', '--outputdir', default='.')
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'])
  parser.add_argument('--test', default=False, action='store_true')
  args = parser.parse_args()
  args.proxy = os.path.abspath(args.proxy) if args.proxy is not None else None
  
  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # handle job submission if requested
  if args.runmode=='condor':
    cmd = 'python3 get_data.py'
    cmd += ' -d {}'.format(args.datasetname)
    cmd += ' -m {}'.format(args.menames)
    if args.proxy is not None: cmd += ' -p {}'.format(args.proxy)
    cmd += ' -r {}'.format(args.redirector)
    cmd += ' -o {}'.format(args.outputdir)
    if args.test: cmd += ' --test'
    cmd += ' --runmode local'
    ct.submitCommandAsCondorJob('cjob_get_data', cmd,
      cmssw_version=CMSSW, proxy=args.proxy, home='auto')
    sys.exit()

  # export the proxy
  if args.proxy is not None: dqmiotools.export_proxy( args.proxy )

  # make a list of input files
  filemode = 'das'
  if os.path.exists(args.datasetname): filemode = 'local'
  inputfiles = dqmiotools.format_input_files( 
                 args.datasetname,
                 filemode=filemode,
                 redirector=args.redirector,
                 istest=args.test )

  # make a list of monitoring elements
  menames = []
  with open(args.menames, 'r') as f:
    menames = json.load(f)

  # print configuration parameters
  print('Running with following parameters:')
  print('Input files ({}):'.format(len(inputfiles)))
  for inputfile in inputfiles: print('  - {}'.format(inputfile))
  print('Monitoring elements: ({})'.format(len(menames)))
  for mename in menames: print('  - {}'.format(mename))

  # make a DQMIOReader instance and initialize it with the DAS files
  print('Initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*inputfiles, sortindex=True)
  print('Initialized DQMIOReader with following properties')
  print('Number of lumisections: {}'.format(len(reader.listLumis())))
  print('Number of monitoring elements per lumisection: {}'.format(len(reader.listMEs())))

  # loop over monitoring elements
  for mename in menames:

    # select the monitoring element
    print('Selecting monitoring element {}...'.format(mename))
    mes = reader.getSingleMEs(mename)

    # write selected monitoring elements to output file
    print('Writing output file...')
    outputfile = (args.datasetname+'-'+mename).strip('/').replace('/','-')+'.root'
    outputfile = os.path.join(args.outputdir, outputfile)
    f = ROOT.TFile.Open(outputfile, 'recreate')
    for me in mes:
      name = 'run{}_ls{}_{}'.format(me.run, me.lumi, me.name.replace('/','_'))
      me.data.SetName(name)
      me.data.SetTitle(name)
      me.data.Write()
    f.Close()
