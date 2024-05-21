###################
# Preprocess data #
###################
# Looper over preprocess.py

# Note: crashes for some files with unclear error message,
# most likely too little RAM for the unpacked dataframe,
# try jobs with larger RAM request (but not urgent for now)

# imports
import sys
import os
from fnmatch import fnmatch
import argparse
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')


def get_oms_file(parquetfile):
  if fnmatch(parquetfile, '*Run2023C-*-v1-*'): return 'omsdata/omsdata_Run2023C-v1.json'
  if fnmatch(parquetfile, '*Run2023C-*-v2-*'): return 'omsdata/omsdata_Run2023C-v2.json'
  if fnmatch(parquetfile, '*Run2023C-*-v3-*'): return 'omsdata/omsdata_Run2023C-v3.json'
  if fnmatch(parquetfile, '*Run2023C-*-v4-*'): return 'omsdata/omsdata_Run2023C-v4.json'
  if fnmatch(parquetfile, '*Run2023D-*-v1-*'): return 'omsdata/omsdata_Run2023D-v1.json'
  if fnmatch(parquetfile, '*Run2023D-*-v2-*'): return 'omsdata/omsdata_Run2023D-v2.json'
  if fnmatch(parquetfile, '*Run2023E-*-v1-*'): return 'omsdata/omsdata_Run2023E.json'
  if fnmatch(parquetfile, '*Run2023F-*-v1-*'): return 'omsdata/omsdata_Run2023F.json'
  if fnmatch(parquetfile, '*Run2024B-*-v1-*'): return 'omsdata/omsdata_Run2024B-v1.json'
  if fnmatch(parquetfile, '*Run2024C-*-v1-*'): return 'omsdata/omsdata_Run2024C-v1.json'
  if fnmatch(parquetfile, '*Run2024D-*-v1-*'): return 'omsdata/omsdata_Run2024D-v1.json'  
  raise Exception('No suitable OMS file found for input file {}'.format(parquetfile))


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Preprocess data')
  parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
  parser.add_argument('--runmode', default='local', choices=['local','condor'])
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # loop over input files
  cmds = []
  for inputfile in args.inputfiles:
    outputfile = inputfile.replace('.parquet', '_preprocessed.parquet')
    configfile = inputfile.replace('.parquet', '_prepconfig.txt')
    omsfile = get_oms_file(inputfile)
    cmd = 'python3 preprocess.py'
    cmd += ' -i {}'.format(inputfile)
    cmd += ' -o {}'.format(outputfile)
    cmd += ' -c {}'.format(configfile)
    cmd += ' --omsfile {}'.format(omsfile)
    cmd += ' --runmode local'
    cmds.append(cmd)

  # run or submit commands
  if args.runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_preprocess', cmds,
      cmssw_version=CMSSW, home='auto')
  else:
    for cmd in cmds:
      print(cmd)
      os.system(cmd) 
