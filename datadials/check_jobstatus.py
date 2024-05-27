#!/usr/bin/env python3


# Check status and progress of jobs


# general imports
import os
import sys
import argparse
from termcolor import colored

# local imports
sys.path.append('../jobsubmission')
from jobcheck import check_start_done, check_error_content

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Check job status')
  parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
  args = parser.parse_args()
  
  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # define colors for printing status
  colordict = {
    'running': 'blue',
    'finished': 'green',
    'error': 'red',
    'unknown': 'gray'
  }
  
  # check file names
  inputfiles = []
  for inputfile in args.inputfiles:
    if not (inputfile.startswith('cjob_') and '_out_' in inputfile):
      msg = 'WARNING: input file {}'.format(inputfile)
      msg += ' does not seem to be a job output log file, skipping.'
      print(msg)
      continue
    inputfiles.append(inputfile)
  inputfiles = sorted(inputfiles)

  # read output log files
  progressdict = {}
  for inputfile in inputfiles:
    # read all lines
    with open(inputfile) as f:
      lines = f.readlines()
    # find current ME
    metag = 'Now running on ME'
    melines = [l for l in lines if l.startswith(metag)]
    if len(melines)==0: meinfo = 'unknown'
    else:
      meinfo = melines[-1].strip(' \t\n').replace(metag,'')
      meinfo = meinfo.split('(')[1].strip(')')
    # find current dataset
    datatag = 'Now running on dataset'
    datalines = [l for l in lines if l.startswith(datatag)]
    if len(datalines)==0: datainfo = 'unknown'
    else:
      datainfo = datalines[-1].strip(' \t\n').replace(datatag,'')
      datainfo = datainfo.split('(')[1].strip(')')
    # get job progress from last line
    lastline = lines[-1]
    lastline = lastline.strip(' \t\n')
    progress = 0.
    if lastline=='Writing output file(s)...': progress = 1.
    elif lastline.startswith('- Run'):
      progress = lastline.split('(')[1].strip(')')
      progress = progress.split('/')
      progress = float(progress[0])/float(progress[1])
    # group info into a dictionary
    thisprogressdict = {
      'progress': progress,
      'meinfo': meinfo,
      'datasetinfo': datainfo
    }
    progressdict[inputfile] = thisprogressdict

  # read error log files
  for inputfile in inputfiles:
    # find corresponding error log file
    errfile = inputfile.replace('_out_','_err_')
    if not os.path.exists(errfile):
      msg = 'WARNING: error log file {}'.format(errfile)
      msg += ' corresponding to output log file {}'.format(inputfile)
      msg += ' does not seem to exist, skipping.'
      progressdict[inputfile]['status'] = 'unknown'
    # get job status from error file
    status = 'finished'
    if check_start_done(errfile, verbose=False)==1: status = 'running'
    if check_error_content(errfile, verbose=False)==1: status = 'error'
    # add to progress dictionary
    progressdict[inputfile]['status'] = status

  # print progress
  for inputfile, info in progressdict.items():
    print('{}: {:.1f}% (dataset: {}, ME: {}) {}'.format(inputfile,
          info['progress']*100,
          info['datasetinfo'],
          info['meinfo'],
          colored(info['status'], colordict[info['status']]) ))
