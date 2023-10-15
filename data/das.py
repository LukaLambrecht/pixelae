#!/usr/bin/env python

# Find available datasets on DAS


# imports
import sys
import os
import argparse


# helper functions

def get_datasets(pd, era, tier):
  dascmd = "dasgoclient -query 'dataset=/{}/{}/{}' --limit 0".format(pd, era, tier)
  dasstdout = os.popen(dascmd).read()
  if 'X509_USER' in dasstdout:
    msg = 'ERROR: proxy does not seem to be set.'
    raise Exception(msg)
  datasets = dasstdout.strip(' \n\t').split('\n')
  return datasets


# main

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Find available datasets on DAS')
  parser.add_argument('-d', '--primary_dataset', default='ZeroBias')
  parser.add_argument('-e', '--era', default='Run2023*PromptReco*')
  parser.add_argument('-t', '--tier', default='DQMIO')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # get datasets
  datasets = get_datasets(args.primary_dataset, args.era, args.tier)

  # print results
  print('Found following datasets:')
  for dataset in datasets: print(dataset)
