#!/usr/bin/env python

# Convert data from ROOT to parquet

# imports
import sys
import os
import argparse
import uproot
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Convert data')
  parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
  parser.add_argument('--runmode', default='local', choices=['local','condor'])
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # handle job submission if requested
  if args.runmode=='condor':
    cmd = 'python3 convert_data.py'
    cmd += ' -i {}'.format(' '.join(args.inputfiles))
    cmd += ' --runmode local'
    ct.submitCommandAsCondorJob('cjob_convert_data', cmd,
      cmssw_version=CMSSW, home='auto')
    sys.exit()

  # print starting tag (for job completion checking)
  sys.stderr.write('###starting###\n')
  sys.stderr.flush()

  # loop over input files
  for fidx,inputfile in enumerate(args.inputfiles):
    print('Now running on file {}/{} ({})'.format(fidx+1, len(args.inputfiles), inputfile))

    # initialize dict
    dfdict = {}

    # open input file
    print('Opening file...')
    with uproot.open(inputfile) as f:
      # get runs, lumis and histogram names
      keys = np.array(list(f.keys()))
      runs = []
      lumis = []
      hnames = []
      for key in keys:
        parts = key.split('_', 2)
        runs.append(int(parts[0].replace('run','')))
        lumis.append(int(parts[1].replace('ls','')))
        hname = parts[2]
        if ';' in hname: hname = hname.split(';')[0]
        hnames.append(hname)
      runs = np.array(runs).astype(int)
      lumis = np.array(lumis).astype(int)
      hnames = np.array(hnames)
      # do sorting
      print('Sorting lumisections...')
      ids = (runs*10000 + lumis).astype(int)
      sorted_inds = np.argsort(ids)
      keys = keys[sorted_inds]
      runs = runs[sorted_inds]
      lumis = lumis[sorted_inds]
      hnames = hnames[sorted_inds]
      # printouts
      print('Found {} keys, corresponding to {} lumisections for {} MEs.'.format(
        len(keys), len(set(list(ids))), len(set(list(hnames)))))
      # get histogram data
      # note: flatten appears to be needed for 2D histograms,
      #       as conversion to parquet does not work for multidim arrays;
      #       the inverse operation (np.reshape) must be called when reading the dataframe.
      print('Reading histograms...')
      sys.stdout.flush()
      hists = []
      for idx, key in enumerate(keys):
        key = keys[idx]
        count = idx+1
        if( (count<=100 and count%10==0)
            or (count<=1000 and count%100==0)
            or count%1000==0 ):
          print('Processed {} out of {} entries'.format(count, len(keys)))
          sys.stdout.flush()
        hists.append(f[key].values().astype(int).flatten())
      entries = np.array([np.sum(hist) for hist in hists], dtype=int)
      print('Reading meta-info...')
      sys.stdout.flush()
      # get metadata (assume the same for all histograms in input file!)
      h = f[keys[0]]
      xbins = h.axis('x').edges()
      nxbins = len(xbins)-1
      xmin = xbins[0]
      xmax = xbins[-1]
      if isinstance(h, uproot.behaviors.TH1.TH1):
        metype = 3
        nybins = 0
        ymin = 0
        ymax = 0
      elif isinstance(h, uproot.behaviors.TH2.TH2):
        metype = 6
        ybins = h.axis('y').edges()
        nybins = len(ybins)-1
        ymin = ybins[0]
        ymax = ybins[-1]
      else:
        msg = 'ERROR: histogram type {} not recognized.'.format(type(h))
        raise Exception(msg)
      # print meta-info
      doprint = True
      if doprint:
        print('Found following meta info:')
        print('  metype: {}'.format(metype))
        print('  nxbins: {}'.format(nxbins))
        print('  xmin: {}'.format(xmin))
        print('  xmax: {}'.format(xmax))
        print('  nybins: {}'.format(nybins))
        print('  ymin: {}'.format(ymin))
        print('  ymax: {}'.format(ymax))
        sys.stdout.flush()
      # append all info
      dfdict['fromrun'] = np.array(runs, dtype=int)
      dfdict['fromlumi'] = np.array(lumis, dtype=int)
      dfdict['hname'] = np.array(hnames)
      dfdict['metype'] = (np.ones(len(lumis))*metype).astype(int)
      dfdict['histo'] = hists
      dfdict['entries'] = entries
      dfdict['Xmin'] = np.ones(len(lumis))*xmin
      dfdict['Xmax'] = np.ones(len(lumis))*xmax
      dfdict['Xbins'] = (np.ones(len(lumis))*nxbins).astype(int)
      dfdict['Ymin'] = np.ones(len(lumis))*ymin
      dfdict['Ymax'] = np.ones(len(lumis))*ymax
      dfdict['Ybins'] = (np.ones(len(lumis))*nybins).astype(int)
      
    # make a dataframe
    print('Converting to DataFrame...')
    df = pd.DataFrame(dfdict)

    # write output file
    print('Writing output file...')
    outputfile = inputfile.replace('.root', '.parquet')
    df.to_parquet(outputfile)

  # print finishing tag (for job completion checking)
  sys.stderr.write('###done###\n')
  sys.stderr.flush()
