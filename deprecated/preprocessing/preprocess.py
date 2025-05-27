################################################################
# Preprocess histograms in a parquet file and store the result #
################################################################

# imports

# external modules
import sys
import os
import pandas as pd
import numpy as np
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import argparse

# ML4DQM modules
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE'))
import jobsubmission.condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')

# local modules
from preprocessor import PreProcessor


def get_crop(mename):
  if 'clusterposition_zphi' in mename: return None
  elif 'clusterposition_xy' in mename: return (slice(20,180), slice(19,179))
  elif '_per_SignedModuleCoord_per_SignedLadderCoord_' in mename: return None
  elif '_per_SignedDiskCoord_per_SignedBladePanelCoord_' in mename: return None
  else:
    msg = 'ERROR: ME {} not recognized.'.format(mename)
    raise Exception(msg)

def get_anticrop(mename):
  if 'clusterposition_zphi' in mename: return None
  elif 'clusterposition_xy' in mename: return None
  elif '_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1' in mename: return (slice(32,40), slice(12,14))
  elif '_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2' in mename: return (slice(32,40), slice(28,30))
  elif '_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3' in mename: return (slice(32,40), slice(44,46))
  elif '_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4' in mename: return (slice(32,40), slice(64,66))
  elif '_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1' in mename: return (slice(24,32), slice(44,48))
  elif '_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2' in mename: return (slice(24,32), slice(68,72))
  else:
    msg = 'ERROR: ME {} not recognized.'.format(mename)
    raise Exception(msg)


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Preprocess data')
  parser.add_argument('-i', '--inputfile', required=True, type=os.path.abspath)
  parser.add_argument('-o', '--outputfile', required=True, type=os.path.abspath)
  parser.add_argument('-c', '--config', default=None, type=os.path.abspath)
  parser.add_argument('-n', '--nentries', default=-1, type=int)
  parser.add_argument('--omsfile', default=None, type=os.path.abspath)
  parser.add_argument('--runmode', default='local', choices=['local','condor'])
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # handle job submission if requested
  if args.runmode=='condor':
    cmd = 'python3 preprocess.py'
    cmd += ' -i {}'.format(args.inputfile)
    cmd += ' -o {}'.format(args.outputfile)
    if args.config is not None: cmd += ' -c {}'.format(args.config)
    if args.nentries > 0: cmd += ' -n {}'.format(args.nentries)
    cmd += ' --runmode local'
    ct.submitCommandAsCondorJob('cjob_preprocess', cmd,
      cmssw_version=CMSSW, home='auto')
    sys.exit()

  # print starting tag (for job completion checking)
  sys.stderr.write('###starting###\n')
  sys.stderr.flush()

  # define a PreProcessor
  prep = PreProcessor(
    crop = get_crop(args.inputfile),
    anticrop = get_anticrop(args.inputfile),
    #time_average_radii={0:2, 50:5},
    #rebin_target=(20,20),
    omsjson=args.omsfile,
    oms_normalization_attr='pileup'
  )

  # load the input file
  if args.nentries <= 0:
    df = pd.read_parquet(args.inputfile)
  else:
    pf = ParquetFile(args.inputfile)
    batch = next(pf.iter_batches(batch_size = args.nentries))
    df = pa.Table.from_batches([batch]).to_pandas()
  nentries = len(df)

  # extract and reshape the histograms
  xbins = df['Xbins'][0]
  ybins = df['Ybins'][0]
  hists = np.array([df['histo'][i].reshape(xbins,ybins) for i in range(nentries)])
  nhists = len(hists)
  runs = np.array(df['fromrun'])
  lumis = np.array(df['fromlumi'])

  # do preprocessing
  preprocessed_hists = prep.preprocess(hists, runs=runs, lumis=lumis)
  xbins = preprocessed_hists.shape[1]
  ybins = preprocessed_hists.shape[2]

  # replace histograms in dataframe
  preprocessed_hists = [preprocessed_hists[i,:,:].flatten() for i in range(nhists)]
  df['histo'] = preprocessed_hists

  # replace meta info to the dataframe
  df['Xbins'] = (np.ones(len(lumis))*xbins).astype(int)
  df['Ybins'] = (np.ones(len(lumis))*ybins).astype(int)

  # write output file
  df.to_parquet(args.outputfile)

  # write preprocessor config file
  if args.config is not None:
    with open(args.config, 'w') as f:
      f.write(prep.__str__())

  # print finishing tag (for job completion checking)
  sys.stderr.write('###done###\n')
  sys.stderr.flush()
