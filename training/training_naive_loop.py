#################################################
# Run training for a set of monitoring elements #
#################################################


# imports
# external modules
import sys
import os
import argparse
# ML4DQM modules
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE/jobsubmission'))
import condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')


if __name__=='__main__':

  # define monitoring elements
  mes = ([
    'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_+1',
    'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_+2',
    'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_+3',
    'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_-1',
    'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_-2',
    'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_-3',
  ])

  # define eras
  eras = ({
    'Run2023C-v1': ['Run2023C-PromptReco-v1'],
    #'Run2023C-v2': ['Run2023C-PromptReco-v2'],
    #'Run2023C-v3': ['Run2023C-PromptReco-v3'],
    #'Run2023C-v4': ['Run2023C-PromptReco-v4'],
    #'Run2023D-v1': ['Run2023D-PromptReco-v1'],
    #'Run2023D-v2': ['Run2023D-PromptReco-v2'],
    #'Run2023E-v1': ['Run2023E-PromptReco-v1'], # low occupancy
    #'Run2023F-v1': ['Run2023F-PromptReco-v1'], # low occupancy
    #'Run2023': [
    #    'Run2023C-PromptReco-v1',
    #    'Run2023C-PromptReco-v2',
    #    'Run2023C-PromptReco-v3',
    #    'Run2023C-PromptReco-v4',
    #    'Run2023D-PromptReco-v1',
    #    'Run2023D-PromptReco-v2',
    #]
  })

  # define input and output files
  datadir = '/pnfs/iihe/cms/store/user/llambrec/dqmio/'
  inputfiles = {}
  outputfiles = {}
  for me in mes:
    for eraname, eratags in eras.items():
      mefiles = ['ZeroBias-{}-DQMIO-{}_preprocessed.parquet'.format(eratag, me) for eratag in eratags]
      modelfile = 'model_20231123_{}_{}.keras'.format(eraname, me)
      mefiles = [os.path.join(datadir, f) for f in mefiles]
      inputfiles[me+'_'+eraname] = mefiles
      outputfiles[me+'_'+eraname] = modelfile

  # define other settings
  settings = ({
    'entries_threshold': 10000,
    'skip_first_lumisections': 5,
    'veto_patterns': 1,
    'loss': 'mse',
    'optimizer': 'adam',
    'batch_size': 32,
    'epochs': 30,
    'validation_split': 0.1,
    'store_average_occupancy': 1,
    'store_average_response': 1
  })

  # make commands
  cmds = []
  for eraname in eras.keys():
    cmds.append([])
    for me in mes:
      cmd = 'python3 training_naive.py'
      cmd += ' -i {}'.format(' '.join(inputfiles[me+'_'+eraname]))
      cmd += ' -o {}'.format(outputfiles[me+'_'+eraname])
      cmd += ' --runmode local'
      for arg, val in settings.items():
        cmd += ' --{} {}'.format(arg, val)
      cmds[-1].append(cmd)

  # submit jobs
  for cmdset in cmds:
    ct.submitCommandsAsCondorCluster('cjob_training_naive', cmdset,
        cmssw_version=CMSSW, home='auto')
