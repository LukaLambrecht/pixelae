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


def get_modelfile(mename):
    if 'clusters_per_SignedModuleCoord_per_SignedLadderCoord' in mename:
        return os.path.abspath('../models/models/model_clusters_pxlayers_test.keras')
    elif 'clusters_per_SignedDiskCoord_per_SignedBladePanelCoord' in mename:
        return os.path.abspath('../models/models/model_clusters_pxrings_test.keras')
    else:
        msg = 'ERROR: could not find suitable model for ME {}.'.format(mename)
        raise Exception(msg)


if __name__=='__main__':

  # define monitoring elements
  mes = ([
    #'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_+1',
    #'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_+2',
    #'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_+3',
    #'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_-1',
    #'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_-2',
    #'PixelPhase1-Tracks-PXForward-clusterposition_xy_ontrack_PXDisk_-3',
    'PixelPhase1-Phase1_MechanicalView-PXBarrel-clusters_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1',
    'PixelPhase1-Phase1_MechanicalView-PXBarrel-clusters_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2',
    'PixelPhase1-Phase1_MechanicalView-PXBarrel-clusters_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3',
    'PixelPhase1-Phase1_MechanicalView-PXBarrel-clusters_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4',
    'PixelPhase1-Phase1_MechanicalView-PXForward-clusters_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1',
    'PixelPhase1-Phase1_MechanicalView-PXForward-clusters_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2'
  ])

  # define eras
  eras = ({
    #'Run2023C-v1': ['Run2023C-PromptReco-v1'],
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
    'Run2024B-v1': ['Run2024B-PromptReco-v1'],
    'Run2024C-v1': ['Run2024C-PromptReco-v1'],
    'Run2024D-v1': ['Run2024D-PromptReco-v1']
  })

  # define input files, model files, and output files
  datadir = '/pnfs/iihe/cms/store/user/llambrec/dqmio2024'
  inputfiles = {}
  outputfiles = {}
  modelfiles = {}
  for me in mes:
    for eraname, eratags in eras.items():
      # define input files
      mefiles = ['ZeroBias-{}-DQMIO-{}_preprocessed.parquet'.format(eratag, me) for eratag in eratags]
      mefiles = [os.path.join(datadir, f) for f in mefiles]
      # check if input files exist
      for mefile in mefiles:
        if not os.path.exists(mefile):
          msg = 'ERROR: input file {} does not exist.'.format(mefile)
          raise Exception(msg)
      # define model file
      modelfile = get_modelfile(me)
      # check if model file exists
      if not os.path.exists(modelfile):
        msg = 'ERROR: model file {} does not exist.'.format(modelfile)
        raise Exception(msg)
      # define output file
      outputfile = 'model_20240521_{}_{}.keras'.format(eraname, me)
      # add all info to dicts
      inputfiles[me+'_'+eraname] = mefiles
      outputfiles[me+'_'+eraname] = outputfile
      modelfiles[me+'_'+eraname] = modelfile

  # define other settings
  settings = ({
    'entries_threshold': 10000,
    #'skip_first_lumisections': 5,
    #'veto_patterns': 1,
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
      cmd += ' -m {}'.format(modelfiles[me+'_'+eraname])
      cmd += ' --runmode local'
      for arg, val in settings.items():
        cmd += ' --{} {}'.format(arg, val)
      cmds[-1].append(cmd)

  # submit jobs
  for cmdset in cmds:
    ct.submitCommandsAsCondorCluster('cjob_training_naive', cmdset,
        cmssw_version=CMSSW, home='auto')
