##################################
# Run training in job submission #
##################################


# imports
# external modules
import sys
import os
import argparse
import numpy as np
import pandas as pd
# ML4DQM modules
sys.path.append(os.path.abspath('../../ML4DQMDC-PixelAE/jobsubmission'))
import condortools as ct
CMSSW = os.path.abspath('../../CMSSW_12_4_6')
# framework modules
sys.path.append('../')
from models.modeldefs import model_dummy
from models.modeldefs import model_ecal_endcap
# local modules
from prepare_training_set import prepare_training_data_from_files


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Train model')
  parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
  parser.add_argument('-o', '--outputfile', required=True)
  parser.add_argument('--entries_threshold', default=0, type=int)
  parser.add_argument('--skip_first_lumisections', default=0, type=int)
  parser.add_argument('--veto_patterns', default=0, type=int)
  parser.add_argument('--loss', default='mse')
  parser.add_argument('--optimizer', default='adam')
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--epochs', default=5, type=int)
  parser.add_argument('--validation_split', default=0.1, type=float)
  parser.add_argument('--store_average_occupancy', default=0, type=int)
  parser.add_argument('--store_average_response', default=0, type=int)
  parser.add_argument('--runmode', default='local', choices=['local', 'condor'])
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # handle job submission if requested
  if args.runmode=='condor':
    cmd = 'python3 training_naive.py'
    cmd += ' -i {}'.format(' '.join(args.inputfiles))
    cmd += ' -o {}'.format(args.outputfile)
    cmd += ' --runmode local'
    for arg in vars(args):
      if arg in ['inputfiles', 'outputfile', 'runmode']: continue
      cmd += ' --{} {}'.format(arg, getattr(args, arg))
    ct.submitCommandAsCondorJob('cjob_training_naive', cmd,
      cmssw_version=CMSSW, home='auto')
    sys.exit()

  # print starting tag (for job completion checking)
  sys.stderr.write('###starting###\n')
  sys.stderr.flush()

  # load the training set
  kwargs = ({
    'verbose': True,
    'entries_threshold': args.entries_threshold,
    'skip_first_lumisections': args.skip_first_lumisections
  })
  if args.veto_patterns > 0: kwargs['veto_patterns'] = [np.zeros((2,2)), np.zeros((3,1)), np.zeros((1,3))]
  (training_data, training_runs, training_lumis) = prepare_training_data_from_files(args.inputfiles, **kwargs)
  
  # make a mask where values are often zero
  shape_mask = (np.sum(training_data[:,:,:,0]==0, axis=0)>len(training_data)/2.)

  # initialize model
  input_shape = training_data.shape[1:]
  model = model_dummy(input_shape)
  model.compile(loss=args.loss, optimizer=args.optimizer)

  # do training
  history = model.fit(
    training_data, training_data,
    batch_size=args.batch_size,
    epochs=args.epochs,
    verbose=True,
    shuffle=True,
    validation_split=args.validation_split
  )

  # store the model
  model.save(args.outputfile)

  # store average occupancy
  if args.store_average_occupancy:
    outputfile = os.path.splitext(args.outputfile)[0] + '_avgoccupancy.npy'
    avgoccupancy = np.mean(training_data, axis=0)[:,:,0]
    np.save(outputfile, avgoccupancy)

  # store average error
  if args.store_average_response>0:
    # evaluate model on training set and calculate average response
    predictions = model.predict(training_data)
    errors = np.square(predictions - training_data)
    avgresponse = np.mean(errors, axis=0)[:,:,0]
    # store average response
    outputfile = os.path.splitext(args.outputfile)[0] + '_avgresponse.npy'
    np.save(outputfile, avgresponse)

  # print done tag (for job completion checking)
  sys.stderr.write('###done###\n')
  sys.stderr.flush()
