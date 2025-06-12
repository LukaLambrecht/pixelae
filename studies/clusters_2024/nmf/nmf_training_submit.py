import os
import sys

thisdir = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)

import jobsubmission.condortools as ct


if __name__=='__main__':

    # general settings
    eras = ([
        'B-v1',
        'C-v1',
        'D-v1',
        'E-v1',
        'E-v2',
        'F-v1',
        #'F-v1-part1',
        #'F-v1-part2',
        #'F-v1-part3',
        #'F-v1-part4',
        'G-v1',
        #'G-v1-part1',
        #'G-v1-part2',
        #'G-v1-part3',
        #'G-v1-part4',
        'H-v1',
        'I-v1',
        'I-v2',
    ])
    layers = [1, 2, 3, 4]
    outputdir = 'output_test'
    runmode = 'condor'

    # preprocessing settings
    global_normalization = 'avg'
    local_normalization = None
    min_entries = 0.5e6

    # NMF settings
    n_components = 5
    forget_factor = 1
    tol = 0
    max_no_improvement = 100
    max_iter = 1000
    alpha_H = 0.1

    # batch settings
    batch_size = 3000
    nbatches = 30
    max_epochs = 10

    # make output directories
    outputdir = os.path.abspath(outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # make commands
    cmds = []
    for era in eras:
        for layer in layers:
            outputfile = f'nmf_model_BPIX{layer}_{era}.pkl'
            outputfile = os.path.join(outputdir, outputfile)
            cmd = 'python3 nmf_training.py'
            cmd += f' --era {era}'
            cmd += f' --layer {layer}'
            cmd += f' --outputfile {outputfile}'
            if global_normalization is not None:
                cmd += f' --preprocessing_global_normalization {global_normalization}'
            if local_normalization is not None:
                cmd += f' --preprocessing_local_normalization {local_normalization}'
            cmd += f' --min_entries {min_entries}'
            cmd += f' --n_components {n_components}'
            cmd += f' --forget_factor {forget_factor}'
            cmd += f' --tol {tol}'
            cmd += f' --max_no_improvement {max_no_improvement}'
            cmd += f' --max_iter {max_iter}'
            cmd += f' --alpha_H {alpha_H}'
            cmd += f' --batch_size {batch_size}'
            cmd += f' --nbatches {nbatches}'
            cmd += f' --max_epochs {max_epochs}'
            cmds.append(cmd)

    # run commands
    if runmode=='local':
        for cmd in cmds: os.system(cmd)

    # submit jobs
    elif runmode=='condor':
        cmssw = '/afs/cern.ch/user/l/llambrec/CMSSW_14_1_X_combine/CMSSW_14_1_0_pre4'
        jobdir = '/afs/cern.ch/user/l/llambrec/pixelae_job_log'
        # (must be on /afs as submission from /eos is not allowed)
        if not os.path.exists(jobdir): os.makedirs(jobdir)
        name = os.path.join(jobdir, 'cjob_nmf_training')
        ct.submitCommandsAsCondorCluster(name, cmds,
          cmssw_version=cmssw, jobflavour='workday')
