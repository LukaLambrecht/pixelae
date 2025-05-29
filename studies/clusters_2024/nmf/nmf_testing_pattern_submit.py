import os
import sys
import argparse

thisdir = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)

import jobsubmission.condortools as ct


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, nargs='+')
    parser.add_argument('-r', '--runmode', choices=['local', 'condor'], default='condor')
    args = parser.parse_args()

    # make commands
    cmds = []
    for config in args.config:
        cmd = 'python3 nmf_testing_pattern.py'
        cmd += f' {config}'
        cmds.append(cmd)

    # run commands
    if args.runmode=='local':
        for cmd in cmds: os.system(cmd)

    # submit jobs
    elif args.runmode=='condor':
        cmssw = '/afs/cern.ch/user/l/llambrec/CMSSW_14_1_X_combine/CMSSW_14_1_0_pre4'
        jobdir = '/afs/cern.ch/user/l/llambrec/pixelae_job_log'
        # (must be on /afs as submission from /eos is not allowed)
        if not os.path.exists(jobdir): os.makedirs(jobdir)
        name = os.path.join(jobdir, 'cjob_nmf_testing')
        ct.submitCommandsAsCondorCluster(name, cmds,
          cmssw_version=cmssw, jobflavour='workday')
