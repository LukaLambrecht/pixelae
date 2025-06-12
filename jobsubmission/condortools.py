###################################################################################
# functionality for making job description files for the condor submission system #
###################################################################################

# general use:
# two ingredients are needed for a condor job:
# - a job description (.txt) file
# - an executable
# the functions in this tool allow creating an executable bash script
# and its submission via a job description file

import os
import sys

def makeUnique(fname):
    ### make a file name unique by appending a number to it,
    # e.g. test.txt -> test1.txt (in case test.txt already exists)
    # todo: works for now, but find cleaner way to submit jobs in a loop...
    if not os.path.exists(fname): return fname
    [name,ext] = os.path.splitext(fname)
    app = 1
    while app < 2500:
        tryname = name+str(app)+ext
        if not os.path.exists(tryname): return tryname
        app += 1
    msg = 'ERROR ###: already 2500 files named {} exist.'.format(fname)
    msg += ' consider choosing more specific names, splitting in folders, etc.'
    raise Exception(msg)

def initJobScript(name, 
                  home=None,
                  workdir=None,
                  cmssw_version=None,
                  proxy=None,
                  conda_activate=None,
                  conda_env=None):
    ### initialize an executable bash script with:
    # - setting HOME environment variable
    #   note: use 'auto' to extract the HOME from os.environ
    # - sourcing the shared cms software
    # - setting correct cmssw release
    # - exporting proxy
    
    # parse filename
    name = os.path.splitext(name)[0]
    fname = name+'.sh'
    if os.path.exists(fname): os.system('rm {}'.format(fname))
    if workdir is None: workdir = os.path.abspath(os.getcwd())
    # parse home
    if home=='auto': home = os.environ['HOME']
    # write script
    with open(fname,'w') as script:
	    # write bash shebang
        script.write('#!/bin/bash\n')
	    # write echo script name
        script.write("echo '###exename###: {}'\n".format(fname))
	    # write export home
        if home is not None:
            script.write('export HOME={}\n'.format(home))
	    # write sourcing of common software
        script.write('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
	    # write setting correct cmssw release
        if cmssw_version is not None:
            script.write('cd {}\n'.format( os.path.join( cmssw_version,'src' ) ) )
            script.write('eval `scram runtime -sh`\n')
	    # write export proxy
        if proxy is not None:
            script.write('export X509_USER_PROXY={}\n'.format( proxy ))
        # write conda activate
        if conda_activate is not None:
            script.write(conda_activate+'\n')
        if conda_env is not None:
            script.write('conda activate {}\n'.format(conda_env))
        script.write('cd {}\n'.format( workdir ) )
    # make executable
    os.system('chmod +x '+fname)
    print('initJobScript created {}'.format(fname))

def makeJobDescription(name, exe, argstrings=None, 
                       stdout=None, stderr=None, log=None,
                       cpus=1, mem=1024, disk=10240, 
                       proxy=None, jobflavour=None):
    ### create a single job description txt file
    # note: exe can for example be a runnable bash script
    # note: argstrings is a single string containing the arguments to exe (space-separated)
    #       update: now it can also be a list of such strings for multiple parallel
    #               submissions of the same exe with different arguments.
    #               this update follows the announcement of the deprecation
    #               of multiple queue statements in favour of a new syntax.
    # note: for job flavour: see here: https://batchdocs.web.cern.ch/local/submit.html
    
    # parse arguments
    name = os.path.splitext(name)[0]
    fname = name+'.txt'
    if os.path.exists(fname): os.system('rm {}'.format(fname))
    if stdout is None: stdout = name+'_out_$(ClusterId)_$(ProcId)'
    if stderr is None: stderr = name+'_err_$(ClusterId)_$(ProcId)'
    if log is None: log = name+'_log_$(ClusterId)_$(ProcId)'
    # write file
    with open(fname,'w') as f:
        f.write('executable = {}\n'.format(exe))
        f.write('output = {}\n'.format(stdout))
        f.write('error = {}\n'.format(stderr))
        f.write('log = {}\n\n'.format(log))
        f.write('request_cpus = {}\n'.format(cpus))
        f.write('request_memory = {}\n'.format(mem))
        f.write('request_disk = {}\n'.format(disk))
        if proxy is not None: 
            f.write('x509userproxy = {}\n'.format(proxy))
            f.write('use_x509userproxy = true\n\n')
        #f.write('should_transfer_files = yes\n\n') 
        # (not fully sure whether to put 'yes', 'no' or omit it completely)
        if jobflavour is not None:
            f.write('+JobFlavour = "{}"\n\n'.format(jobflavour))
        # make queue statement
        if argstrings is None: qcmd = 'queue'
        else:
            if isinstance(argstrings, str): argstrings = [argstrings]
            qcmd = 'queue arguments from (\n'
            for argstring in argstrings:
                qcmd += '    "{}"\n'.format(argstring)
            qcmd += ')'
        f.write(qcmd + '\n')
    print('makeJobDescription created {}'.format(fname))

def submitCondorJob(jobDescription):
    ### submit a job description file as a condor job
    
    # check if file exists
    fname = os.path.splitext(jobDescription)[0]+'.txt'
    if not os.path.exists(fname):
        print('ERROR: job description file {} not found'.format(fname))
        sys.exit()

    # check if need to move to another directory
    jddir, fname = os.path.split(fname)
    if len(jddir) > 0: cmd = f'cd {jddir}; condor_submit {fname}'
    else: cmd = f'condor_submit {fname}'

    # run the condor_submit command
    os.system(cmd)

def submitCommandAsCondorJob(name, command, **kwargs):
    ### submit a single command as a single job
    # command is a string representing a single command (executable + args)
    submitCommandsAsCondorJobs(name, [[command]], **kwargs)

def submitCommandsAsCondorCluster(name, commands, stdout=None, stderr=None, log=None,
                        cpus=1, mem=1024, disk=10240,
                        home=None,
                        workdir=None,
                        proxy=None,
                        cmssw_version=None,
                        conda_activate=None,
                        conda_env=None,
                        jobflavour=None):
    ### run several similar commands within a single cluster of jobs
    # note: each command must have the same executable and number of args, only args can differ!
    # note: commands can be a list of commands (-> a job will be submitted for each command)
    
    # parse arguments
    name = os.path.splitext(name)[0]
    shname = makeUnique(name+'.sh')
    jdname = name+'.txt'
    exes = [command.split(' ', 1)[0] for command in commands]
    exe = exes[0]
    for test_exe in exes[1:]:
        if test_exe != exe:
            msg = 'The executable (i.e. the first element of each command)'
            msg += ' must be the same for each of the provided commands.'
            raise Exception(msg)
    argstrings = [command.split(' ', 1)[1] for command in commands]
    nargs = len(argstrings[0].split(' '))
    for test_argstring in argstrings[1:]:
        if len(test_argstring.split(' ')) != nargs:
            msg = 'The number of arguments to the executable'
            msg += ' must be the same for each of the provided commands.'
            raise Exception(msg)

    # make the job script
    initJobScript(shname, home=home, workdir=workdir, cmssw_version=cmssw_version, proxy=proxy,
      conda_activate=conda_activate, conda_env=conda_env)
    with open(shname,'a') as script:
        script.write(exe)
        script.write(' $@')
        script.write('\n')

    # make the job description
    makeJobDescription(name, shname, argstrings=argstrings,
                       stdout=stdout, stderr=stderr, log=log,
                       cpus=cpus, mem=mem, disk=disk, proxy=proxy,
                       jobflavour=jobflavour)
    
    # finally submit the job cluster
    submitCondorJob(jdname)

def submitCommandsAsCondorJob(name, commands, **kwargs):
    ### submit a set of commands as a single job
    # commands is a list of strings, each string represents a single command (executable + args)
    # the commands can be anything and are not necessarily same executable or same number of args.
    submitCommandsAsCondorJobs(name, [commands], **kwargs)

def submitCommandsAsCondorJobs(name, commands, stdout=None, stderr=None, log=None,
            cpus=1, mem=1024, disk=10240,
            home=None,
            workdir=None,
            proxy=None,
            cmssw_version=None,
            conda_activate=None,
            conda_env=None,
            jobflavour=None):
    ### submit multiple sets of commands as jobs (one job per set)
    # commands is a list of lists of strings, each string represents a single command
    # the commands can be anything and are not necessarily same executable or number of args.
    for commandset in commands:
        # parse arguments
        name = os.path.splitext(name)[0]
        shname = makeUnique(name+'.sh')
        jdname = name+'.txt'
        # first make the executable
        initJobScript(shname, home=home, workdir=workdir, cmssw_version=cmssw_version, proxy=proxy,
          conda_activate=conda_activate, conda_env=conda_env)
        with open(shname,'a') as script:
             for cmd in commandset: script.write(cmd+'\n')
        # then make the job description
        makeJobDescription(name,shname,stdout=stdout,stderr=stderr,log=log,
                            cpus=cpus,mem=mem,disk=disk,proxy=proxy,
                            jobflavour=jobflavour)
        # finally submit the job
        submitCondorJob(jdname)
