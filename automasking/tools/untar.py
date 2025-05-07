#!/usr/bin/env python3


import os
import sys
import six
import glob
import subprocess


if __name__=='__main__':

    # read input directory from command line
    inputdir = os.path.abspath(sys.argv[1])

    # find all .tar.gz files recursively
    pattern = os.path.join(inputdir, '**/*.tar.xz')
    tarfiles = glob.glob(pattern, recursive=True)
    print(f'Found {len(tarfiles)} tar files in {inputdir}.')

    # ask for confirmation
    print('Continue? (y/n)')
    go = six.moves.input()
    if not go=='y': sys.exit()

    # loop over tar files
    for idx, tarfile in enumerate(tarfiles):
        print(f'Untarring file {idx+1} / {len(tarfiles)}...', end='\r')

        # untar
        targetdir = os.path.dirname(tarfile)
        cmd = f'tar -xf {tarfile} -C {targetdir}'
        p = subprocess.run(cmd, shell=True, check=True)
        
        print(tarfile)
        sys.exit()
