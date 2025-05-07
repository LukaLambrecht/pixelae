import os
import sys
import tarfile as tar
import numpy as np


def get_automask_from_txt(txt, tracker='BPix', layer=1):
    '''
    Get automask from a txt representation.
    Input arguments:
      - txt: list of strings, e.g. as obtained with file.readlines()
      - tracker: choose from "BPix" of "FPix" (latter not yet implemented)
      - layer: choose from 1-4 (for BPix), to implement for FPix
    Returns:
      - list of lists of the following form:
        [signed ladder coordinate, signed module coordinate, 
         first ROC number, last ROC number]
    '''

    # check tracker
    if tracker=='BPix': pass
    elif tracker=='FPix': raise Exception('Not yet implemented')
    else: raise Exception(f'Tracker system {tracker} not recognized.')

    # filter interesting lines in txt
    txt = [l for l in txt if l.startswith('FED ')]
    automasks = []
    for line in txt:
        fedpart, modulepart = line.split(' -> ')
        moduleparts = modulepart.split('_')
        
        # determine tracker system
        system = moduleparts[0]
        if system != tracker: continue
        
        # determine region
        region = moduleparts[1]
        modulesign = 1
        laddersign = 1
        if region.startswith('Bp'): pass
        elif region.startswith('Bm'): modulesign = -1
        else: raise Exception(f'Region {region} not recognized.')
        if region.endswith('I'): pass
        elif region.endswith('O'): laddersign = -1
        else: raise Exception(f'Region {region} not recognized.')
        
        # determine layer
        lyr = int(moduleparts[3].replace('LYR', ''))
        if lyr != layer: continue
        
        # determine ladder, module and ROCs
        ladder = laddersign * int(moduleparts[4].replace('LDR', '').strip('FH'))
        module = modulesign * int(moduleparts[5].replace('MOD', ''))
        firstroc, lastroc = moduleparts[6].split('[')[1].strip('\n\t ]').split(':')
        firstroc = int(firstroc)
        lastroc = int(lastroc)
        
        # save in data structure
        automasks.append([ladder, module, firstroc, lastroc])
   
    return automasks


def get_automask_from_tarfile(tarfile, ltp=None, **kwargs):
    '''
    Get automasks from a tar file.
    Input arguments:
      - tarfile: tar file holding automask txt files.
        note: must be in conventional format, consisting of the following:
              - the tar file is named automasked_yyyy-mm-dd.tar.xz
              - the txt files in it are named automasked_yyyy-mm-dd_hh:mm:ss_<some tag>.txt
              - for the content of the txt files, see examples.
      - ltp: a LumisectionTimeParser instance.
        if not provided, the timestamps are not converted to run/lumisection numbers.
      - kwargs: passed down to get_automask_from_txt
    Returns:
      - dict of the following form: {timestamp: list of masks}
        where the timestamp is "run_lumi" if ltp was provided or raw timestamp if not,
        and where each element in the list of masks is a list of the following form:
        [signed ladder coordinate, signed module coordinate,
         first ROC number, last ROC number]
    '''
    
    # open tar file
    with tar.open(tarfile) as f:
       
        try:
            f.getnames()
            f.getmembers()
        except:
            msg = f'WARNING: file {tarfile} could not be read; skipping.'
            print(msg)
            return
            
        # loop over contents
        res = {}
        for name, member in zip(f.getnames(), f.getmembers()):
            memberf = f.extractfile(member)
            content = memberf.readlines()
            
            # extract timestamp from filename
            nameparts = name.split('_')
            date = nameparts[1]
            time = nameparts[2]
            timestamp = int(date.replace('-','') + time.replace(':',''))
    
            # find corresponding lumisection
            if ltp is not None:
                try:
                    (run, lumi) = ltp.get_lumi(timestamp)
                    timestamp = str(run)+'_'+str(lumi)
                except: continue

            # read contents
            content = [l.decode('utf-8') for l in content]
            automasks = get_automask_from_txt(content, **kwargs)
            res[timestamp] = automasks
    return res


def get_automask_from_tarfiles(tarfiles, **kwargs):
    '''
    Looper over get_automask_from_tarfile for multiple files
    '''

    res = {}
    for tarfileidx, tarfile in enumerate(tarfiles):
        print(f'Reading file {tarfileidx+1} / {len(tarfiles)}', end='\r')
        this_res = get_automask_from_tarfile(tarfile, **kwargs)
        if this_res is not None: res.update(this_res)
    return res


def automask_to_map(automasks, tracker='BPix', layer=1):
    '''
    Convert automask to boolean np array
    '''

    # set number of ladders and modules
    if tracker!='BPix': raise Exception('Not yet implemented')
    if layer==1: (nladders, nmodules) = (6, 4)
    elif layer==2: (nladders, nmodules) = (14, 4)
    elif layer==3: (nladders, nmodules) = (22, 4)
    elif layer==4: (nladders, nmodules) = (32, 4)
    else: raise Exception(f'Layer {layer} not recognized.')

    # make base map
    nybins = 2*(2*nladders+1)
    nxbins = 8*(2*nmodules+1)
    automask_map = np.zeros((nybins, nxbins)).astype(bool)

    # define numbering of ROCs within a module
    roc_ycoord_extra = {}
    for roc in [4,5,6,7,12,13,14,15]: roc_ycoord_extra[roc] = 0
    for roc in [0,1,2,3,8,9,10,11]: roc_ycoord_extra[roc] = 1
    roc_xcoord_extra = {}
    for roc in [0, 4]: roc_xcoord_extra[roc] = 7
    for roc in [1, 5]: roc_xcoord_extra[roc] = 6
    for roc in [2, 6]: roc_xcoord_extra[roc] = 5
    for roc in [3, 7]: roc_xcoord_extra[roc] = 4
    for roc in [8, 12]: roc_xcoord_extra[roc] = 3
    for roc in [9, 13]: roc_xcoord_extra[roc] = 2
    for roc in [10, 14]: roc_xcoord_extra[roc] = 1
    for roc in [11, 15]: roc_xcoord_extra[roc] = 0
        
    # mask automasks
    for automask in automasks:
        ycoord_base = 2*(automask[0] + nladders)
        xcoord_base = 8*(automask[1] + nmodules)
        for roc in np.arange(automask[2], automask[3]+1):
            ycoord = ycoord_base + roc_ycoord_extra[roc]
            xcoord = xcoord_base + roc_xcoord_extra[roc]
            automask_map[ycoord, xcoord] = True

    return automask_map
