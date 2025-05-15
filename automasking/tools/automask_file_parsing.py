# external modules
import os
import sys
import tarfile as tar
import numpy as np
from fnmatch import fnmatch

# local modules
thisdir = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

from automasking.tools.roc_numbering import roc_to_coords, coords_to_roc
from automasking.tools.automask_operations import group_rocs


def get_automask_descriptions_from_txt(txt):
    '''
    Get the automask description strings from a typical automask txt file content.
    Input arguments:
    - txt: a list of strings (one per line) read from a typical automask txt file.
    Returns:
    - a list of strings of the form "BPix_BpO_SEC4_LYR2_LDR7F_MOD4_ROC[0:3]"
    '''
    
    # filter interesting lines in txt
    lines = [l for l in txt if fnmatch(l, 'FED * channel * -> *')]
    
    # remove unneeded part(s) in each line
    lines = [l.split(' -> ')[1].strip(' \t\n') for l in lines]
    
    return lines


def get_automask_from_description(automask_description):
    '''
    Get automask from a txt description.
    Input arguments:
    - automask_description: a string of the form "BPix_BpO_SEC4_LYR2_LDR7F_MOD4_ROC[0:3]"
    Returns:
    - a list of the form [layer, signed ladder coordinate, signed module coordinate, first ROC number, last ROC number]
    '''

    # split description in parts
    parts = automask_description.split('_')
        
    # determine tracker system
    system = parts[0]
    if system != "BPix":
        raise Exception('Not yet implemented')
        # (skip FPix for now, proper parsing to implement later)
        
    # determine region
    region = parts[1]
    modulesign = 1
    laddersign = 1
    if region.startswith('Bp'): pass
    elif region.startswith('Bm'): modulesign = -1
    else: raise Exception(f'Region {region} not recognized.')
    if region.endswith('I'): pass
    elif region.endswith('O'): laddersign = -1
    else: raise Exception(f'Region {region} not recognized.')
        
    # determine layer
    layer = int(parts[3].replace('LYR', ''))
        
    # determine ladder, module and ROCs
    ladder = laddersign * int(parts[4].replace('LDR', '').strip('FH'))
    module = modulesign * int(parts[5].replace('MOD', ''))
    firstroc, lastroc = parts[6].split('[')[1].strip('\n\t ]').split(':')
    firstroc = int(firstroc)
    lastroc = int(lastroc)
        
    # return the result
    return [layer, ladder, module, firstroc, lastroc]


def get_automask_from_txt(txt):
    '''
    Get automask from a txt representation.
    Input arguments:
      - txt: list of strings, e.g. as obtained with file.readlines()
    Returns:
      - a dictionary matching subsystem to masked channels.
        the keys are "BPix1", "BPix2", "BPix3" or "BPix4" (FPix: to implement).
        the masked channels are provided as a list of lists of the following form:
        [signed ladder coordinate, signed module coordinate, first ROC number, last ROC number]
    '''

    # filter interesting lines in txt
    automask_descriptions = get_automask_descriptions_from_txt(txt)
    
    # initialize result
    automasks = {
        "BPix1": [], "BPix2": [], "BPix3": [], "BPix4": []
    }
    
    # loop over lines
    for automask_description in automask_descriptions:
        if automask_description.startswith('FPix'):
            continue
            # (not yet implemented, will raise error below if not skipped here)
        automask = get_automask_from_description(automask_description)
        key = f"BPix{automask[0]}"
        automasks[key].append(automask[1:])
   
    return automasks


def get_automask_from_tarfile(tarfile, ltp=None, parse=True):
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
      - parse: whether to parse the raw automask descriptions to parsed automasks.
    Returns:
      - if parse is True:
        dict of the following form: {timestamp: dict of masks}
        where the timestamp is "run_lumi" if ltp was provided or raw timestamp if not,
        and where the dict of masks is as returned by get_automask_from_txt.
      - if parse is False:
        dict of the following form: {timestamp: list of raw automask descriptions}
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
            if parse: automasks = get_automask_from_txt(content)
            else: automasks = get_automask_descriptions_from_txt(content)
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


def automask_to_map(automasks, subsystem='BPix1'):
    '''
    Convert automask to boolean np array.
    Input:
    - automasks: a list of lists of the following form:
      [signed ladder coordinate, signed module coordinate, first ROC number, last ROC number]
    - subsystem: choose from "BPix[1,2,3,4]" (needed to set the correct dimensions of the output map)
    Returns:
    - a 2D boolean np array of the correct dimensions (depending on the subsystem),
      with True for the ROCs that are automasked, and False elsewhere.
    '''

    # set number of ladders and modules
    if subsystem=='BPix1': (nladders, nmodules) = (6, 4)
    elif subsystem=='BPix2': (nladders, nmodules) = (14, 4)
    elif subsystem=='BPix3': (nladders, nmodules) = (22, 4)
    elif subsystem=='BPix4': (nladders, nmodules) = (32, 4)
    else: raise Exception(f'Subsystem {subsystem} not recognized.')

    # make base map
    nybins = 2*(2*nladders+1)
    nxbins = 8*(2*nmodules+1)
    automask_map = np.zeros((nybins, nxbins)).astype(bool)
     
    # mask automasks
    for automask in automasks:
        ycoord_base = 2*(automask[0] + nladders)
        xcoord_base = 8*(automask[1] + nmodules)
        for roc in np.arange(automask[2], automask[3]+1):
            ycoord_relative, xcoord_relative = roc_to_coords(roc)
            ycoord = ycoord_base + ycoord_relative
            xcoord = xcoord_base + xcoord_relative
            automask_map[ycoord, xcoord] = True

    return automask_map

def map_to_automask(automask_map):
    '''
    Convert boolean np array to automask
    (inverse operation of automask_to_map).
    Input:
    - automask_map: a 2D boolean np array of the correct dimensions (depending on the subsystem),
      with True for the ROCs that should automasked, and False elsewhere.
    Returns:
    - a list of lists of the following form:
      [signed ladder coordinate, signed module coordinate, first ROC number, last ROC number]
    '''
    
    # get the number of ladders and modules from the shape of the input array
    nladders = int((automask_map.shape[0]-2)/4)
    nmodules = int((automask_map.shape[1]-8)/16)
    
    # get the coordinates of masked ROCs
    y_coords, x_coords = np.nonzero(automask_map.astype(int))
    
    # for each pair of coordinates, get corresponding ladder and module coordinates
    ladders = (y_coords / 2).astype(int) - nladders
    modules = (x_coords / 8).astype(int) - nmodules
    
    # for each pair of coordinates, get the corresponding ROC numbers
    y_coords_relative = y_coords % 2
    x_coords_relative = x_coords % 8
    rocs = [coords_to_roc((y,x)) for y,x in zip(y_coords_relative, x_coords_relative)]
    
    # format the result
    res = ([[ladder, module, roc] for ladder, module, roc in zip(ladders, modules, rocs)])
    res = group_rocs(res)
    return res