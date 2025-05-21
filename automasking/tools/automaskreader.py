import os
import sys
import json
import numpy as np

# local modules
thisdir = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

from automasking.tools.automask_file_parsing import automask_to_map


class AutomaskReader(object):

    def __init__(self, automask_json_file):
        
        # read the json file
        with open(automask_json_file, 'r') as f:
            self.automask_info = json.load(f)
            
        # make a faster-access version of the available keys
        self.automask_keys = np.zeros(len(self.automask_info))
        for idx, key in enumerate(self.automask_info.keys()):
            run, ls = key.split('_')
            self.automask_keys[idx] = int(run)*10000+int(ls)
        if not np.all(self.automask_keys[:-1] <= self.automask_keys[1:]):
            msg = 'Input is not sorted; this case is not yet implemented.'
            raise Exception(msg)
            
    def get_automask_for_ls(self, run, ls, subsystem, verbose=False):
        strkey = str(run) + '_' + str(ls)
        intkey = int(run)*10000+int(ls)
        
        # simplest case where the key is directly present
        if intkey in self.automask_keys:
            if verbose: print(f'Returning automask for key {strkey}') 
            return self.automask_info[strkey][subsystem]
        
        # otherwise, need to determine whether to take closest key before (default)
        # or after (if the key before is from the previous run)
        idx = np.searchsorted(self.automask_keys, intkey)
        intkey_after = self.automask_keys[idx]
        intkey_before = self.automask_keys[idx-1]
        run_before = int(intkey_before / 10000)
        ls_before = int(intkey_before % 10000)
        if run_before==int(run):
            strkey = str(run_before)+'_'+str(ls_before)
            if verbose: print(f'Returning automask for key {strkey}') 
            return self.automask_info[strkey][subsystem]
        run_after = int(intkey_after / 10000)
        ls_after = int(intkey_after % 10000)
        strkey = str(run_after)+'_'+str(ls_after)
        if verbose: print(f'Returning automask for key {strkey}') 
        return self.automask_info[strkey][subsystem]
    
    def get_automask_map_for_ls(self, run, ls, subsystem, invert=False):
        automask = self.get_automask_for_ls(run, ls, subsystem)
        automask_map = automask_to_map(automask, subsystem=subsystem)
        if invert: automask_map = np.invert(automask_map)
        return automask_map
    
    def get_automask_maps_for_ls(self, runs, ls, subsystem, **kwargs):
        return np.array([self.get_automask_map_for_ls(run, lumi, subsystem, **kwargs) for run, lumi in zip(runs, ls)])