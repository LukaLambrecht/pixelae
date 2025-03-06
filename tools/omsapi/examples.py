### imports

# external modules
import os
import sys
import json
import time
import numpy as np
from fnmatch import fnmatch

# local modules
thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

import tools.omstools as omstools
import tools.omsapi.get_oms_data as oms


def get_lumisection_info(omsapi, runrange, attributes, defaults=None, run_filter=None):
    '''
    Retrieve per-lumisection info from OMS, using the 'lumisections' endpoint.
    Input arguments:
    - omsapi: OMSAPI instance.
    - runrange: tuple of the form (first run number, last run number).
    - attributes: list of attributes to retrieve.
    - defaults: dict matching attributes to default values to substitute None.
    - run_filter: list or np.array of runs to keep.
    '''
    
    # make the API call
    print('Making API call...')
    info = oms.get_oms_data_iterative(omsapi, 'lumisections', runrange, attributes)
    attrs = list(info.keys())
    firstkey = attrs[0]
    nlumis = len(info[firstkey])
    nattrs = len(attrs)
    print(f'Found {nlumis} lumisections with {nattrs} attributes')
    
    # substitute occasional None values
    if defaults is not None:
        print('Substituting None values with defaults...')
        for key, val in defaults.items():
            print(f'  - Now checking attribute {key}...')
            if key not in info.keys():
                print(f'    Attribute "{key}" not found, skipping...')
                continue
            nnone = info[key].count(None)
            if nnone==0: continue
            msg = f'    Found {nnone} None instances (out of {len(info[key])} total);'
            msg += f' substituting with {val}'
            print(msg)
            info[key] = [el if el is not None else val for el in info[key]]
    
    # filter run numbers
    if run_filter is not None:
        run_filter = np.array(run_filter)
        print('Filtering runs...')
        runkey = 'run_number'
        if not runkey in info.keys():
            print(f'Attribute "{run_number}" which is needed for run filtering not found; skipping...')
        else:
            runnbs = np.array(info[runkey]).astype(int)
            unique_runnbs = np.unique(runnbs)
            run_mask = np.isin(unique_runnbs, run_filter).astype(bool)
            ls_mask = np.isin(runnbs, run_filter).astype(bool)
            print(f'Keeping {np.sum(run_mask)} / {len(run_mask)} runs and {np.sum(ls_mask)} / {len(ls_mask)} lumisections.')
            for attribute in info.keys():
                info[attribute] = [info[attribute][idx] for idx in range(len(info[attribute])) if ls_mask[idx]]
    
    # return the result
    return info


def get_hltpaths(omsapi, run, hltpaths=None):
    '''
    Retrieve per-run HLT path info from OMS, using the 'hltpathinfo' endpoint.
    Input arguments:
    - omsapi: OMSAPI instance.
    - run: run number.
    - hltpaths: list of hlt path names to keep in the result (default: all).
      (note: may contain UNIX-style wildcards).
    '''
    
    # initializations
    run = int(run)
        
    # while-try-except wrapper
    success = False
    while not success:
        try:
            limit_entries = 10000
            hltinfo = oms.get_oms_data(omsapi, 'hltpathinfo', runnb=run,
                                   attributes=['path_name'], limit_entries=limit_entries)
            available_hltpaths = oms.get_oms_response_attribute(hltinfo, 'path_name')
            success = True
        except:
            print('WARNING: OMS query failed, retrying...')
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(3)
            
    # filter hlt paths if requested
    if hltpaths is not None:
        filtered_hltpaths = []
        for available_hltpath in available_hltpaths:
            keep = False
            for hltpath in hltpaths:
                if fnmatch(available_hltpath, hltpath): keep = True
            if keep: filtered_hltpaths.append(available_hltpath)
        available_hltpaths = filtered_hltpaths
    
    # return the result
    return available_hltpaths
             
    
def get_hltrate_info(omsapi, runrange, hltpaths,
                     attributes=None, run_filter=None,
                     outputdir=None, outputfilebase=None):
    '''
    Retrieve HLT rate info from OMS, using the 'hltpathrates' endpoint.
    Input arguments:
    - omsapi: OMSAPI instance.
    - runrange: tuple of the form (first run number, last run number).
    - hltpaths: list of HLT paths to retrieve (note: may contain UNIX-style wildcards).
    - attributes: list of attributes to retrieve for each trigger
      (default: 'rate', 'first_lumisection_number', and 'run_number')
    - run_filter: list or np.array of runs to keep.
    - outputdir: directory to store temporary per-run files.
      (this is optional, but can help to avoid having to rerun everything
      in case of transient network errors).
    - outputfilebase: base name to uniquely identify output files for this call.
    '''
    
    # initializations
    info = {}
    if attributes is None:
        attributes = ([
            'rate',
            'first_lumisection_number',
            'run_number'
        ])
    
    # get a list of runs in the requested range
    limit_entries = 2000
    runs = oms.get_oms_data(omsapi, 'runs', runnb=runrange, attributes=['run_number'], limit_entries=limit_entries)
    runs = oms.get_oms_response_attribute(runs, 'run_number')
    if len(runs) >= limit_entries:
        msg = f'WARNING: the number of runs is equal to the limit ({limit_entries}),'
        msg += ' so a part of the runs is likely missing; make a smaller run range.'
        print(msg)
        
    # filter runs if requested
    if run_filter is not None:
        filtered_runs = [r for r in runs if r in run_filter]
        print('Found {} runs in OMS, of which {} are in the DQMIO files.'.format(len(runs), len(filtered_runs)))
        runs = filtered_runs
    
    # loop over runs
    print('Looping over {} runs...'.format(len(runs)))
    for run in runs: 
        info[run] = {}
        
        # get the trigger names for this run
        triggers = get_hltpaths(omsapi, run, hltpaths=hltpaths)
        print(f'Run {run}: found following triggers: {triggers}.')
        
        # loop over trigger names
        for trigger in triggers:
            path_filter = {'attribute_name':'path_name', 'value':trigger, 'operator':'EQ'}
            per_lumi_arg = {'group[granularity]':'lumisection'}
            success = False
            while not success:
                try:
                    limit_entries = 10000
                    rate_info = oms.get_oms_data(omsapi, 'hltpathrates', runnb=run, attributes=attributes,
                                     extraargs=per_lumi_arg, extrafilters=[path_filter], limit_entries=limit_entries)
                    rate = {attribute: oms.get_oms_response_attribute(rate_info, attribute) for attribute in attributes}
                    if len(attributes) > 0 and len(rate[attributes[0]]) >= limit_entries:
                        msg = f'WARNING: the number of lumisections is equal to the limit ({limit_entries}),'
                        msg += ' so a part of the lumisections is likely missing; increase the limit.'
                        print(msg)
                    success = True
                except:
                    print('WARNING: OMS query failed, retrying...')
                    sys.stdout.flush()
                    sys.stderr.flush()
                    time.sleep(3)
            info[run][trigger] = rate   
            
        # store the information for this run
        if outputdir is not None:
            if not os.path.exists(outputdir): os.makedirs(outputdir)
            if outputfilebase is None: outputfilebase = 'hltrate'
            outputfilename = os.path.splitext(os.path.basename(outputfilebase))[0]
            outputfilename = outputfilename + f'_{run}.json'
            outputfile = os.path.join(outputdir, outputfilename)
            with open(outputfile, 'w') as f:
                json.dump(info[run], f)
                
    # return result
    return info


def get_hltprescale_info(omsapi, runrange, hltpaths,
                     run_filter=None,
                     outputdir=None, outputfilebase=None):
    
    # initializations
    info = {}
    
    # get a list of runs in the requested range
    limit_entries = 2000
    runs = oms.get_oms_data(omsapi, 'runs', runnb=runrange, attributes=['run_number'], limit_entries=limit_entries)
    runs = oms.get_oms_response_attribute(runs, 'run_number')
    if len(runs) >= limit_entries:
        msg = f'WARNING: the number of runs is equal to the limit ({limit_entries}),'
        msg += ' so a part of the runs is likely missing; make a smaller run range.'
        print(msg)
        
    # filter runs if requested
    if run_filter is not None:
        filtered_runs = [r for r in runs if r in run_filter]
        print('Found {} runs in OMS, of which {} are in the DQMIO files.'.format(len(runs), len(filtered_runs)))
        runs = filtered_runs
    
    # loop over runs
    print('Looping over {} runs...'.format(len(runs)))
    for run in runs: 
        info[run] = {}
        
        # get the trigger names for this run
        triggers = get_hltpaths(omsapi, run, hltpaths=hltpaths)
        print('Run {}: found following triggers: {}'.format(run, triggers))
        
        # get prescale info for this run
        success = False
        extraargs = {'filter[run_number]': str(run)}
        while not success:
            try:
                info_raw = oms.get_oms_data( omsapi, 'hltprescalesets',
                                     extraargs=extraargs, limit_entries=10000 )
                path_names = oms.get_oms_response_attribute(info_raw, 'path_name')
                prescales = oms.get_oms_response_attribute(info_raw, 'prescales')
                success = True
            except:
                print('WARNING: OMS query failed, retrying...')
                sys.stdout.flush()
                sys.stderr.flush()
                time.sleep(3)
                
        # loop over trigger names
        for trigger in triggers:
            path_idx = path_names.index(trigger)
            this_prescales = prescales[path_idx]
            info[run][trigger] = this_prescales   
        
        # store the information for this run
        if outputdir is not None:
            if not os.path.exists(outputdir): os.makedirs(outputdir)
            if outputfilebase is None: outputfilebase = 'hltprescale'
            outputfilename = os.path.splitext(os.path.basename(outputfilebase))[0]
            outputfilename = outputfilename + f'_{run}.json'
            outputfile = os.path.join(outputdir, outputfilename)
            with open(outputfile, 'w') as f:
                json.dump(info[run], f)
                
    # return the result
    return info