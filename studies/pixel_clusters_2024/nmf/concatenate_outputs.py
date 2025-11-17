import os
import sys
import six
import numpy as np
import json


def concatenate_output(outputs):
    # concatenate the outputs of run_evaluation
    
    # make lists
    batch_filter_results = []
    batch_flagged_run_numbers = []
    batch_flagged_ls_numbers = []
    for output in outputs:
        if output is None: continue
        batch_filter_results.append(output['filter_results'])
        if len(output['flagged_run_numbers'])>0:
            batch_flagged_run_numbers.append(output['flagged_run_numbers'])
            batch_flagged_ls_numbers.append(output['flagged_ls_numbers'])

    # contatenate the lists
    filter_results = {}
    if len(batch_filter_results)>0:
        for key in batch_filter_results[0].keys():
            filter_results[key] = []
            for batch_filter_result in batch_filter_results:
                if key not in batch_filter_result.keys(): continue
                filter_results[key] += batch_filter_result[key]
    if len(batch_flagged_run_numbers) > 0:
        flagged_run_numbers = np.concatenate(batch_flagged_run_numbers)
        flagged_ls_numbers = np.concatenate(batch_flagged_ls_numbers)
    else:
        flagged_run_numbers = np.array([])
        flagged_ls_numbers = np.array([])
    
    # make final result
    res = {
        'flagged_run_numbers': flagged_run_numbers,
        'flagged_ls_numbers': flagged_ls_numbers,
        'filter_results': filter_results
    }
    return res


if __name__=='__main__':

    inputdir = sys.argv[1]

    # find input files only differing by part
    merge = {}
    for f in os.listdir(inputdir):
        if not f.startswith('flagged_lumisections'): continue
        if not '_part' in f: continue
        merged = f.split('_part')[0] + '.json'
        merged = os.path.join(inputdir, merged)
        if merged in merge: merge[merged].append(os.path.join(inputdir, f))
        else: merge[merged] = [os.path.join(inputdir, f)]

    # ask confirmation
    print('Will merge as follows:')
    print(json.dumps(merge, indent=2))
    print('Continue? (y/n)')
    go = six.moves.input()
    if go!='y': sys.exit()

    for key, values in merge.items():
        print(f'Now merging {key}...')
        inputs = []
        for value in values:
            with open(value, 'r') as f:
                inputs.append(json.load(f))
        merged = concatenate_output(inputs)
        merged['flagged_run_numbers'] = [int(n) for n in merged['flagged_run_numbers']]
        merged['flagged_ls_numbers'] = [int(n) for n in merged['flagged_ls_numbers']]
        with open(key, 'w') as f:
            json.dump(merged, f)
