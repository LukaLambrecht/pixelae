# copied and modified from here:
# https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/scripts/test_predictions.py
# extra: OMS data fetching copied and modified from here:
# https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/src/fetch_sumET_data.ipynb

import os
import sys
import pickle
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import requests
from fnmatch import fnmatch


# set variables for OMS access
KEYCLOAK_SERVER_URL = "https://auth.cern.ch/auth/"
KEYCLOAK_REALM = "cern"
OMS_AUDIENCE = "cmsoms-prod"
OMS_API_URL = "https://cmsoms.cern.ch/agg/api/v1"

# read OMS client ID and secret
try:
    import omsid
    OMS_SSO_CLIENT_ID = omsid.OMS_SSO_CLIENT_ID
    OMS_SSO_CLIENT_SECRET = omsid.OMS_SSO_CLIENT_SECRET
except:
    msg = 'WARNING: could not read OMS client ID and secret;'
    msg += ' OMS authentication will probably not work.'
    print(msg)


def issue_oms_api_token() -> dict:
    '''Issue an OMS api token for future authentication'''
    response = requests.post(
        f"{KEYCLOAK_SERVER_URL}realms/{KEYCLOAK_REALM}/api-access/token",
        data={
            "grant_type": "client_credentials",
            "client_id": OMS_SSO_CLIENT_ID,
            "client_secret": OMS_SSO_CLIENT_SECRET,
            "audience": OMS_AUDIENCE,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def oms_query(endpoint: str, token_type: str, access_token: str, **kwargs):
    '''Make an OMS query'''
    headers = {"Authorization": f"{token_type} {access_token}", "Content-type": "application/json"}
    response = requests.get(
        f"{OMS_API_URL}/{endpoint}",
        headers=headers,
        timeout=30,
        verify=False,
        **kwargs,
    )

    response.raise_for_status()
    return response.json()

def get_oms_data(run_numbers, ls_numbers):
    '''
    Wrapper around oms_query for a given set of lumisections
    '''

    # set attributes to retrieve
    attrs = [
      "lumisection_number",
      "run_number",
      "cms_active",
      "beams_stable",
      "bpix_ready",
      "fpix_ready",
      "tibtid_ready",
      "tecm_ready",
      "tecp_ready",
      "tob_ready",
      "pileup",
    ]

    # extract unique run numbers
    runs = np.unique(run_numbers)

    # loop over runs and make the OMS API call per run
    oms_df = []
    for run in runs:
        params = {
          "fields": ",".join(attrs) + ",lumisection_number",
          "filter[run_number][EQ]": run,
          "page[offset]": 0,
          "page[limit]": 5000,
          "sort": "lumisection_number",
        }
        oms_response = oms_query(
          endpoint="lumisections",
          token_type=token["token_type"],
          access_token=token["access_token"],
          params=params
        )
        for entry in oms_response.get("data", []):
            attrs = entry["attributes"]
            oms_df.append(attrs)
    oms_df = pd.DataFrame(oms_df)

    # filter on provided run and lumisection numbers
    filter_df = pd.DataFrame({
      'run_number': run_numbers,
      'lumisection_number': ls_numbers
    })
    filtered_df = filter_df.merge(oms_df, on=['run_number', 'lumisection_number'])

    # return result
    return filtered_df

def get_hlt_data(run_numbers, ls_numbers):
    '''
    Wrapper around oms_query for a given set of lumisections
    '''

    # set attributes to retrieve
    hlt_path = "HLT_ZeroBias_v*"

    # extract unique run numbers
    runs = np.unique(run_numbers)

    # loop over runs and make the OMS API call per run
    oms_df = []
    for run in runs:
        # first get available triggers
        params = {
          "fields": "path_name",
          "filter[run_number][EQ]": run,
          "page[offset]": 0,
          "page[limit]": 5000,
        }
        oms_response = oms_query(
            endpoint="hltpathinfo",
            token_type=token["token_type"],
            access_token=token["access_token"],
            params=params
        )
        matching_hlt_paths = []
        for entry in oms_response.get('data', []):
            path = entry['attributes']['path_name']
            if fnmatch(path, hlt_path): matching_hlt_paths.append(path)
        if len(matching_hlt_paths)!=1:
            msg = f'Found no matching trigger paths for pattern "{hlt_path}"'
            raise Exception(msg)
        this_hlt_path = matching_hlt_paths[0]
        # get rate
        params = {
          "filter[run_number][EQ]": run,
          "filter[path_name][EQ]": this_hlt_path,
          "group[granularity]": "lumisection",
          "page[offset]": 0,
          "page[limit]": 5000,
          "sort": "lumisection_number",
        }
        oms_response = oms_query(
          endpoint="hltpathrates",
          token_type=token["token_type"],
          access_token=token["access_token"],
          params=params
        )
        for entry in oms_response.get("data", []):
            attrs = entry['attributes']
            oms_df.append({
                'run_number': attrs['run_number'],
                'lumisection_number': attrs['first_lumisection_number'],
                'hlt_rate': attrs['rate']
                })
    oms_df = pd.DataFrame(oms_df)

    # filter on provided run and lumisection numbers
    filter_df = pd.DataFrame({
      'run_number': run_numbers,
      'lumisection_number': ls_numbers
    })
    filtered_df = filter_df.merge(oms_df, on=['run_number', 'lumisection_number'])

    # return result
    return filtered_df

def inference_over_http(data: dict, model_name: str, port=None, oms_data=None):
    '''Make inference request over local network'''
    
    print('Formatting request...')
    url = f"http://127.0.0.1:{port}/v2/models/{model_name}/infer"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # build inputs
    inputs = []
    for name, arr in data.items():
        this_input = {}
        this_input['name'] = name
        this_input['shape'] = arr.shape
        this_input['datatype'] = "INT32"
        # note: data below should be flattened!
        #       it will run fine locally if not flattened,
        #       but that doesn't work in production.
        #       see more info here:
        #       https://gitlab.cern.ch/cms-ppd/technical-support/web-services/dials-service/-/issues/136#note_10063426
        this_input['data'] = arr.flatten().tolist()
        inputs.append(this_input)

    # extend inputs with OMS data
    if oms_data is not None:
        for name, arr in oms_data.items():
            # default case for DCS bits
            this_input = {}
            this_input['name'] = 'dcs_bits__'+name
            this_input['shape'] = arr.shape
            this_input['datatype'] = "BOOL"
            this_input['data'] = arr.flatten().tolist()
            # special cases
            if name=='run_number' or name=='lumisection_number':
                this_input['name'] = 'general__'+name
                this_input['datatype'] = 'INT32'
            if name=='pileup':
                this_input['name'] = 'pileup__'+name
                this_input['datatype'] = "FP32"
            if name=='hlt_rate':
                this_input['name'] = 'hlt_zerobias__hlt_zerobias_rate'
                this_input['datatype'] = "FP32"
            # add to list
            inputs.append(this_input)

    # format body
    body = {
        "inputs": inputs
    }

    # make request
    print('Posting request...')
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    print('Response received.')
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process run number, optional URL, and optional headers.")
    parser.add_argument("-p", "--port", type=str, help="Web server port.")
    parser.add_argument("-n", "--model-name", type=str, help="Model name.")
    parser.add_argument("-d", "--data", type=str, help="Path to pkl file with input data.")
    args = parser.parse_args()
    
    if not args.port:
        quit("ERROR: The port number must be provided.")
    if not args.model_name:
        quit("ERROR: The model name must be provided.")

    # load data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    firstkey = list(data.keys())[0]
    print('Loaded data with following shapes:')
    for name, arr in data.items():
        print(f'  - {name}: {arr.shape}')

    # extract run and lumisection numbers,
    # and keep the actual input data separate
    run_numbers = data['run_number']
    ls_numbers = data['ls_number']
    data = {key: val for key, val in data.items()
            if key not in ['run_number', 'ls_number']}

    # take only a small part of the data for quicker testing
    select_slice = None
    #select_slice = slice(2000, 3000)
    if select_slice is not None:
        data = {name: arr[select_slice, :, :] for name, arr in data.items()}
        run_numbers = run_numbers[select_slice]
        ls_numbers = ls_numbers[select_slice]

    # get OMS data
    print('Retrieving OMS data for selected data')
    token = issue_oms_api_token()
    oms_data = get_oms_data(run_numbers, ls_numbers)
    hlt_data = get_hlt_data(run_numbers, ls_numbers)
    oms_data = oms_data.merge(hlt_data, on=['run_number', 'lumisection_number'])
    oms_data = {name: np.array(oms_data[name].values) for name in oms_data.columns}

    # do the inference
    predictions = inference_over_http(data, args.model_name, args.port, oms_data=oms_data)

    # parse the output predictions
    outputs = predictions["outputs"]
    for output in outputs:
        data = np.array(output["data"]).reshape(output["shape"])
        print(f"Output name: {output['name']}, shape: {output['shape']}, datatype: {output['datatype']}")
        print(f"Data: {data}")
        print(f'Number of LS: {len(data)}')
        print(f'Number of flagged LS: {np.sum(data.astype(int))}')
        print("==============================")
