import os
import sys
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
    thisdir = os.path.dirname(__file__)
    sys.path.append(thisdir)
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

def get_oms_data(attributes, run_numbers, ls_numbers):
    '''
    Wrapper around oms_query for a given set of lumisections
    '''
    token = issue_oms_api_token()
    attributes += ['run_number', 'lumisection_number']

    # extract unique run numbers
    runs = np.unique(run_numbers)

    # loop over runs and make the OMS API call per run
    oms_df = []
    for run in runs:
        params = {
          "fields": ",".join(attributes),
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

def get_hlt_data(hlt_path, run_numbers, ls_numbers):
    '''
    Wrapper around oms_query for a given set of lumisections
    '''
    token = issue_oms_api_token()

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

        # find triggers matching requested one
        matching_hlt_paths = []
        for entry in oms_response.get('data', []):
            path = entry['attributes']['path_name']
            if fnmatch(path, hlt_path): matching_hlt_paths.append(path)

        # safety in case on matching trigger found:
        # use a dummy path name (needed to preserve the syntax of the output),
        # but set it to 0 afterwards.
        if len(matching_hlt_paths)!=1:
            msg = f'WARNING: found no matching trigger paths for pattern "{hlt_path}"'
            print(msg)
            this_hlt_path = oms_response['data'][0]['attributes']['path_name']
            isdummy = True
        else:
            this_hlt_path = matching_hlt_paths[0]
            isdummy = False

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
                'hlt_rate': attrs['rate'] if not isdummy else np.zeros(len(attrs['run_number']))
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
