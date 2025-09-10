# copied and modified from here:
# https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/scripts/test_predictions.py

import pickle
import argparse
from typing import Optional
import numpy as np
import requests


def inference_over_http(data: dict, model_name: str, port: Optional[str] = None):
    
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
    args = parser.parse_args()
    
    if not args.port:
        quit("ERROR: The port number must be provided.")
    if not args.model_name:
        quit("ERROR: The model name must be provided.")

    # load data
    with open('test_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # take only a small part
    #data = {name: arr[:1000, :, :] for name, arr in data.items()}

    # do the inference
    predictions = inference_over_http(data, args.model_name, args.port)

    # Parse the output predictions
    outputs = predictions["outputs"]
    for output in outputs:
        data = np.array(output["data"]).reshape(output["shape"])
        print(f"Output name: {output['name']}, shape: {output['shape']}, datatype: {output['datatype']}")
        print(f"Data: {data}")
        print(f'Number of LS: {len(data)}')
        print(f'Number of flagged LS: {np.sum(data.astype(int))}')
        print("==============================")
