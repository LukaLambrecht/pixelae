# copied from here:
# https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/scripts/test_predictions.py

import argparse
from typing import Optional

import numpy as np
import requests


def inference_over_http(data: np.array, model_name: str, port: Optional[str] = None):
    url = f"http://127.0.0.1:{port}/v2/models/{model_name}/infer"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = {
        "inputs": [
            {
                "name": "input_0",
                "shape": data.shape,
                "datatype": "FP32",
                "data": data.tolist()
            }
        ]
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process run number, optional URL, and optional headers.")
    parser.add_argument("-r", "--run_number", type=int, help="The run number (required integer argument).")
    parser.add_argument("-p", "--port", type=str, help="Web server port.")
    parser.add_argument("-n", "--model-name", type=str, help="Model name.")
    args = parser.parse_args()
    
    if args.run_number < 0:
        quit("ERROR: The provided run number must be a non-negative integer.")
    if not args.port:
        quit("ERROR: The port number must be provided.")
    if not args.model_name:
        quit("ERROR: The model name must be provided.")

    data_arr = np.load("../src/data_unfiltered.npy")
    run_arr = np.load("../src/runs_unfiltered.npy")
    unique_runs = np.unique(run_arr)

    if args.run_number not in unique_runs:
        quit(f"ERROR: The specified run number is not present in the sample data. Please, choose one of the following: {unique_runs.tolist()}")
    
    # Collect the data to send to the model
    target_data = data_arr[run_arr == args.run_number]

    # Do the inference
    predictions = inference_over_http(target_data, args.model_name, args.port)

    # Parse the output predictions
    outputs = predictions["outputs"]
    for output in outputs:
        data = np.array(output["data"]).reshape(output["shape"])
        print(f"Output name: {output['name']}, shape: {output['shape']}, datatype: {output['datatype']}")
        print(f"Data: {data}")
        print("==============================")
