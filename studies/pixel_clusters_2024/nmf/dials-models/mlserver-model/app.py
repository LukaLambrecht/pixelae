# Handler class defining the interface between central DIALS syntax and custom model syntax

# modified from this example:
# https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/sklearn_nmf/mlserver_model/app.py?ref_type=heads

# note: inside the container, only the files in the mlserver-model directory are visible.
#       importing local modules from outside this directory is not trivial
#       (although allegedly possible by making everything a package, but I did not test that yet),
#       and the simplest (though far from optimal) solution might be to duplicate all local dependencies
#       in the local model definition; see pixelnmf.py.


# import external modules
import os
import sys
from typing import List, Optional, Tuple, Dict
import joblib
import numpy as np
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, RequestInput, RequestOutput, ResponseOutput
from mlserver.utils import get_model_uri

# import local modules
from datatype import dtype_to_datatype
from datatype import datatype_to_dtype
from pixelnmf import PixelNMF


class Handler(MLModel):
    async def load(self):
        model_uri = await get_model_uri(self._settings)
        self.model_name = self._settings.name
        self.model_version = self._settings.version
        self.model = joblib.load(model_uri)

    async def preprocess(self, inputs: List[RequestInput]) -> Dict[str, np.ndarray]:
        """Process data sent from HTTP request"""
        input_data = {}
        # loop over monitoring elements
        for idx in range(len(inputs)):
            # get basic attributes
            input_name = inputs[idx].name
            input_datatype = datatype_to_dtype(inputs[idx].datatype)
            input_shape = tuple(inputs[idx].shape)
            # get actual data
            # note: data should be received in flattened format,
            #       so need to unflatten here; see more info here:
            #       https://gitlab.cern.ch/cms-ppd/technical-support/web-services/dials-service/-/issues/136#note_10063426
            data = np.array(inputs[idx].data, dtype=input_datatype)
            if data.shape != input_shape:
                data = data.flatten().reshape(input_shape)
            input_data[input_name] = data
        return input_data

    async def inference(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run inference"""
        flags = self.model.predict(inputs)
        return flags

    async def postprocess(self, results: np.ndarray) -> List[ResponseOutput]:
        """Process results from model inference and make each output compliant with Open Inference Protocol"""
        outputs = [
                    ResponseOutput(
                      name='Flag',
                      shape=results.shape,
                      datatype=dtype_to_datatype(results.dtype),
                      data=results.flatten().tolist(),
                    )
                  ]
        # For now, DIALS requires a MetricKey (which is supposed to be a continuous score),
        # on top of an optional FlaggingKey (which is the MetricKey with a built-in threshold applied).
        # In this method, the MetricKey cannot be well defined, so return a dummy value for now.
        # This can be changed in future versions of DIALS, when the MetricKey is no longer required.
        dummy_metric = results.astype(float)
        outputs.append(
            ResponseOutput(
                name='Metric',
                shape=dummy_metric.shape,
                datatype=dtype_to_datatype(dummy_metric.dtype),
                data=dummy_metric.flatten().tolist(),
            )
        )

        return outputs

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """
        Main handler called on each inference HTTP request.
        """
        data = await self.preprocess(request.inputs)
        data = await self.inference(data)
        data = await self.postprocess(data)
        return InferenceResponse(
            id=request.id, model_name=self.model_name, model_version=self.model_version, outputs=data
        )
