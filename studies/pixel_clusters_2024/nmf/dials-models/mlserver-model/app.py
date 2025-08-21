# from this example: https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/sklearn_nmf/mlserver_model/app.py?ref_type=heads

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

# import pixelNMF
thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '..'))
sys.path.append(topdir)
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
            # check shape
            # todo
            # get actual data
            input_data[input_name] = np.array(inputs[idx].data, dtype=input_datatype)
        return input_data

    async def inference(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run inference"""
        flags = self.model.predict(inputs)
        return flags

    async def postprocess(self, results: np.ndarray) -> List[ResponseOutput]:
        """Process results from model inference and make each output compliant with Open Inference Protocol"""
        outputs = ResponseOutput(
                    name='Flag',
                    shape=result.shape,
                    datatype=dtype_to_datatype(result.dtype),
                    data=result.flatten().tolist(),
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