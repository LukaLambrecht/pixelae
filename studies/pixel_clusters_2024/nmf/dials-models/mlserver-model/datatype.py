# from this example: https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/sklearn_nmf/mlserver_model/datatype.py?ref_type=heads

import numpy as np


DTYPE_MAP = {
    np.bool_: "BOOL",
    np.uint8: "UINT8",
    np.uint16: "UINT16",
    np.uint32: "UINT32",
    np.uint64: "UINT64",
    np.int8: "INT8",
    np.int16: "INT16",
    np.int32: "INT32",
    np.int64: "INT64",
    np.float16: "FP16",
    np.float32: "FP32",
    np.float64: "FP64",
}
TENSOR_TYPE_MAP = {type_string: np_dtype for np_dtype, type_string in DTYPE_MAP.items()}


def datatype_to_dtype(tensor_type):
    """
    Convert a tensor datatype string compliant to OIP to a numpy dtype
    Docs: https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/#tensor-data-types_1
    """
    return TENSOR_TYPE_MAP[tensor_type]


def dtype_to_datatype(dtype):
    """
    Convert a numpy dtype to a tensor datatype string compliant to OIP
    Docs: https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/#tensor-data-types_1
    """
    return DTYPE_MAP.get(np.dtype(dtype).type, "UNKNOWN")