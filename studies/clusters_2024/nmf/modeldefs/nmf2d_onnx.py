# Wrapper class around NMF2D for conversion to ONNX format


# imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from onnx import onnx_pb as onnx_proto

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxRelu, OnnxAdd, OnnxDiv, OnnxMul
from skl2onnx.algebra.onnx_ops import OnnxReshape

from nmf2d import NMF2D


class NMF2DTransformWrapper(BaseEstimator, TransformerMixin):
    
    def __init__(self, nmf2d):
        self.nmf2d = nmf2d
        
    def fit(self, X, y=None):
        # dummy since model is supposed to be already trained
        return self
    
    def transform(self, X):
        # implement the transformation on new data
        return self.nmf2d.predict(X)

    
def skl2onnx_shape_calculator(operator):
        input_type = operator.inputs[0].type
        data_dims = list(input_type.shape)[1:]
        operator.outputs[0].type = FloatTensorType([None] + data_dims)

        
def skl2onnx_converter(scope, operator, container):
    
        # set number of approximation steps
        # (hard-coded for now, maybe find a way to give it as an argument later)
        n_approx_steps = 50
        
        # get references to raw model
        nmf2d = operator.raw_operator.nmf2d # reference to NMF2D instance
        nmf = nmf2d.nmf # reference to MiniBatchNMF instance
        
        # get the model components (H) and their transpose
        # H is of shape (n_components, n_features)
        # H_T is of shape (n_features, n_components)
        H = nmf.components_.astype(np.float32)
        H_T = H.T
        H_name   = scope.get_unique_variable_name("H")
        H_T_name = scope.get_unique_variable_name("H_T")
        container.add_initializer(H_name,
                              onnx_proto.TensorProto.FLOAT,
                              H.shape,
                              H.flatten())
        container.add_initializer(H_T_name,
                              onnx_proto.TensorProto.FLOAT,
                              H_T.shape,
                              H_T.flatten())
        
        # Step 0: Flatten input
        input_name = operator.inputs[0]
        input_shape = nmf2d.xshape
        input_shape_flat = np.prod(input_shape)
        flat_input = scope.get_unique_variable_name("X_flattened")

        flatten_shape_name = scope.get_unique_variable_name("flatten_shape")
        container.add_initializer(flatten_shape_name, onnx_proto.TensorProto.INT64,
                              [2], np.array([-1, input_shape_flat], dtype=np.int64))

        reshape_in = OnnxReshape(input_name, flatten_shape_name,
                             output_names=[flat_input],
                             op_version=container.target_opset)
        
        # intermezzo: get H times H_T, the epsilon parameter, and X times H_T,
        # needed for the approximation of sklearns internal least squares solver
        if n_approx_steps > 0:
            
            # H times H_T
            HH_T = np.matmul(H, H_T)
            HH_T_name = scope.get_unique_variable_name("HH_T")
            container.add_initializer(HH_T_name, onnx_proto.TensorProto.FLOAT, HH_T.shape, HH_T.flatten())
            
            # epsilon
            eps = np.array([1e-8], dtype=np.float32)
            eps_name = scope.get_unique_variable_name("eps")
            container.add_initializer(eps_name, onnx_proto.TensorProto.FLOAT, [1], eps)
            
            # X (flattened) times H_T
            XH_T_name = scope.get_unique_variable_name("XH_T")
            XH_T = OnnxMatMul(flat_input, H_T_name, output_names=[XH_T_name], op_version=container.target_opset)
            XH_T.add_to(scope, container)

        # Step 1: W = ReLU(X @ H^T)
        # matrix dimensons: (n_instances, n_features)
        #                   * (n_features, n_components)
        #                   = (n_instances, n_components)
        W0_raw = scope.get_unique_variable_name("W0_raw")
        W0 = scope.get_unique_variable_name("W0")

        matmul1 = OnnxMatMul(flat_input, H_T_name,
                         output_names=[W0_raw],
                         op_version=container.target_opset)

        relu = OnnxRelu(matmul1,
                    output_names=[W0],
                    op_version=container.target_opset)
        
        W_current = W0
        
        # intermezzo: improve the approximation (optional)
        for stepidx in range(n_approx_steps):
            denom = scope.get_unique_variable_name(f"denom_{stepidx}")
            denom_eps = scope.get_unique_variable_name(f"denom_eps_{stepidx}")
            frac = scope.get_unique_variable_name(f"frac_{stepidx}")
            W_next = scope.get_unique_variable_name(f"W_{stepidx}")
        
            matmul_denom = OnnxMatMul(W_current, HH_T_name, output_names=[denom], op_version=container.target_opset)
            add_eps = OnnxAdd(denom, eps_name, output_names=[denom_eps], op_version=container.target_opset)
            div = OnnxDiv(XH_T_name, denom_eps, output_names=[frac], op_version=container.target_opset)
            update = OnnxMul(W_current, frac, output_names=[W_next], op_version=container.target_opset)
        
            W_current = W_next
        
            for node in [matmul_denom, add_eps, div, update]:
                node.add_to(scope, container)


        # Step 2: X̂ = W @ H
        # matrix dimensions: (n_instances, n_components)
        #                    * (n_components, n_features)
        #                    = (n_instances, n_features)
        X_hat = scope.get_unique_variable_name("X_hat")

        matmul2 = OnnxMatMul(W_current, H_name,
                         output_names=[X_hat],
                         op_version=container.target_opset)
        
        # Step -1: Undo the flattening
        final_output = operator.outputs[0]
        unflatten_shape_name = scope.get_unique_variable_name("unflatten_shape")
        container.add_initializer(unflatten_shape_name, onnx_proto.TensorProto.INT64,
                              [1 + len(input_shape)], np.array([-1, *input_shape], dtype=np.int64))

        reshape_out = OnnxReshape(matmul2, unflatten_shape_name,
                              output_names=[final_output.full_name],
                              op_version=container.target_opset)

        # Add nodes to container
        reshape_in.add_to(scope, container)
        matmul1.add_to(scope, container)
        relu.add_to(scope, container)
        matmul2.add_to(scope, container)
        reshape_out.add_to(scope, container)