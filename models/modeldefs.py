import keras
from keras.layers import Input, Conv2D, ReLU
from keras.layers import Conv2DTranspose, Dense, Add
from keras.layers import Reshape, MaxPooling2D, UpSampling2D
from keras.layers import Flatten
from keras.models import Model


def residual_block_enc(x, filter_number, kernel_size, strides=1, **kwargs):
    y = Conv2D(filter_number, kernel_size, strides=strides, **kwargs)(x)
    y = ReLU()(y)
    y = Conv2D(filter_number, kernel_size, strides=1, **kwargs)(y)
    x = Conv2D(filter_number, 1, strides=strides, **kwargs)(x)
    out = Add()([x, y])
    #out = ReLU()(out)
    return out

def residual_block_dec(x, filter_number, kernel_size, strides=1, **kwargs):
    y = Conv2DTranspose(filter_number, kernel_size, strides=strides, **kwargs)(x)
    y = ReLU()(y)
    y = Conv2DTranspose(filter_number, kernel_size, strides=1, **kwargs)(y)
    x = Conv2DTranspose(filter_number, 1, strides=strides, **kwargs)(x)
    out = Add()([x, y])
    #out = ReLU()(out)
    return out

def model_dummy( input_shape ):
    input_layer = Input(shape=input_shape)
    
    # encoder
    resnet_layer = Conv2D(16, 3, strides=1, padding="same")(input_layer)
    resnet_layer = MaxPooling2D((2,2))(resnet_layer)
    resnet_layer = residual_block_enc(resnet_layer, 16, 3, strides=2, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 32, 3, strides=2, padding="same")
    
    # decoder
    resnet_layer = residual_block_dec(resnet_layer, 32, 3, strides=2, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 16, 3, strides=2, padding="same")
    resnet_layer = UpSampling2D((2, 2))(resnet_layer)
    output_layer = Conv2DTranspose(1, 3, strides=1, padding="same")(resnet_layer)
    
    # model
    resnet_model = Model(inputs = [input_layer], outputs = [output_layer])
    resnet_model.compile(loss="mse", optimizer="adam")
    return resnet_model

def model_ecal_barrel( input_shape ):
    input_layer = Input(shape=input_shape)
    
    # encoder
    resnet_layer = Conv2D(16, 3, strides=1, padding="same")(input_layer)
    resnet_layer = MaxPooling2D((2,2))(resnet_layer)
    resnet_layer = residual_block_enc(resnet_layer, 16, 3, strides=1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 32, 3, strides=2, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 32, 3, strides=1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 64, 3, strides=2, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 64, 3, strides=1, padding="same")
    shape = resnet_layer.shape
    shape = (shape[1], shape[2])
    resnet_layer = MaxPooling2D(shape)(resnet_layer)
    
    # decoder
    resnet_layer = UpSampling2D(shape)(resnet_layer)
    resnet_layer = residual_block_dec(resnet_layer, 64, 3, strides=1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 64, 3, strides=2, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 32, 3, strides=1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 32, 3, strides=2, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 16, 3, strides=1, padding="same")
    resnet_layer = UpSampling2D((2, 2))(resnet_layer)
    output_layer = Conv2DTranspose(1, 3, strides=1, padding="same")(resnet_layer)
    
    # model
    resnet_model = Model(inputs = [input_layer], outputs = [output_layer])
    resnet_model.compile(loss="mse", optimizer="adam")
    return resnet_model

def model_ecal_endcap( input_shape ):
    input_layer = Input(shape=input_shape)
    
    # encoder
    resnet_layer = Conv2D(16, 3, strides=1, padding="same")(input_layer)
    resnet_layer = MaxPooling2D((2,2))(resnet_layer)
    resnet_layer = residual_block_enc(resnet_layer, 16, 3, strides=1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 32, 3, strides=2, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 32, 3, strides=1, padding="same")
    shape = resnet_layer.shape
    shape = (shape[1], shape[2])
    resnet_layer = MaxPooling2D(shape)(resnet_layer)
    
    # decoder
    resnet_layer = UpSampling2D(shape)(resnet_layer)
    resnet_layer = residual_block_dec(resnet_layer, 32, 3, strides=1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 32, 3, strides=2, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 16, 3, strides=1, padding="same")
    resnet_layer = UpSampling2D((2, 2))(resnet_layer)
    output_layer = Conv2DTranspose(1, 3, strides=1, padding="same")(resnet_layer)
    
    # model
    resnet_model = Model(inputs = [input_layer], outputs = [output_layer])
    resnet_model.compile(loss="mse", optimizer="adam")
    return resnet_model
