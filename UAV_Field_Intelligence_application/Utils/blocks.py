import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Softmax, Reshape, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Add, DepthwiseConv2D, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Multiply

from .attentions import *


def DepthwiseSeparable_ConvBlock(input_tensor, filters, kernel_size, strides=(1, 1), dilation_rate=(1,1)):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(input_tensor)
    x = tensorflow.keras.layers.Activation(gelu)(x)
    x = Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
    x = tensorflow.keras.layers.Activation(gelu)(x)
    return x

def LSFB(inputs, filters, kernel_size, num_layers, strides=(1, 1), dilation_rate=(1,1), dropout_rate=0.0, l2_reg=0.0):
  concat_layers = [inputs]
  for _ in range(num_layers):
    x = DepthwiseSeparable_ConvBlock(inputs, filters, kernel_size, strides, dilation_rate)  # x = tf.keras.layers.BatchNormalization()(inputs)
    x = Dropout(dropout_rate)(x)
    concat_layers.append(x)
    inputs = tf.keras.layers.Concatenate(axis=-1)(concat_layers)
  return inputs


def PRF(input_tensor, filters=32, dropout_rate=0.0, l2_reg=0.0):
    branch1 = Conv2D(filters, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(input_tensor)
    branch1 = Dropout(dropout_rate)(branch1)
    branch1_output = concatenate([input_tensor, branch1])
    branch2 = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(branch1_output)
    branch2 = Dropout(dropout_rate)(branch2)
    branch2_output = concatenate([branch1_output, branch2])
    branch3 = Conv2D(filters, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(branch2_output)
    branch3 = Dropout(dropout_rate)(branch3)
    branch3_output = concatenate([branch2_output, branch3])
    output = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(l2_reg))(branch3_output)
    return output


def CCB(input_tensor, filters=32):
    atrous1 = Conv2D(filters, (3, 3), padding='same', dilation_rate=1)(input_tensor)
    atrous2 = Conv2D(filters, (3, 3), padding='same', dilation_rate=2)(input_tensor)
    atrous3 = Conv2D(filters, (3, 3), padding='same', dilation_rate=3)(input_tensor)
    atrous_concat = Concatenate()([atrous1, atrous2, atrous3])
    atrous_concat = Conv2D(filters, (1,1), padding="same")(atrous_concat) # reducing the number of channels of orignal
    attention = CSA(atrous_concat, custom=False)
    output = Activation('relu')(attention)
    return output


def PRF_atten(input_tensor, filters, dropout_rate=0.0, l2_reg=0.0):
    inception_applied = PRF(input_tensor, filters, dropout_rate=dropout_rate, l2_reg=l2_reg)
    attention_applied = CSA(inception_applied)
    output = Activation('relu')(attention_applied)
    return output
