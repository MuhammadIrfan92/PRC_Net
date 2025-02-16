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
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    x = tensorflow.keras.layers.Activation(gelu)(x)

    
    x = Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    x = tensorflow.keras.layers.Activation(gelu)(x)

    
    return x

def DenseDepthwiseSep_Block(inputs, filters, kernel_size, num_layers, strides=(1, 1), dilation_rate=(1,1), dropout_rate=0.0, l2_reg=0.0):
  """
  Implements a DenseNet block.

  Args:
    inputs: Input tensor.
    growth_rate: Number of filters to be added per layer.
    num_layers: Number of layers in the block.

  Returns:
    Output tensor of the DenseNet block.
  """

  concat_layers = [inputs]

  for _ in range(num_layers):
    x = DepthwiseSeparable_ConvBlock(inputs, filters, kernel_size, strides, dilation_rate)  # x = tf.keras.layers.BatchNormalization()(inputs)
    x = Dropout(dropout_rate)(x)
    concat_layers.append(x)
    inputs = tf.keras.layers.Concatenate(axis=-1)(concat_layers)

  return inputs




def Atrous_Channel_AttentionGCP_Block(input_tensor, filters=32):
    """Spatial attention focuses on relationships between elements within a local neighborhood
      (defined by the kernel size of the attention convolution) in the feature maps."""
    
    # Atrous convolutions
    atrous1 = Conv2D(filters, (3, 3), padding='same', dilation_rate=1)(input_tensor)
    atrous2 = Conv2D(filters, (3, 3), padding='same', dilation_rate=2)(input_tensor)
    atrous3 = Conv2D(filters, (3, 3), padding='same', dilation_rate=3)(input_tensor)
    atrous_concat = Concatenate()([atrous1, atrous2, atrous3])
    atrous_concat = Conv2D(filters, (1,1), padding="same")(atrous_concat) # reducing the number of channels of orignal

    # Vanila Channel Attention.
    attention = CSA(atrous_concat, custom=False)


    # Activation
    output = Activation('relu')(attention)
    
    return output





def InceptionDenseAttentionBlock(input_tensor, filters, dropout_rate=0.0, l2_reg=0.0):
    inception_applied = PRF(input_tensor, filters, dropout_rate=dropout_rate, l2_reg=l2_reg)
    attention_applied = CSA(inception_applied)
    output = Activation('relu')(attention_applied)
    return output
