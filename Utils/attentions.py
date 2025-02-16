import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Softmax, Reshape, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Add, SimpleRNN, Conv1D, GlobalAveragePooling1D
from tensorflow.keras import backend as K
import numpy as np

from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Multiply

seed = 0
np.random.seed(seed)
tensorflow.random.set_seed(seed)



def global_covariance_pooling(input_tensor):
    # Step 1: Calculate the global average
    mean = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    
    # Step 2: Center the features by subtracting the mean
    centered_features = input_tensor - mean
    
    # Step 3: Calculate the covariance
    # For batch processing, we can use einsum to compute outer products and averages over spatial dimensions
    cov_matrix = tf.einsum('...ijc,...ijd->...cd', centered_features, centered_features)
    cov_matrix /= tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    
    # Optional Step 4: Apply matrix power normalization (e.g., matrix square root)
    # You can use TensorFlow's linear algebra operations here if needed

    return cov_matrix



class GlobalCovPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalCovPoolingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create weights, biases, etc. if needed
        super(GlobalCovPoolingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return global_covariance_pooling(inputs)
    
    def get_config(self):
        base_config = super(GlobalCovPoolingLayer, self).get_config()
        return {**base_config}

import tensorflow as tf



def CSA(input_tensor, ratio=8, custom=False):
    channels = K.int_shape(input_tensor)[-1]
    
    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, channels))(avg_pool)  # Reshape to (batch_size, 1, channels)
    
    # Global Max Pooling
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, channels))(max_pool)  # Reshape to (batch_size, 1, channels)
    
    # Global covariance Pooling

    gcp_pool = GlobalCovPoolingLayer()(input_tensor)


    gcp_pool = Conv1D(filters=channels, kernel_size=1, padding='same', activation='relu')(gcp_pool)

    gcp_pool = GlobalAveragePooling1D()(gcp_pool)


    combined = Add()([avg_pool, max_pool, gcp_pool])

    # Shared Conv1D layer
    shared_conv = Conv1D(filters=channels // ratio, kernel_size=3, activation='relu', padding='same')
    combined = shared_conv(combined)

    conv_out = Conv1D(filters=channels, kernel_size=1, activation='sigmoid', padding='same')
    combined = conv_out(combined)
    
    # Apply GlobalAveragePooling1D 
    combined = GlobalAveragePooling1D()(combined)

    #************************
    # Reshape to match input tensor's channel dimensions
    combined = Reshape((1, 1, channels))(combined)
    
    # Scale input tensor
    modified_tensor = Multiply()([input_tensor, combined])
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&CUSTOM GCP NOT APPLIED&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    
    return modified_tensor



















