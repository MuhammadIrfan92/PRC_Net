import tensorflow as tf
from tensorflow.keras.models import load_model
from Utils.loss import ccfl_dice
import numpy as np
from tensorflow.keras.utils import normalize
from PIL import Image
from tensorflow.keras import backend as K
import cv2

def global_covariance_pooling(input_tensor):
    mean = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    centered_features = input_tensor - mean
    cov_matrix = tf.einsum('...ijc,...ijd->...cd', centered_features, centered_features)
    cov_matrix /= tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return cov_matrix

def global_covariance_pooling(input_tensor):
    mean = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    centered_features = input_tensor - mean
    cov_matrix = tf.einsum('...ijc,...ijd->...cd', centered_features, centered_features)
    cov_matrix /= tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return cov_matrix

# Example usage within a Keras model
class GlobalCovPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalCovPoolingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(GlobalCovPoolingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return global_covariance_pooling(inputs)
    
    def get_config(self):
        base_config = super(GlobalCovPoolingLayer, self).get_config()
        return {**base_config}



def perform_inference(model_path, img):

    try:
        model = load_model(model_path, custom_objects={"GlobalCovPoolingLayer": GlobalCovPoolingLayer, f"ccfl_dice": ccfl_dice}) # loads model
        test_img = normalize(img)

        model_output = model.predict(test_img)
        print(model_output[0].shape)
        return model_output[0]
    except:
        print('Using sample output image, since no model is found')
        sample_output = cv2.imread('sample_output.png', cv2.IMREAD_GRAYSCALE)
        sample_output = cv2.resize(sample_output, (320,320))
        print(sample_output.shape)
        return sample_output
        
