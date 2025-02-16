

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Softmax, Reshape, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import numpy as np

from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Multiply, Conv3D, Conv3DTranspose

from .attentions import *
from .blocks import *


seed = 0
np.random.seed(seed)
tensorflow.random.set_seed(seed)



def PRC_Net(n_classes=3, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1, dropout_rate=0.0, l2_reg=0.0):

    block1_filters = 24
    block2_filters = 48
    block3_filters = 128
    block4_filters = 256
    bottleneck_filters = 256

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Contracting Path
    c1 = InceptionDenseAttentionBlock(inputs, filters=block1_filters, dropout_rate=dropout_rate, l2_reg=l2_reg)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = InceptionDenseAttentionBlock(p1, filters=block2_filters, dropout_rate=dropout_rate, l2_reg=l2_reg)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = InceptionDenseAttentionBlock(p2, filters=block3_filters, dropout_rate=dropout_rate, l2_reg=l2_reg)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = InceptionDenseAttentionBlock(p3, filters=block4_filters, dropout_rate=dropout_rate, l2_reg=l2_reg)
    # start of LCFE
    p4 = MaxPooling2D((2, 2))(c4)
    c4skip = Atrous_Channel_AttentionGCP_Block(c4)

    c5 = DenseDepthwiseSep_Block(p4, bottleneck_filters, kernel_size=3, num_layers=4, dropout_rate=dropout_rate, l2_reg=l2_reg)


    # Expansive Path
    u6 = Conv2DTranspose(block4_filters, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4skip])
    # End of LCFE
    c6 = Conv2D(block4_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(u6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(block4_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(c6)
    # c6 = ChannelAttentionBlock_GAP_MaxP_Conv1D_GCP(c6)


    u7 = Conv2DTranspose(block3_filters, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(block3_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(u7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(block3_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(c7)
    # c7 = ChannelAttentionBlock_GAP_MaxP_Conv1D_GCP(c7)


    u8 = Conv2DTranspose(block2_filters, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(block2_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(u8)
    c8 = Dropout(dropout_rate)(c8)
    c8 = Conv2D(block2_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(c8)
    # c8 = ChannelAttentionBlock_GAP_MaxP_Conv1D_GCP(c8)


    u9 = Conv2DTranspose(block1_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(block1_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(u9)
    c9 = Dropout(dropout_rate)(c9)
    c9 = Conv2D(block1_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(c9)
    # c9 = ChannelAttentionBlock_GAP_MaxP_Conv1D_GCP(c9)


    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
