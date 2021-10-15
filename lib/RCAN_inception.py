from re import S
from tensorflow.keras.layers import *
from tensorflow.keras import *

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

import math

# RCAN implementation (TF 2.x) based on:
#   Original implementation (PyTorch): https://github.com/yulunzhang/RCAN
#   Unofficial implementation (TF 1.13): https://github.com/dongheehand/RCAN-tf

'''
Modificación experimental de la arquitectura
Los bloques residuales (función RCA_Block) contendrán tres capas de convolución en paralelo de distinto tamaño de kernel
inspirado levemente en la arquitectura Inception
'''

# Mean Shifter

RGB_MEAN = [0.4488, 0.4371, 0.4040]

def Mean_Shifter(x, rgb_mean, sign, rgb_range=255):
    
    kernel = tf.constant(shape=[1,1,3,3], value=np.eye(3).reshape(1,1,3,3), dtype=tf.float32)
    bias = tf.constant(shape=[3], value=[ele * rgb_range * sign for ele in rgb_mean], dtype=tf.float32)
    
    return tf.nn.conv2d(x, kernel, strides=[1,1,1,1], padding='SAME') + bias


def Base_Conv(x, kernel_size, num_features_input, num_features_output, activation=None, strides=1):

    n = 1 / np.sqrt(kernel_size**2 * num_features_input) # Xavier/Glorot = U[-1/sqrt(N), 1/sqrt(N)]

    x = Conv2D(num_features_output, 
               (kernel_size,kernel_size), 
               padding='same', 
               strides=strides,
               activation=activation,
               kernel_initializer=tf.initializers.RandomUniform(minval=-n, maxval=n),
               bias_initializer=tf.initializers.RandomUniform(minval=-n, maxval=n))(x)  

    # x = Conv2D(num_features_output, 
    #            (kernel_size,kernel_size), 
    #            padding='same', 
    #            strides=strides,
    #            activation=activation,
    #            kernel_initializer=tf.initializers.constant(0.001),
    #            bias_initializer=tf.initializers.constant(0.001))(x)

    return x


# Channel Attention (CA) Layer
def Channel_Attention(input, num_features_input, num_features_output, reduction=16):

    # Following the RCAN unofficial implementation on tensorflow 1.13
    # This operation is equivalent to GlobalPooling (in paper) --- Adaptive Average Pooling 2D in PyTorch
    x = tf.reduce_mean(input, axis=[1,2], keepdims=True)
    
    x = Base_Conv(x, 1, num_features_input, num_features_output // reduction, activation='relu')

    x = Base_Conv(x, 1, num_features_output // reduction, num_features_output, activation='sigmoid')

    x = Multiply()([x, input])

    return x


# Residual Channel Attention Block (RCAB)
def RCA_Block(input, kernel_size, num_features, reduction, normalization=False):

    _res = input

    num_features_aux = num_features // 3

    x1 = Base_Conv(input, kernel_size, num_features, num_features_aux)
    x2 = Base_Conv(input, kernel_size+2, num_features, num_features_aux)
    x3 = Base_Conv(input, kernel_size+4, num_features, num_features_aux)

    x = concatenate([x1, x2, x3], axis=-1)
    x = Base_Conv(x, kernel_size, num_features_aux*3, num_features)

    if normalization:
        x = BatchNormalization()(x)
    x = ReLU()(x)

    x1 = Base_Conv(x, kernel_size, num_features, num_features_aux)
    x2 = Base_Conv(x, kernel_size+2, num_features, num_features_aux)
    x3 = Base_Conv(x, kernel_size+4, num_features, num_features_aux)

    x = concatenate([x1, x2, x3], axis=-1)
    x = Base_Conv(x, kernel_size, num_features_aux*3, num_features)

    if normalization:
        x = BatchNormalization()(x)

    x = Channel_Attention(x, num_features, num_features, reduction)

    x = Add()([x, _res])

    return x


# Residual Group (RG)
def Residual_Group(input, num_residual_blocks, kernel_size, num_features, reduction, normalization=False):

    x = input

    for i in range(num_residual_blocks):
        x = RCA_Block(x, kernel_size, num_features, reduction, normalization)
    
    # Add one last convolutional layer/block
    x = Base_Conv(x, kernel_size, num_features, num_features)

    # Add skip connection
    x = Add()([x, input])

    return x


# Upscaling method (Sub-Pixel Convolution / Pixel-Shuffle)
def Upscaling_Block(input, kernel_size, num_features, scale):

    x = input

    # If scale can be represented as a 2^n expression (x2, x4, x8)
    aux = math.log(scale, 2)

    if aux.is_integer():

        for i in range(int(aux)):

            # Sub-Pixel convolution: for each feature map, we need r² feature maps
            # where 'r' is the upscaling factor

            x = Base_Conv(x, kernel_size, num_features, 4*num_features)
            x = tf.nn.depth_to_space(x, 2)

    else:
        x = Base_Conv(x, kernel_size, num_features, scale**2*num_features)
        x = tf.nn.depth_to_space(x, scale)

    return x


def get_RCAN(num_residual_groups, 
             num_residual_blocks, 
             num_features, 
             kernel_size, 
             reduction, 
             num_channels, 
             scale, 
             normalization=False,
             training_loop=keras.Model):

    # Input layer
    input = Input(shape=(None, None, num_channels))

    # Mean Shift
    x = Mean_Shifter(input, RGB_MEAN, sign=-1)

    # Head module
    x = Base_Conv(x, kernel_size, num_channels, num_features)

    # Save long skip connection 
    long_skip = x

    for _ in range(num_residual_groups):
        x = Residual_Group(x, num_residual_blocks, kernel_size, num_features, reduction, normalization)
    
    x = Base_Conv(x, kernel_size, num_features, num_features)

    # Add long skip connection
    x = Add()([x, long_skip])

    # Upscaling block
    x = Upscaling_Block(x, kernel_size, num_features, scale)

    x = Base_Conv(x, kernel_size, num_features, num_channels, strides=1)

    # Mean Shift
    x = Mean_Shifter(x, RGB_MEAN, sign=1)

    model_args = {'inputs': input, 'outputs': x}

    return training_loop(**model_args)



