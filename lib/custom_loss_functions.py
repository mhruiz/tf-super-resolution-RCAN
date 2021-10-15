'''
These are the different loss functions that can be used with the models.

These functions will have this structure: fn(y_true, y_pred).

'''

from . import constants

import tensorflow.keras.backend as K
import tensorflow as tf


def mse_loss(y_true, y_pred, *args):
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def mae_loss(y_true, y_pred, *args):
    return tf.reduce_mean(tf.abs(tf.math.subtract(y_true, y_pred)))


# https://stackoverflow.com/questions/47346615/how-to-implement-custom-sobel-filter-based-loss-function-using-keras
# No se puede utilizar tf.image.sobel_edges. No es diferenciable
# Hay que recurrir a funciones de tf.math, tf.nn o tf.keras.backend
'''
def compute_sobel_gradients(image):
    # Recevives a 4D Tensor with shape: [b, h, w, d]
    sobel = tf.image.sobel_edges(image)
    
    # tf.image.sobel_edges outputs the same input with a new 5th dimmension, 
    # where 0 is horizontal gradients and 1 is vertical gradients
    sobel_h = sobel[...,0]
    sobel_v = sobel[...,1]
    # sobel_h, sobel_v = tf.split(sobel, num_or_size_splits=2, axis=4)
    
    # Sobel_h ---> Gy
    # Sobel_v ---> Gx
    # G = sqrt(Gx^2 + Gy^2)
    gradients = tf.math.sqrt(tf.math.add(tf.math.square(sobel_h), tf.math.square(sobel_v)))
    
    return gradients

# Mean Squared Gradient Error
def msge_loss(y_true, y_pred, *args):
    
    # Get Sobel gradients
    true_gradients = compute_sobel_gradients(y_true)
    pred_gradients = compute_sobel_gradients(y_pred)
    
    return tf.reduce_mean(tf.math.squared_difference(true_gradients, pred_gradients))

# Mean Absolute Gradient Error
def mage_loss(y_true, y_pred, *args):
    
    # Get Sobel gradients
    true_gradients = compute_sobel_gradients(y_true)
    pred_gradients = compute_sobel_gradients(y_pred)
    
    return tf.reduce_mean(tf.abs(tf.math.subtract(true_gradients, pred_gradients)))
'''


# Sobel loss function got from:
# https://stackoverflow.com/questions/47346615/how-to-implement-custom-sobel-filter-based-loss-function-using-keras

# Base sobel filter for image shape [b, h, w, 1]
sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                  [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                  [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])

def get_sobel_filter(input_tensor):

    #this considers data_format = 'channels_last'
    channels = K.reshape(K.ones_like(input_tensor[0,0,0,:]),(1,1,-1,1))

    # Repeat sobel filter channels times
    return sobelFilter * channels

# Creo que funciona mejor como métrica que como loss
def sobel_loss(y_true, y_pred, *args):

    # Get the sobel filter repeated for each input channel
    filt = get_sobel_filter(y_true)

    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobel_y_true = K.depthwise_conv2d(y_true, filt) # No padding
    sobel_y_pred = K.depthwise_conv2d(y_pred, filt) # No padding

    #now you just apply the mse:
    return K.mean(K.square(sobel_y_true - sobel_y_pred))


# Create loss function adapted to image shapes
def get_loss(loss_name):
    
    if loss_name == constants.LOSS_FUNCTIONS.MSE:
        loss_fn = mse_loss
        
    elif loss_name == constants.LOSS_FUNCTIONS.MAE:
        loss_fn = mae_loss
        
    elif loss_name == constants.LOSS_FUNCTIONS.SOBEL:
        loss_fn = sobel_loss
        
    else:
        raise Exception('Please provide an implemented loss function')
    
    # Create loss function
    def fn(y_true, y_pred):
        return loss_fn(y_true, y_pred)
    
    # Change the name of the newly created metric function
    fn.__name__ = loss_name
    
    return fn

# Create a custom loss function by joining 2 different functions
def get_mix_loss_function(fn_1, weight_1, fn_2, weight_2):
    
    def fn(y_true, y_pred):
        return fn_1(y_true, y_pred) * weight_1 + fn_2(y_true, y_pred) * weight_2
    
    # Change the name of the newly created loss function
    fn.__name__ = 'mixed_' + fn_1.__name__ + '_' + fn_2.__name__
    
    return fn


'''
# INTENTO #########################################################################################################
# Mezclando codigo de pagina de MGE: https://medium.com/analytics-vidhya/loss-functions-for-image-super-resolution-sisr-8a65644fbd85
# Con codigo de pagina: https://www.programmersought.com/article/60761520090/

def MGE(outputs, targets):
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    # Given an input tensor of shape batch_shape + [in_height, in_width, in_channels] 
    # and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], 
    # this op performs the following:

        # Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
    
        # Extracts image patches from the input tensor to form a virtual tensor of shape 
        # [batch, out_height, out_width, filter_height * filter_width * in_channels].
    
        # For each patch, right-multiplies the filter matrix and the image patch vector.
    
    def get_sobel_gradients(image):
        
        # Define Sobel kernels
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    
        sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])

        sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
        
        # Apply kernels, with 'zero' as padding (SAME)
        filtered_x = tf.nn.conv2d(image, 
                                  sobel_x_filter,
                                  strides=[1, 1, 1, 1], 
                                  padding='SAME')
        
        filtered_y = tf.nn.conv2d(image, 
                                  sobel_y_filter,
                                  strides=[1, 1, 1, 1], 
                                  padding='SAME')

        # Squared gradients
        squared_gx = tf.math.square(filtered_x)
        squared_gy = tf.math.square(filtered_y)
    
        # Sum and sqrt
        abs_gradient = tf.math.sqrt(tf.math.add(squared_gx, squared_gy))
        
        return abs_gradient, filtered_x, filtered_y
    
    predicted_gradients,_,_ = get_sobel_gradients(outputs)
    
    real_gradients,_,_ = get_sobel_gradients(targets)

    # Compute mean gradient error
    shape = targets.shape

    mge = tf.reduce_mean(tf.math.squared_difference(predicted_gradients, real_gradients))

    return mge

def MixMAE_MGE(y_pred, y_true, weight=0.1):
    
    # MAE Error
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # MGE Error, with 0.1 weight
    mge_loss = MGE(tf.image.rgb_to_grayscale(y_pred), 
                   tf.image.rgb_to_grayscale(y_true))
    
    return mae_loss + mge_loss * weight
'''