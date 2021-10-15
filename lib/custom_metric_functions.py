'''
These functions permit to create metric functions adapted to dataset shapes.

'''

from .custom_loss_functions import *
from . import constants

import tensorflow as tf


def psnr_without_inf(a, b, max_val):

    psnr_value = tf.image.psnr(a, b, max_val)

    all_100 = tf.ones_like(psnr_value) * 100.

    # Replace all infinite values with 100
    return tf.where(tf.math.is_inf(psnr_value), all_100, psnr_value)


def get_metric(metric_name, min_val=0, max_val=1.0):

    # Choose metric function     
    if metric_name == constants.METRIC_FUNCTIONS.PSNR:
        # metric_fn = tf.image.psnr
        metric_fn = psnr_without_inf

    elif metric_name == constants.METRIC_FUNCTIONS.SSIM:
        metric_fn = tf.image.ssim
        
    elif metric_name == constants.METRIC_FUNCTIONS.SSIM_MS:
        # REASON FOR USING LESS POWER FACTORS: https://github.com/tensorflow/tensorflow/issues/33840#issuecomment-633715778
        # There are some images whose size is lower than 176, which is the minimum size for applying ssim_multiscale with its original arguments
        # There are 2 options:
        #   Use a smaller filter
        #   Use fewer power factors -- I think this is better
        # ORIGINAL POWER FACTORS: https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
        metric_fn = lambda x, y, z: tf.image.ssim_multiscale(x, y, z, power_factors=(0.0448, 0.2856, 0.3001, 0.2363))
        
    elif metric_name == constants.METRIC_FUNCTIONS.MSE:
        metric_fn = mse_loss
        
    elif metric_name == constants.METRIC_FUNCTIONS.MAE:
        metric_fn = mae_loss
    
    elif metric_name == constants.METRIC_FUNCTIONS.SOBEL:
        metric_fn = sobel_loss
        
    else:
        raise Exception('Unknown metric function ' + metric_name)
        
    # Create metric function
    def fn(y_true, y_pred):
        # Clip model's output to valid range [min_val, max_val]
        y_pred_clipped = tf.clip_by_value(y_pred, 
                                          clip_value_min = min_val, 
                                          clip_value_max = max_val)
        
        if max_val == 255 and metric_name != constants.METRIC_FUNCTIONS.SOBEL:
            y_pred_clipped = tf.round(y_pred_clipped)
            y_pred_clipped = tf.cast(y_pred_clipped, tf.int32)

            y_true = tf.cast(y_true, tf.int32)
        
        # Calculate metric
        return metric_fn(y_true, y_pred_clipped, max_val)
    
    # Change the name of the newly created metric function
    fn.__name__ = metric_name
    
    # Return function
    return fn
