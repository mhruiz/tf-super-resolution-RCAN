from . import data_processing_functions as data_fns
from .custom_metric_functions import *
from .custom_loss_functions import *
from . import constants

import tensorflow as tf

from importlib import reload
reload(data_fns)

class PrepareDataset():
    '''
    This class will make easier loading and processing datasets. And it will also
    permit the user to get loss and metric functions adapted to image shapes.
    '''
    
    def __init__(self, 
                 dataframes, 
                 channel_mode=constants.COLOR_MODE.Y, 
                 scale=3,
                 normalize=True,
                 batch_size=1,
                 patch_size=None, 
                 data_augmentation=False,
                 shuffle=False,
                 repeat=None):
        
        
        # First, get all image paths from DataFrames
        paths_hr = []
        for df in dataframes:
            paths_hr.extend(df['path'].tolist())
            
        # Create path to LR images
        # These images will be stored in 'data/lr_images/...'

        paths_lr = []
        for x in paths_hr:
            directories = x.split('/')
            new_path = [directories[0]] + ['lr_images'] + directories[1:]
            paths_lr.append('/'.join(new_path))
        
        # Create dataset of image paths
        self.dataset = tf.data.Dataset.from_tensor_slices((paths_lr, paths_hr))
        
        # Apply shuffle
        # Applying shuffle before loading images is better for memmory usage
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=tf.data.experimental.cardinality(self.dataset))
        
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        # Auxiliar function
        def auxiliar_function(func, img_lr, img_hr):
            return func(img_lr), func(img_hr)

        # Load images on specified space color
        if channel_mode == constants.COLOR_MODE.Y:
            self.dataset = self.dataset.map(lambda x, y: auxiliar_function(data_fns.load_Y_images, x, y), num_parallel_calls=AUTOTUNE)
        elif channel_mode == constants.COLOR_MODE.RGB:
            self.dataset = self.dataset.map(lambda x, y: auxiliar_function(data_fns.load_RGB_images, x, y), num_parallel_calls=AUTOTUNE)
        else:
            raise Exception('Unknown space color given: ' + str(channel_mode))
        
        
        # Extract patches
        if patch_size is not None:
            self.dataset = self.dataset.map(lambda x, y: data_fns.get_patch_from_pair_of_images(x, y, patch_size, scale))

        # Apply data augmentation if needed
        if data_augmentation:
            self.dataset = self.dataset.map(lambda x, y: data_fns.apply_data_augmentation_on_pair(x, y), num_parallel_calls=AUTOTUNE)

        self.normalized = normalize

        if self.normalized:
            # Normalize images to [0, 1]
            self.dataset = self.dataset.map(lambda x, y: auxiliar_function(data_fns.normalize_to_0_1, x, y), num_parallel_calls=AUTOTUNE)
        else:
            # Cast to float
            self.dataset = self.dataset.map(lambda x, y: auxiliar_function(data_fns.cast_to_float, x, y), num_parallel_calls=AUTOTUNE)
        
        # Apply repeat
        if repeat is not None:
            self.dataset = self.dataset.repeat(repeat)

        # Divide in batches
        self.dataset = self.dataset.batch(batch_size)

        self.dataset = self.dataset.prefetch(1)
        
        # Save others arguments
        self.num_channels = 1 if channel_mode == constants.COLOR_MODE.Y else 3
        
        self.scale = scale
        
    
    def get_loss_function(self, loss_name):
        return get_loss(loss_name)
    
    def get_metric_function(self, metric_name):
        min_value, max_value = (0, 1) if self.normalized else (0, 255)
        return get_metric(metric_name, min_val=min_value, max_val=max_value)