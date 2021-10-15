'''
These are all mapping functions available to load and preprocess the different
datasets
'''

import imgaug.augmenters as iaa
import tensorflow as tf


def load_RGB_images(path_img):
    return tf.image.decode_bmp(tf.io.read_file(path_img), channels=3)


def py_convert_to_YCrCb(image):
    return iaa.color.ChangeColorspace(to_colorspace='YCrCb', 
                                      from_colorspace='RGB').augment_image(image.numpy())


def load_Y_images(path_img):
    image = tf.image.decode_bmp(tf.io.read_file(path_img), channels=3)
    
    # Convert to YCrCb space color -> Py_function
    # It's importat to get the image shape before process it -- TensorFlow website
    im_shape = image.shape 
    
    # tf.py_function wraps a python function into a TensorFlow op that executes it eagerly
    [image, ] = tf.py_function(py_convert_to_YCrCb, [image], [image.dtype])
    
    # Recover shape -- As indicated in TensorFlow website
    image.set_shape(im_shape)
    
    # Split in channels and get first
    Y, _, _ = tf.split(image, num_or_size_splits=3, axis=2)
    
    return Y


data_augmenter = iaa.Sequential([
    iaa.flip.Fliplr(0.5),
    iaa.flip.Flipud(0.5),
    iaa.geometric.Rot90([0, 1, 2, 3], keep_size=False)
])


def apply_data_augmentation_on_pair(img_lr, img_hr):

    # PENDIENTE DE ARREGLAR

    # def py_apply_data_augmentation(img_lr, img_hr):

    #     # Make the augmentation sequence deterministic so it will apply the same operations to both images
    #     aug = data_augmenter.to_deterministic()

    #     img_lr_aug = aug.augment_image(img_lr.numpy())
    #     img_hr_aug = aug.augment_image(img_hr.numpy())

    #     return img_lr_aug, img_hr_aug

    # # Get image dtype
    # im_dtype = img_hr.dtype

    # # Get images shapes
    # im_lr_shape = img_lr.shape
    # im_hr_shape = img_hr.shape

    # # Excecute python function inside tf.py_function
    # [img_lr_aug, img_hr_aug, ] = tf.py_function(py_apply_data_augmentation, [img_lr, img_hr], [im_dtype, im_dtype])

    # # Recover shape
    # img_lr_aug.set_shape(im_lr_shape)
    # img_hr_aug.set_shape(im_hr_shape)


    # Data augmentation original de la implementación en TF 1.13
    rot = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    flip_lr = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
    flip_ud = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)

    # Random rotation -- same for both LR and HR images
    img_lr_aug = tf.image.rot90(img_lr, rot)
    img_hr_aug = tf.image.rot90(img_hr, rot)

    if flip_lr == 0:
        img_lr_aug = tf.image.flip_left_right(img_lr_aug)
        img_hr_aug = tf.image.flip_left_right(img_hr_aug)
    
    if flip_ud == 0:
        img_lr_aug = tf.image.flip_up_down(img_lr_aug)
        img_hr_aug = tf.image.flip_up_down(img_hr_aug)

    return img_lr_aug, img_hr_aug


def normalize_to_0_1(img):
    # Convert to float32 and multiply by 1/255 -> [0, 1]
    img = tf.multiply(tf.cast(img, tf.float32), tf.constant([1.0/255.0]))

    return img


def cast_to_float(img):
    return tf.cast(img, tf.float32)


'''
Functions to get LR and HR image pairs
'''

def get_patch_from_pair_of_images(img_lr, img_hr, patch_size, scale):

    im_shape = tf.shape(img_lr)

    ih = im_shape[0]
    iw = im_shape[1]

    ix = tf.random.uniform([], minval=0, maxval=iw-patch_size, dtype=tf.int32)
    iy = tf.random.uniform([], minval=0, maxval=ih-patch_size, dtype=tf.int32)

    tx = ix * scale
    ty = iy * scale

    patch_size_hr = patch_size * scale

    patch_lr = img_lr[iy:iy+patch_size, ix:ix+patch_size]

    patch_hr = img_hr[ty:ty+patch_size_hr, tx:tx+patch_size_hr]

    return patch_lr, patch_hr

