import tensorflow as tf
import pandas as pd
import cv2
import os

# GPU memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Import own modules
import lib.custom_callbacks as callbacks
import lib.PrepareDataset2 as dt
import lib.constants as ctes
import lib.evaluation as ev
import lib.models as models

import lib.custom_loss_functions as loss_fns

######################################################################################
# DEFINE ARGUMENTS

# Load datasets from CSV files
TRAINING_DATASETS = [pd.read_csv('data/0_csvs/DIV2K_train_HR.csv')]
VALIDATION_DATASETS = [pd.read_csv('data/0_csvs/DIV2K_valid_HR.csv')]

# Define dataset parameters
TRAINING_COLOR_MODE = ctes.COLOR_MODE.RGB
TRAINING_SCALE = 3
TRAINING_BATCH_SIZE = 16
TRAINING_PATCH_SIZE = 48
TRAINING_DATA_AUG = True
TRAINING_SHUFFLE = True
TRAINING_REPEAT = None
TRAINING_NORMALIZE = False
TRAINING_REDUCED_LR = True

VALIDATION_COLOR_MODE = TRAINING_COLOR_MODE
VALIDATION_SCALE = TRAINING_SCALE
VALIDATION_BATCH_SIZE = 1
VALIDATION_CENTRAL_PATCH_SIZE = TRAINING_PATCH_SIZE
VALIDATION_DATA_AUG = False
VALIDATION_SHUFFLE = False
VALIDATION_REPEAT = None
VALIDATION_NORMALIZE = False
VALIDATION_REDUCED_LR = TRAINING_REDUCED_LR


# Get model info from file name
info_from_script_name = __file__.split('_') 

model_structure_name = info_from_script_name[2].upper() 

model_name = info_from_script_name[3:]            
model_name = ('_'.join(['model'] + model_name)).split('.')[0]
model_name_for_metrics = model_structure_name + '_' + model_name         

BASE_MODEL_PATH = 'TRAINED_MODELS_NEW/'
BASE_METRIC_PATH = 'TRAINING_METRIC_EVOLUTIONS_NEW/'

if not os.path.exists(BASE_MODEL_PATH):
    os.mkdir(BASE_MODEL_PATH)
if not os.path.exists(BASE_METRIC_PATH):
    os.mkdir(BASE_METRIC_PATH)

save_path = BASE_MODEL_PATH + model_structure_name + '/'
save_metrics_path = BASE_METRIC_PATH

if not os.path.exists(save_path):
    os.mkdir(save_path)

# Load datasets as PrepareDataset objects
train = dt.PrepareDataset2(dataframes=TRAINING_DATASETS, 
                          channel_mode=TRAINING_COLOR_MODE, 
                          scale=TRAINING_SCALE,
                          batch_size=TRAINING_BATCH_SIZE, 
                          patch_size=TRAINING_PATCH_SIZE,
                          data_augmentation=TRAINING_DATA_AUG, 
                          shuffle=TRAINING_SHUFFLE, 
                          repeat=TRAINING_REPEAT,
                          normalize=TRAINING_NORMALIZE,
                          reduced_lr=TRAINING_REDUCED_LR)

val = dt.PrepareDataset2(dataframes=VALIDATION_DATASETS, 
                        channel_mode=VALIDATION_COLOR_MODE, 
                        scale=VALIDATION_SCALE, 
                        batch_size=VALIDATION_BATCH_SIZE,
                        # Permits to avoid an intensive GPU memory usage
                        fixed_central_patch_size=VALIDATION_CENTRAL_PATCH_SIZE, 
                        data_augmentation=VALIDATION_DATA_AUG,
                        shuffle=VALIDATION_SHUFFLE,
                        repeat=VALIDATION_REPEAT,
                        normalize=VALIDATION_NORMALIZE,
                        reduced_lr=VALIDATION_REDUCED_LR)

print('----------------------------------')
print('Num images for training:', sum(x.shape[0] for x in TRAINING_DATASETS))
print('Num images for validating:', sum(x.shape[0] for x in VALIDATION_DATASETS))
print('----------------------------------')

def aux_fn(datasets, condition_fn):
    return condition_fn(condition_fn(cv2.imread(row.path).shape[1:-1]) for i in range(len(datasets)) for row in datasets[i].itertuples())

print('Smallest image in train dataset:', aux_fn(TRAINING_DATASETS, min))
smallest_validation_size = aux_fn(VALIDATION_DATASETS, min)
print('Smallest image in validation dataset:', smallest_validation_size)
print('----------------------------------')
print('Biggest image in train dataset:', aux_fn(TRAINING_DATASETS, max))
biggest_validation_image = aux_fn(VALIDATION_DATASETS, max)
print('Biggest image in validation dataset:', biggest_validation_image)
print('----------------------------------')


# Define model
NUM_RESIDUAL_GROUPS = 10
NUM_RESIDUAL_BLOCKS = 20
NUM_FEATURES = 64
KERNEL_SIZE = 3
REDUCTION = 16
NUM_IMAGE_CHANNELS = TRAINING_COLOR_MODE
SCALE = TRAINING_SCALE
NORMALIZATION = False

import sys
sys.setrecursionlimit(2000) # Sino salta error de recursion maxima excedida, pero en el notebook no salta.

model = models.get_RCAN(NUM_RESIDUAL_GROUPS,
                        NUM_RESIDUAL_BLOCKS,
                        NUM_FEATURES,
                        KERNEL_SIZE,
                        REDUCTION,
                        NUM_IMAGE_CHANNELS,
                        SCALE,
                        NORMALIZATION)

# Load weights
model.load_weights('TRAINED_MODELS_NEW/RCAN-SCALE3/model_replica_original_best_ssim.h5')

model.summary()

# Save model structure to JSON file
if os.path.exists(save_path + model_name + '.json'):
    os.remove(save_path + model_name + '.json')
with open(save_path + model_name + '.json', 'w') as json_file:
    json_file.write(model.to_json())

# Save model type for current structure
if not os.path.exists(save_path + 'model_type.txt'):
    with open(save_path + 'model_type.txt', 'w') as f:
        f.write('UPSCALE' if train.reduced_lr else 'KEEP_SIZE')

# Get maximum input image size that this model structure can hold
if not os.path.exists(save_path + 'maximum_input_size.txt'):
    with open(save_path + 'maximum_input_size.txt', 'w') as f:
        
        maximum_input_size = ev.get_aproximate_maximum_image_size_for_model(model=model,
                                                                            process_in_parallel=TRAINING_COLOR_MODE==ctes.COLOR_MODE.RGB,
                                                                            use_gpu=True, # Always train with GPU
                                                                            start_size=smallest_validation_size,
                                                                            max_size=biggest_validation_image*4)
        
        f.write(str(maximum_input_size))

        raise Exception('Restart script')

# Metrics for model evaluation
METRICS = [
    ctes.METRIC_FUNCTIONS.PSNR,
    ctes.METRIC_FUNCTIONS.SSIM,
    ctes.METRIC_FUNCTIONS.SSIM_MS, # ---- https://github.com/tensorflow/tensorflow/issues/33840
                                   #      MODIFIED ARGUMENTS IN SSIM MULTISCALE (lib.custom_metric_functions.py)
    ctes.METRIC_FUNCTIONS.MSE,
    ctes.METRIC_FUNCTIONS.MAE,
    ctes.METRIC_FUNCTIONS.SOBEL
]

# Metrics for saving model's checkpoints
METRICS_CHECKPOINTS = [
    (ctes.METRICS_ALL.PSNR, ctes.METRICS_ALL.VAL_PSNR, 'max'),
    (ctes.METRICS_ALL.SSIM, ctes.METRICS_ALL.VAL_SSIM, 'max'),
    # (ctes.METRICS_ALL.SOBEL, ctes.METRICS_ALL.VAL_SOBEL, 'min')
]

# Define training hyperparameters
INITIAL_LEARNING_RATE = 0.0001

LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE, decay_steps=200000, decay_rate=0.5)

NUM_EPOCHS = 20000 # Num_max_steps = 1e6 -- 1 epoch == 800 / 16 (batch) == 50 steps ---> 20.000 epochs
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNCTION = train.get_loss_function(ctes.LOSS_FUNCTIONS.MAE)

# Define callbacks

# Model checkpoints callbacks
checkpoint_callbacks = [tf.keras.callbacks.ModelCheckpoint(save_path + model_name + '_best_' + mtr[0] + '.h5',
                                                           monitor=mtr[1],
                                                           save_best_only=True,
                                                           mode=mtr[2],
                                                           save_weights_only=True) for mtr in METRICS_CHECKPOINTS]

# Save training metrics evolution callback
metrics_evolution_callback = callbacks.Save_Training_Evolution(save_metrics_path + model_name_for_metrics + '_evolution.csv')


CBACKS = checkpoint_callbacks + [metrics_evolution_callback] 


# TRAIN
model.compile(optimizer=OPTIMIZER,
              loss=LOSS_FUNCTION,
              # SSIM_MS needs greater images than training patches, so ignore this metric on training
              metrics=[train.get_metric_function(x) for x in METRICS if x != ctes.METRIC_FUNCTIONS.SSIM_MS])

print('Starts training')
model.fit(train.dataset, epochs=NUM_EPOCHS, verbose=1, validation_data=val.dataset, callbacks=CBACKS)
print('Ends training')


with open('ended_scripts.txt', 'a') as f:
    f.write(__file__ + '\n')