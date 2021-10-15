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
TRAINING_REPEAT = None # ----------------- Según la implementación en TF 1.13, se valida cada 10.000 training steps. Al ser 800 imágenes de entrenamiento y un batch de 16, hay 50 steps por época
                      #                   Hay que realizar 10.000 iteraciones antes de validar. Estas son equivalentes a 200 épocas de entrenamiento sin validación, osea, repetir 200 veces el dataset
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

FILENAME = __file__.split('.')[0]

# Get model info from file name
info_from_script_name = FILENAME.split('_') 

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
NUM_RESIDUAL_GROUPS = 2
NUM_RESIDUAL_BLOCKS = 6
NUM_FEATURES = 64
KERNEL_SIZE = 3
REDUCTION = 16
NUM_IMAGE_CHANNELS = TRAINING_COLOR_MODE
SCALE = TRAINING_SCALE
NORMALIZATION = False

model = models.get_RCAN(NUM_RESIDUAL_GROUPS,
                        NUM_RESIDUAL_BLOCKS,
                        NUM_FEATURES,
                        KERNEL_SIZE,
                        REDUCTION,
                        NUM_IMAGE_CHANNELS,
                        SCALE,
                        NORMALIZATION)

# Load weights
# model.load_weights('TRAINED_MODELS_NEW/RCAN-SCALE3/model_replica_original_best_ssim.h5')

# model.summary()
types = {
    'Conv': tf.keras.layers.Conv2D,
    'ReLU': tf.keras.layers.ReLU,
    'Add': tf.keras.layers.Add 
}

total_num = 0

for k in types:
    num = len(list(filter(lambda x: type(x) == types[k], model.layers)))
    print('   Num', k, 'layers:', num)
    total_num += num
print('Total layers:', total_num)

print('Num layers:', len(model.layers))
print('Num parameters:', model.count_params())

print('Num trainable variables:', sum(len(l.trainable_variables) for l in model.layers))

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

NUM_EPOCHS = 100 # Nº máximo de iteraciones de entrenamiento: 1.000.000 -- Como antes definimos una época como 10.000 iteraciones (repetir 200 veces el dataset de 800 imágenes dividio en batches de 16)
                 # 1.000.000 de iteraciones serán 100 épocas

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
LOSS_FUNCTION = train.get_loss_function(ctes.LOSS_FUNCTIONS.MAE)

# Define callbacks

# Se indica en el paper original, y se aplica en la implementación en TF 1.13, que cada 200.000 iteraciones, el learning rate se debe reducir a la mitad
# Estableciendo que una época se compone de 200 repeticiones del dataset de entrenamiento para conseguir 10.000 iteraciones, cada 20 'épocas' se deberá reducir a la mitad el learning rate

learning_rate_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(callbacks.create_scheduler_function(20, 0.5)) # Cada 20 épocas, multiplicar lr por 0.5

# Model checkpoints callbacks
checkpoint_callbacks = [tf.keras.callbacks.ModelCheckpoint(save_path + model_name + '_best_' + mtr[0] + '.h5',
                                                           monitor=mtr[1],
                                                           save_best_only=True,
                                                           mode=mtr[2],
                                                           save_weights_only=True) for mtr in METRICS_CHECKPOINTS]

# Save training metrics evolution callback
metrics_evolution_callback = callbacks.Save_Training_Evolution(save_metrics_path + model_name_for_metrics + '_evolution.csv')


CBACKS = [learning_rate_scheduler_callback, checkpoint_callbacks, metrics_evolution_callback] 


# TRAIN
model.compile(optimizer=OPTIMIZER,
              loss=LOSS_FUNCTION,
              # SSIM_MS needs greater images than training patches, so ignore this metric on training
              metrics=[train.get_metric_function(x) for x in METRICS if x != ctes.METRIC_FUNCTIONS.SSIM_MS])

print('Starts training')
model.fit(train.dataset, epochs=NUM_EPOCHS, verbose=1, validation_data=val.dataset, callbacks=CBACKS)
print('Ends training')


with open('ended_scripts.txt', 'a') as f:
    f.write(FILENAME + '\n')