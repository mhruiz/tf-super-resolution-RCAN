'''
This file contains all constants needed on this project
'''

# Color modes
class COLOR_MODE():
    RGB = 3
    Y = 1

# Loss function names
class LOSS_FUNCTIONS():
    MSE = 'mse'
    MAE = 'mae'
    MIX_MSE_EDGE = 'mse_edge'
    MIX_MAE_EDGE = 'mae_edge'
    SOBEL = 'sobel_loss'

# Metric function names
class METRIC_FUNCTIONS():
    PSNR = 'psnr'
    SSIM = 'ssim'
    SSIM_MS = 'ssim_multiscale'
    MSE = LOSS_FUNCTIONS.MSE
    MAE = LOSS_FUNCTIONS.MAE
    MIX_MSE_EDGE = LOSS_FUNCTIONS.MIX_MSE_EDGE
    MIX_MAE_EDGE = LOSS_FUNCTIONS.MIX_MAE_EDGE
    SOBEL = LOSS_FUNCTIONS.SOBEL

    TIME = 'time(ms)'
    
# All metric names (including validation names)
class METRICS_ALL(METRIC_FUNCTIONS):
    
    LOSS = 'loss'
    VAL_LOSS = 'val_loss'
    
    VAL_PSNR = 'val_' + METRIC_FUNCTIONS.PSNR
    RGB_VAL_PSNR = 'RGB_' + VAL_PSNR
    
    VAL_SSIM = 'val_' + METRIC_FUNCTIONS.SSIM
    RGB_VAL_SSIM = 'RGB_' + VAL_SSIM

    VAL_SSIM_MS = 'val_' + METRIC_FUNCTIONS.SSIM_MS
    RGB_VAL_SSIM_MS = 'RGB_' + VAL_SSIM_MS
    
    VAL_MSE = 'val_' + METRIC_FUNCTIONS.MSE
    RGB_VAL_MSE = 'RGB_' + VAL_MSE
    
    VAL_MAE = 'val_' + METRIC_FUNCTIONS.MAE
    RGB_VAL_MAE = 'RGB_' + VAL_MAE
    
    VAL_SOBEL = 'val_' + METRIC_FUNCTIONS.SOBEL
    RGB_VAL_SOBEL = 'RGB_' + VAL_SOBEL
    
    
    TYPES = {
        LOSS: 1, # The lower the better
        VAL_LOSS: 1,
        
        METRIC_FUNCTIONS.PSNR: 0, # The higher the better
        VAL_PSNR: 0,
        RGB_VAL_PSNR: 0,
        
        METRIC_FUNCTIONS.SSIM: 0,
        VAL_SSIM: 0,
        RGB_VAL_SSIM: 0,
        
        METRIC_FUNCTIONS.SSIM_MS: 0,
        VAL_SSIM_MS: 0,
        RGB_VAL_SSIM_MS: 0,
        
        METRIC_FUNCTIONS.MSE: 1, # The lower the better
        VAL_MSE: 1,
        RGB_VAL_MSE: 1,
        
        METRIC_FUNCTIONS.MAE: 1,
        VAL_MAE: 1,
        RGB_VAL_MAE: 1,
        
        METRIC_FUNCTIONS.SOBEL: 1,
        VAL_SOBEL: 1,
        RGB_VAL_SOBEL: 1,

        METRIC_FUNCTIONS.TIME: 1
    }
    
    VAL_RGB_NAME = 'valRGB'
 