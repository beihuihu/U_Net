# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi and Qiuqi Luo

import os

# Configuration of the parameters for the 3-FinalRasterAnalysis.ipynb notebook
class Configuration:
    '''
    Configuration for the notebook where objects are predicted in the image.
    Copy the configTemplate folder and define the paths to input and output data.
    '''
    def __init__(self):
        
        self.input_image_dir = r'D:\lakemapping\4_predition\val'
        self.input_image_type = '.png'
        self.ndwi_fn_st = 'ndwi'
        self.red_fn_st = 'red'
        self.blue_fn_st = 'blue'
        self.green_fn_st = 'green'
        self.swir_fn_st = 'swir'
        self.type_num=6
        self.band_num=1
        self.trained_model_path = r'D:\lakemapping\U_Net\saved_models\UNet\lakes_20231101-2140_AdaDelta_dice_loss_01_512.h5'
        print('self.trained_model_path:', self.trained_model_path)
        
        self.output_image_type = '.tif'
        self.output_dir = r'D:\lakemapping\4_predition\1band\validation'
        
        self.output_prefix = 'predition_b1_'  
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files =False
        self.output_dtype='uint8'

        # Variables related to batches and model
        self.BATCH_SIZE =16# Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=512 # Should be same as the WIDTH used for training the model
        self.HEIGHT=512 # Should be same as the HEIGHT used for training the model
        self.STRIDE=256 # STRIDE = WIDTH   means no overlap；  STRIDE = WIDTH/2   means 50 % overlap in prediction 