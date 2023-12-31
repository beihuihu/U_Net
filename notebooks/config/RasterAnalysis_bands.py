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
        
        # self.input_image_dir = r'D:\lakemapping\4_predition\test_img'
        self.input_image_dir = r'I:\new_image'
        self.input_image_type = '.tif'
        self.ndwi_fn_st =  'ndwi_int8'
        self.bands_fn_st =  'bands_int16'
        
        self.type_num=6
        self.band_num=5
        self.ignore_edge_width=100
        # self.trained_model_path = r'D:\lakemapping\U_Net\saved_models\UNet\lakes_20231111-1239_AdaDelta_dice_loss_0123_512_percentages.h5'
        self.trained_model_path = r'/home/nkd/hbh/saved_models/lakes_20240101-2240_AdaDelta_dice_loss_012345_576.h5'
        print('self.trained_model_path:', self.trained_model_path)
        
        self.output_image_type = '.tif'
        # self.output_dir = r'D:\lakemapping\4_predition\sample588\3band\padding'
        self.output_dir = r'I:\result\tif'
        
        self.output_prefix = 'predition_b5_'  
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files =False
        self.output_dtype='uint8'

        # Variables related to batches and model
        self.BATCH_SIZE =16# Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=576 # Should be same as the WIDTH used for training the model
        self.HEIGHT=576 # Should be same as the HEIGHT used for training the model
        self.STRIDE=288 # STRIDE = WIDTH   means no overlap；  STRIDE = WIDTH/2   means 50 % overlap in prediction 