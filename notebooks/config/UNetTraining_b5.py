# Author: Ankit Kariryaa, University of Bremen.
# Modified by Beihui Hu

import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the NDWI and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        
        # self.base_dir = r'D:\lakemapping\U_Net'
        # self.dataset_dir=r'D:\lakemapping\2_dataset\patchesReshape600'
        # self.dataset_dir=os.path.join(self.base_dir,'patchesReshape')
        self.base_dir = r'/home/nkd/hbh/U_Net'
        self.dataset_dir=r'/home/nkd/hbh/patchesReshape'
        self.image_type = '.png'       
        self.NDWI_fn = 'ndwi'
        self.red_fn = 'red'
        self.blue_fn = 'blue'
        self.green_fn = 'green'
        self.swir_fn = 'swir'
        self.annotation_fn = 'annotation'
        self.type_num=6
        self.patch_size = (512,512,6) # Height * Width * (Input or Output) channels  
        self.step_size = (512,512)# # When stratergy == sequential, then you need the step_size as well
        
        # Probability with which the generated patches should be normalized  0 -> don't normalize,    1 -> normalize all 
        self.normalize = 0
        # Shape of the input data, height*width*channel; Here channel is NDWI
        self.input_shape = (512,512,5)
        self.input_image_channel = [0,1,2,3,4]
        self.input_label_channel = [5]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 32
        self.NB_EPOCHS = 150

        # number of validation images to use
        self.VALID_IMG_COUNT = 67#186       
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 217#534 #steps_per_epoch=(num_train/batch_size)
        self.model_path = r'/home/nkd/hbh/saved_models'
        # self.model_path = os.path.join(self.base_dir, 'saved_models/UNet') 