# Author: Ankit Kariryaa, University of Bremen.
# Modified by Beihui Hu

import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        self.base_dir = r'D:\lakemapping\U_Net'
        self.dataset_dir=os.path.join(self.base_dir,'patchesReshape')
        self.image_type = '.png'       
        self.NDWI_fn = 'ndwi'
        self.red_fn = 'red'
        self.blue_fn = 'blue'
        self.green_fn = 'green'
        self.swir_fn = 'swir'
        self.annotation_fn = 'annotation'
        self.type_num=7
        self.patch_size = (512,512,3) # Height * Width * (Input or Output) channels  
        self.patch_dir = os.path.join(self.base_dir,'patches{}'.format(self.patch_size[0])) 
        self.step_size = (512,512)# # When stratergy == sequential, then you need the step_size as well
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio.
        self.test_ratio = 0.2
        self.val_ratio = 0.25 
        
        # Probability with which the generated patches should be normalized  0 -> don't normalize,    1 -> normalize all 
        self.normalize = 0
        # Shape of the input data, height*width*channel; Here channel is NDWI
        self.input_shape = (512,512,2)
        self.input_image_channel = [0,1]
        self.input_label_channel = [2]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 16
        self.NB_EPOCHS = 50

        # number of validation images to use
        self.VALID_IMG_COUNT = 130       
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 390 #steps_per_epoch=(num_train/batch_size)
        self.model_path = os.path.join(self.base_dir, 'saved_models/UNet') 