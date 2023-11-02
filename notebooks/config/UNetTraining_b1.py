# Author: Ankit Kariryaa, University of Bremen.
# Modified by Beihui Hu

import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        self.base_dir = r'D:\lakemapping\U_Net'
#         self.dataset_dir=os.path.join(self.base_dir,'patchesReshape')
        self.dataset_dir=r'D:\lakemapping\2_dataset\patchesReshape550'
        self.image_type = '.png'       
        self.NDWI_fn = 'ndwi'
        self.annotation_fn = 'annotation'
        self.type_num=6
        self.patch_size = (512,512,2) # Height * Width * (Input or Output) channels  
        self.step_size = (512,512)# # When stratergy == sequential, then you need the step_size as well
        
        # Probability with which the generated patches should be normalized  0 -> don't normalize,    1 -> normalize all 
        self.normalize = 0
        # Shape of the input data, height*width*channel; Here channel is NDWI
        self.input_shape = (512,512,1)
        self.input_image_channel = [0]
        self.input_label_channel = [1]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 16
        self.NB_EPOCHS = 100

        # number of validation images to use
        self.VALID_IMG_COUNT = 260       
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 800 #steps_per_epoch=(num_train/batch_size)
        self.model_path = os.path.join(self.base_dir, 'saved_models/UNet') 