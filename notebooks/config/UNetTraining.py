# Author: Ankit Kariryaa, University of Bremen.
# Modified by Beihui Hu

import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the NDWI and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        
        self.base_dir = r'D:\lakemapping'
        self.dataset_dir=os.path.join(self.base_dir,'2_dataset\patchesReshape')
#         self.base_dir = r'/home/nkd/hbh'
#         self.dataset_dir=os.path.join(self.base_dir,'patchesReshape')
#         self.dataset_dir=r'/home/nkd/hbh/patchesReshape'
        self.image_type = '.tif'       
        self.ann_type = '.png'
        self.annotation_fn = 'annotation'
        self.image_fn = 'image'
        self.type_num=4
        self.patch_size = (576,576,6) # Height * Width * (Input or Output) channels  
        self.step_size = (576,576)# # When stratergy == sequential, then you need the step_size as well
        self.input_shape = (576,576,5)
        
        # Probability with which the generated patches should be normalized  0 -> don't normalize,    1 -> normalize all 
        self.normalize = 0
        # Shape of the input data, height*width*channel; Here channel is NDWI
        self.input_image_channel = [0,1,2,3,4]
        self.input_label_channel = [5]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 16
        self.NB_EPOCHS = 150

#         self.model_path = r'/home/nkd/hbh/saved_models'
        self.model_path = os.path.join(self.base_dir, '5_saved_models') 