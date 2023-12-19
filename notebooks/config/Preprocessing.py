# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi and Qiuqi Luo

import os

# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_NDWI_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons 
        self.training_base_dir = r'D:\lakemapping\2_dataset'
        self.training_area_fn = r'sampleAnnotations\finalSample\total_area.shp'         
        self.training_polygon_fn = r'sampleAnnotations\finalSample\polygon_type2345.shp' 
        self.type_num=6

        # For reading images
#         self.bands0 = [0]# If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
        self.bands = [0,1,2,3,4,5]
        self.raw_image_base_dir =r'D:\lakemapping\sample750_pad'
        self.raw_image_file_type = '.tif'
        self.raw_image_prefix = 'sample_'
#          self.raw_NDWI_image_prefix = 'ndwi_int8'
#         self.raw_bands_image_prefix = 'bands_int16'
        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.png'
#         self.extracted_NDWI_filename = 'ndwi'
        self.extracted_bands_filename = ['ndwi_','red_','green_','blue_','swir_']
        self.extracted_annotation_filename = 'annotation'
