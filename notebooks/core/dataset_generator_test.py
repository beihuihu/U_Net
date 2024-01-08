#   Author: Ankit Kariryaa, University of Bremen
  
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import rasterio
import random

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5  : iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images 
        iaa.Flipud(0.5),  # vertically flip 20% of all images 
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
#         sometimes(iaa.PiecewiseAffine(0.05), 0.3),
#         sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """
    def __init__(self, input_size,output_size,patch_size, frames, augmenter=None):
        """Datagenerator constructor
        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.   #input_image_channel = [0]
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class. 
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined in imageAugmentationWithIAA function.
            annotation_channel:annotation
        """
        self.input_size = input_size
        self.output_size= output_size
        self.patch_size= patch_size
        self.frames = frames
        self.augmenter = augmenter
        
    # Return all training and label images, generated sequentially with the given step size
    def all_sequential_patches(self, step_size):
        """Generate all patches from all assigned frames sequentially.
            step_size (tuple(int,int)): Size of the step when generating frames.
        """
        seq = imageAugmentationWithIAA()
        img_patches = []
        ann_patches = []
        for frame in self.frames:
            ips,aps= frame.sequential_patches(self.input_size,self.output_size, self.patch_size,step_size)
            img_patches.extend(ips)
            ann_patches.extend(aps)
        img = np.array(img_patches)
        ann = np.array(ann_patches)
        if self.augmenter == 'iaa':   #augmenter = 'iaa'  
            seq_det = seq.to_deterministic()
            img = seq_det.augment_images(img)
            ann = seq_det.augment_images(ann) 
        return img, ann

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE,percentages):
        """Generate patches from random location in randomly chosen frames.
        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently). 8
        """
        img_patches = []
        ann_patches = []
        for i in range(BATCH_SIZE):
            frame = np.random.choice(self.frames,p=percentages)
            img_patch, ann_patch = frame.random_patch(self.input_size,self.output_size,self.patch_size)
            img_patches.append(img_patch)
            ann_patches.append(ann_patch)
        img = np.array(img_patches)
        ann = np.array(ann_patches)
        return img, ann
    
    def random_generator(self, BATCH_SIZE,percentages = None, normalize = 0):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.
        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).  
            normalize (float): Probability with which a frame is normalized.
        """
        seq = imageAugmentationWithIAA()

        while True:
            X, y = self.random_patch(BATCH_SIZE,percentages)
            if self.augmenter == 'iaa':   #augmenter = 'iaa'  
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                y = seq_det.augment_images(y) 
            yield X, y