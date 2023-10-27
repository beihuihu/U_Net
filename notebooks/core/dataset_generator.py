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
        iaa.Flipud(0.2),  # vertically flip 20% of all images 
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
#         sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
#         iaa.Add(value=(-0.5,0.5),per_channel=True),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """
    def __init__(self, input_image_channel, patch_size, frames, annotation_channel = [1], augmenter=None):
        """Datagenerator constructor
        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.   #input_image_channel = [0]
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class. 
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined in imageAugmentationWithIAA function.
            annotation_channel:annotation
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter
        
    # Return all training and label images, generated sequentially with the given step size
    def all_sequential_patches(self, step_size):
        """Generate all patches from all assigned frames sequentially.

            step_size (tuple(int,int)): Size of the step when generating frames.
        """
        seq = imageAugmentationWithIAA()
            
        patches = []
        for frame in self.frames:
            ps= frame.sequential_patches(self.patch_size, step_size)
            patches.extend(ps)
        data = np.array(patches)
        img = data[..., self.input_image_channel]#self.input_image_channel = [0]
        ann = data[..., self.annotation_channel]#[1]
        if self.augmenter == 'iaa':   #augmenter = 'iaa'  
                seq_det = seq.to_deterministic()
                img = seq_det.augment_images(img)
    
                # ann would have one channel, i.e. annotations. We need to augmenty for operations such as crop and transform
                ann = seq_det.augment_images(ann) 
                # Some augmentations can change the value of y, so we re-assign values just to be sure.
                ann =  ann[...,[0]]
                ann_joint=ann
                return (img, ann_joint)
        else:
    
                ann =  ann[...,[0]]
                ann_joint=ann
                return (img, ann_joint)

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE):
        """Generate patches from random location in randomly chosen frames.
        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently). 8
        """
        patches = []
#         count=0
#         while count<BATCH_SIZE:
#             frame = random.sample(self.frames,1)
#             patch = frame[0].random_patch(self.patch_size, normalize)
#             if patch[1].any()>0:
#                 patches.append(patch)
#                 count=count+1
        for i in range(BATCH_SIZE):
            frame = np.random.choice(self.frames)
            patch = frame.random_patch(self.patch_size)
            patches.append(patch)
        data = np.array(patches)
        img = data[..., self.input_image_channel]#self.input_image_channel = [0]
        ann_joint = data[..., self.annotation_channel]#[1]
        return (img, ann_joint)
    
    def random_generator(self, BATCH_SIZE, normalize = 0):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.
        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).  
            normalize (float): Probability with which a frame is normalized.
        """
        seq = imageAugmentationWithIAA()

        while True:
            X, y = self.random_patch(BATCH_SIZE)
            if self.augmenter == 'iaa':   #augmenter = 'iaa'  
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
    
                # y would have one channel, i.e. annotations. We need to augmenty for operations such as crop and transform
                y = seq_det.augment_images(y) 
                # Some augmentations can change the value of y, so we re-assign values just to be sure.
                ann =  y[...,[0]]
#                 ann[ann<0.5] = 0
#                 ann[ann>=0.5] = 1
                
                ann_joint=ann
                yield X, ann_joint
            else:
    
                ann =  y[...,[0]]
                ann_joint=ann
                yield X, ann_joint