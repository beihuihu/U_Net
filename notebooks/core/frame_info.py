# Author: Ankit Kariryaa, University of Bremen




import numpy as np
import os  
from PIL import Image
from imgaug import augmenters as iaa

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5  : iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images 左右翻转
        iaa.Flipud(0.5),  # vertically flip 50% of all images 上下翻转
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq
 
# Each area (NDWI, annotation) is represented as an Frame
class FrameInfo:
    """ Defines a frame, includes its constituent images, annotation.
    """
    def __init__(self, img, annotations,  dtype=np.float32):
        """FrameInfo constructor.
        Args:
            img: ndarray
                3D array containing various input channels.
            annotations: ndarray
                3D array containing human labels, height and width must be same as img.
            dtype: np.float32, optional
                datatype of the array.
        """
        self.img = img
        self.annotations = annotations
        self.dtype = dtype

    #
    def getPatch(self, i, j, patch_size, img_size):
        """Function to get patch from the given location of the given size.  
        Args:
            i: int
                Starting location on first dimension (x axis).
            j: int
                Starting location on second dimension (y axis).
            patch_size: tuple(int, int)  
                Size of the patch.
            img_size: tuple(int, int)
                Total size of the images from which the patch is generated.
        """
        patch = np.zeros(patch_size, dtype=self.dtype)
    
        im = self.img[i:i + img_size[0], j:j + img_size[1]]#im.shape: (512,512, 1)      
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        an = np.expand_dims(an, axis=-1)#  (256, 256, 1) 
        comb_img = np.concatenate((im, an), axis=-1)  #(512,512,2)
        patch[:img_size[0], :img_size[1], ] = comb_img  #(512,512,2)
        return (patch)
    
   
    
    def sequential_patches(self, patch_size, step_size):
        """All sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Total size of the images from which the patch is generated.
        """
        img_shape = self.img.shape
        
        if (img_shape[0] <= patch_size[0]):
            x = [0]
        else:
            x = range(0, img_shape[0] - patch_size[0], step_size[0])
            #hbh: 当最后一步长度小于步长的1/3，就不走最后一步
            if ((img_shape[0] - patch_size[0]-x[-1]) > patch_size[0]/3) :
                x=np.append(x,img_shape[0] - patch_size[0])
                
        if (img_shape[1] <= patch_size[1]):
            y = [0]
        else:
            y = range(0, img_shape[1] - patch_size[1], step_size[1])
            if ((img_shape[1] - patch_size[1]- y[-1]) > patch_size[0]/3) :
                y=np.append(y,img_shape[1] - patch_size[1])

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        for i, j in xy:
#             if not img_patch[...,1].all()>0:
#                  img_patches.append(img_patch)
            img_patch = self.getPatch(i, j, patch_size, ic)
            img_patches.append(img_patch)
        # print(len(img_patches))
        return (img_patches)
    
    # Returns a single patch, startring at a random image
    def random_patch(self, patch_size):
        """A random from this frame.
        Args:
            patch_size: tuple(int, int)
                Size of the patch.
        """
        img_shape = self.img.shape
        if (img_shape[0] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        if (img_shape[1] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch = self.getPatch(x, y, patch_size, ic)
        return (img_patch)
    
        # Returns all patches in a image, sequentially generated
