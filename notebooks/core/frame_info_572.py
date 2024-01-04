# Author: Ankit Kariryaa, University of Bremen




import numpy as np
import os  
from PIL import Image

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
    def getPatch(self, i, j, input_size, output_size, ann_size):
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
        patch_im = np.zeros(input_size, dtype=self.dtype)
        patch_an  =np.zeros(output_size, dtype=self.dtype)
    
        an = self.annotations[i:i + ann_size[0], j:j + ann_size[1]]
        an = np.expand_dims(an, axis=-1)#  (256, 256, 1) 
        orig_x=int((input_size[0]-output_size[0])/2)
        orig_y=int((input_size[1]-output_size[1])/2)
        img_size_x= min(2*orig_x + ann_size[0], input_size[0])
        img_size_y= min(2*orig_y + ann_size[1], input_size[1])
        im=self.img[i: i + img_size_x, j: j + img_size_y]
        patch_im[:img_size_x,:img_size_y]=im
        patch_an[:ann_size[0],:ann_size[1]]=an
        return patch_im,patch_an
    
   
    
    def sequential_patches(self,input_size,output_size, step_size):
        """All sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Total size of the images from which the patch is generated.
        """
        ann_shape = self.annotations.shape
        
        if (ann_shape[0] <= output_size[0]):
            x = [0]
        else:
            x = range(0, ann_shape[0] - output_size[0], step_size[0])
            #hbh: 当最后一步长度小于步长的1/3，就不走最后一步
            # if ((ann_shape[0] - output_size[0]-x[-1]) > output_size[0]/3) :
            #     x=np.append(x,ann_shape[0] - output_size[0])
                
        if (ann_shape[1] <= output_size[1]):
            y = [0]
        else:
            y = range(0, ann_shape[1] - output_size[1], step_size[1])
            # if ((ann_shape[1] - output_size[1]- y[-1]) > output_size[0]/3) :
            #     y=np.append(y,ann_shape[1] - output_size[1])

        ic = (min(ann_shape[0], output_size[0]), min(ann_shape[1], output_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        ann_patches = []
        for i, j in xy:
            img_patch,ann_patch = self.getPatch(i, j, input_size,output_size, ic)
#             if img_patch[...,-1].any()>0:
#                  img_patches.append(img_patch)
            img_patches.append(img_patch)
            ann_patches.append(ann_patch)
#         print(len(img_patches))
        return img_patches,ann_patches
    
    # Returns a single patch, startring at a random image
    def random_patch(self, input_size,output_size):
        """A random from this frame.
        Args:
            patch_size: tuple(int, int)
                Size of the patch.
        """
        ann_shape = self.annotations.shape
        
        if (ann_shape[0] <= output_size[0]):
            x = 0
        else:
             x = np.random.randint(0, ann_shape[0] - output_size[0])
        if (ann_shape[1] <= output_size[1]):
            y = 0
        else:
             y = np.random.randint(0, ann_shape[1] - output_size[1])
        ic = (min(ann_shape[0], output_size[0]), min(ann_shape[1], output_size[1]))
        img_patch,ann_patch = self.getPatch(x, y, input_size,output_size, ic)
        return img_patch,ann_patch
    
        # Returns all patches in a image, sequentially generated
