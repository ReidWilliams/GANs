from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.util import crop
import numpy as np
import os
import math

def imshow(imgs, cols=4):
    fig = plt.figure(figsize=(15,8))
    nimgs = len(imgs)
    rows = math.ceil(float(nimgs)/cols) 
    row = 0
    for i in range(nimgs):
        a=fig.add_subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

# resize smaller, cropping if necessary
def resize_crop(img, desired_dims):
    img_dims = np.array(img[:,:,0].shape)
    desired_dims = np.array(desired_dims)
    
    scale = desired_dims / img_dims
    resize_dim = np.argmax(scale)
    # this is the dim we'll resize to, the other
    # dimension will need to be cropped
       
    new_dims = (scale[resize_dim] * img_dims).astype('int_')
    img_resized = resize(img, new_dims)
    # resize the image

    # now crop along the longer dimension
    crop_dim = np.argmin(scale)
    crop_from_edge = int((new_dims[crop_dim] - desired_dims[crop_dim])/2)
    
    # assume the crop is the first dim
    crops = [[crop_from_edge, crop_from_edge], [0, 0]]
    # swap if it's not
    if (crop_dim == 1): crops = list(reversed(crops))
    # add a crop of 0 for third dim (which is really the channels dim)
    # crop fn wants list of lists, with inner list amount to cut at 
    # each edge of the dim
    crops = crops + [[0, 0]]
    img_cropped = crop(img_resized, crops)
    
    # finally, resize again in case crop_from_edge was an odd number
    img_final = resize(img_cropped, desired_dims)
    
    return img_final

# rescale pixel values linearly from [0,1] to [-1,1]
# This is supposedto help converging on GANs
def pixels11(images):
    return 2*images - 1

def pixels01(images):
    return 0.5*(images + 1)


