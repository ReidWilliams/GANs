import numpy as np
import os

# Just some functions that help out

# rescale pixel values linearly from [0,1] to [-1,1]
# This is supposedto help converging on GANs
def pixels11(images):
    return 2*images - 1

def pixels01(images):
    return 0.5*(images + 1)

# from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
def tile(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')