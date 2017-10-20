import scipy as sp
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.util import crop
import numpy as np
import os
import math

def interleave(a, b, c):
    ''' Returns data set with a[0], b[0], c[0],
    a[1], b[1], c[1], etc. Used for interleaving real, 
    encoded/decoded, generated images to feed to 
    discriminator for training. Intuition is that batches
    should be composed of a mix of all three types.'''
    data_shape = a.shape
    
    collected = [np.expand_dims(x, 1) for x in (a, b, c)]
    catted = np.concatenate(collected, axis=1)

    # new shape is 1 by number of data points by dims of data itself
    new_shape = (1, 3*data_shape[0]) + data_shape[1:]
    reshaped = np.reshape(catted, new_shape)

    # remove the 0th axis because it only has one item
    return np.squeeze(reshaped, axis=0)

def imshow(imgs, cols=4):
    fig = plt.figure(figsize=(15,8))
    nimgs = len(imgs)
    rows = math.ceil(float(nimgs)/cols) 
    row = 0
    for i in range(nimgs):
        a=fig.add_subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def load_img(id, img_directory):
    filename = '%06d.jpg' % id
    path = os.path.join(img_directory, filename)
    return sp.ndimage.imread(path)

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