"""Keras based implementation of Autoencoding beyond pixels using a learned similarity metric

References:

Autoencoding beyond pixels using a learned similarity metric
by: Anders Boesen Lindbo Larsen, Soren Kaae Sonderby, Hugo Larochelle, Ole Winther
https://arxiv.org/abs/1512.09300

Adapted from https://github.com/commaai/research/tree/master/models
"""

# Keras v.2.0.6
# https://faroit.github.io/keras-docs/2.0.6/

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras import backend as K
from keras.layers import (
  Activation,
  BatchNormalization as BN,
  Conv2D,
  Conv2DTranspose,
  Cropping2D,
  Dense, 
  ELU,
  Flatten,
  Input, 
  Lambda, 
  Reshape,
)
from keras.models import Model
from layers import LatentLossLayer, SamplingLayer

class ImageAutoencoder():
  def __init__(self, img_shape=(64, 64, 3), zsize=128, batch_size=32):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    # latent (z) vector length
    self.zsize = zsize
    self.batch_size = batch_size

  def _encoder(self):
    inputs = Input(shape=self.img_shape)
    
    # Base number of 2D convolution filters. 64 is from paper.
    filters = 64
    t = inputs
    # Using Keras functional api
    # 64 filters of 5x5 field with stride 2
    t = Conv2D(filters, 5, strides=2, data_format='channels_last')(t)
    # Batch normalize per channel (per the paper) and channels are last dim.
    # This means find average accross the batch and apply it to the inputs, 
    # but do it separately for each channel. Also note that in the input layer,
    # we call them channels (red, green, blue) but in deepe layers each channel
    # is the output of a convolution filter.

    t = BN(axis=-1)(t)
    # Exponential linear unit for activation
    t = ELU(alpha=1)(t)

    t = Conv2D(filters*2, 5, strides=2, data_format='channels_last')(t)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2D(filters*4, 5, strides=2, data_format='channels_last')(t)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Flatten()(t)
    
    # In a variational autoencoder, the encoder outputs a mean and sigma vector
    # from which samples are drawn. In practice, treat the second output as
    # log(sigma**2), but we'll call it logsigma
    mean = Dense(self.zsize, activation="elu")(t)
    logsigma = Dense(self.zsize, activation="elu")(t)

    model = Model(inputs=inputs, outputs=(mean, logsigma))
    return model

  def _decoder(self):
    inputs = Input(shape=(self.zsize,))
    t = inputs

    filters = 64
    # deconvolution mirrors convolution, start with many filters, then
    # shrink down to a base level of filters. This is lowest number of filters
    # before wiring to 3 channel image (rgb).

    rows = [int(np.ceil(self.img_shape[0] / i)) for i in [16., 8., 4., 2.]]
    cols = [int(np.ceil(self.img_shape[1] / i)) for i in [16., 8., 4., 2.]]
    # What size should image be as we create larger and larger images with
    # each conv transpose layer.

    t = Dense(rows[0]*cols[0]*filters*8)(t)
    # densely connect z vector to enough units to supply first deconvolution layer.
    # That's rows*cols and at this layer use 8 times the base number of filters.

    t = Reshape((rows[0], cols[0], filters*8))(t)
    # for 64x64 images, this is 4x4 by 512 filters
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2DTranspose(filters*4, 5, strides=(2))(t)
    # Keras doesn't seem to let you specify the output rows/cols for
    # a transpose convolution. Because of the way the kernel slides accross
    # the input, and b/c we're using stride 2, output is double the input
    # rows/cols plus a few more due to width of kernel. Just crop out extras
    # before moving on.
    # e.g. for input image of size 4x4, with 5x5 kernel and stride of 2, 
    # we get output of 11x11, but want 8x8.
    t = Cropping2D(cropping=self._crops(t, rows[1], cols[1]))(t)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2DTranspose(filters*2, 5, strides=(2))(t)
    t = Cropping2D(cropping=self._crops(t, rows[2], cols[2]))(t)
    # for 64x64 images, this is 16x16 by 128 filters
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2DTranspose(filters, 5, strides=(2))(t)
    t = Cropping2D(cropping=self._crops(t, rows[3], cols[3]))(t)
    # for 64x64 images, this is 32x32 by 64 filters
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2DTranspose(self.img_shape[2], 5, strides=(2))(t)
    t = Cropping2D(cropping=self._crops(t, self.img_shape[0], self.img_shape[1]))(t)
    # for 64x64 rgb images, this is 64x64 by 3 channels

    outputs = Activation('sigmoid')(t)
    model = Model(inputs=inputs, outputs=outputs)
    return model

  def _crops(self, tensor, want_rows, want_cols):
    rows = K.int_shape(tensor)[1]
    cols = K.int_shape(tensor)[2]
    # crop at bottom and right of image
    return ((0, rows - want_rows), (0, cols - want_cols))

  def build_model(self):
    inputs = Input(shape=self.img_shape)

    # combine the pieces
    t = inputs
    t = self._encoder()(t)
    t = LatentLossLayer()(t)
    t = SamplingLayer(self.zsize, batch_size=self.batch_size)(t)
    outputs = self._decoder()(t)
    
    self.model = Model(inputs, outputs)
    return self.model


