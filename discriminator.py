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

class Discriminator():
  ''' Discriminator network. Takes in image and outputs predition of whether
  image is real or fake. Has two outputs. One is scalar that is probability
  that an image is fake. Other is a vector that can be used to determine
  similarity between two candidate images. '''
  def __init__(self, img_shape=(64, 64, 3), batch_size=32):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    self.batch_size = batch_size

  def _discriminator(self):
    inputs = Input(shape=self.img_shape)
    
    # architecture is similar to autoencoder's encoder. See that for 
    # detailed comments.
    t = inputs
    t = Conv2D(32, 5, strides=2, data_format='channels_last')(t)
    # according to paper, no BN after first conv layer
    t = ELU(alpha=1)(t)

    t = Conv2D(128, 5, strides=2, data_format='channels_last')(t)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2D(256, 5, strides=2, data_format='channels_last')(t)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    t = Conv2D(256, 5, strides=2, data_format='channels_last')(t)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    similarity = Flatten()(t)
     
    t = Dense(512)(similarity)
    t = BN(axis=-1)(t)
    t = ELU(alpha=1)(t)

    # output prediction
    prediction = Dense(1, activation='sigmoid')(t)
    # for some reason this is outputting (8, 5, 5, 1)
  
    model = Model(inputs=inputs, outputs=(prediction, similarity))
    return model

  def build_model(self):
    self.model = self._discriminator()
    return self.model


