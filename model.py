"""Keras based implementation of Autoencoding beyond pixels using a learned similarity metric

References:

Autoencoding beyond pixels using a learned similarity metric
by: Anders Boesen Lindbo Larsen, Soren Kaae Sonderby, Hugo Larochelle, Ole Winther
https://arxiv.org/abs/1512.09300

Adapted from https://github.com/commaai/research/tree/master/models
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
# from layers import Deconv2D
from keras import backend as K
from keras.layers import Input, Conv2D, ELU, Flatten, Dense, Lambda, BatchNormalization as BN
# from keras.layers import Input, Dense, Reshape, Activation, Conv2D, LeakyReLU, Flatten, BatchNormalization as BN
from keras.models import Model

class Vaegan():
  def __init__(self, img_shape=(64, 64, 3), zsize=128, batch_size=32):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    # latent (z) vector length
    self.zsize = zsize
    self.batch_size = batch_size
    self._build_model()

  def _build_encoder(self):
    inputs = Input(shape=self.img_shape)
    
    # Base number of 2D convolution filters. 64 is from paper.
    filters = 64
    t = inputs
    # Using Keras functional api
    # 64 filters of 5x5 field with stride 2
    t = Conv2D(filters, 5, strides=2, data_format='channels_last')(t)
    # Batch normalize per channel (per the paper) and channels are last dim
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

    model = Model(inputs=inputs, outputs=[mean, logsigma])
    return model

  def _sample(self, x):
    """
    Given a mean and logsigma value from the encoder, draw a zsize
    vector from gassian and return. This is a defining feature of a variational
    autoencoder.
    """

    mean, logsigma = x
    sigma = K.exp(logsigma/2.0) # see Hands on machine learning, Geron, p. 435

    # need to explicitly account for fact that input tensors include a whole
    # batch. 
    sample = K.random_normal_variable((self.batch_size, self.zsize), 0., 1.)
    return sigma*sample + mean

  def _sampling_layer(self):
    """
    Custom Keras layer that samples gaussian according to mean and sigma
    from the encoder.
    """
    return Lambda(self._sample)

    
    # HERE: build the decoder network.
    # PLAN: build decoder, implement custom loss function using keras backend, use
    # MSE for loss of X vs X' (for now). Test with some images.

  # def decode():
  #   pass

  def _build_model(self):
    
    encoder = self._build_encoder()
    sampling_layer = self._sampling_layer()
    inputs = Input(shape=self.img_shape)

    tensor = inputs
    tensor = encoder(tensor)
    outputs = sampling_layer(tensor)
    
    self.model = Model(inputs=inputs, outputs=outputs)








