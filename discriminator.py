import tensorflow as tf
import numpy as np

from ops import *

he_init = tf.contrib.layers.variance_scaling_initializer

class Discriminator():
  ''' Discriminator network. Takes in image and outputs predition of whether
  image is real or fake. Has two outputs. One is scalar that is probability
  that an image is fake. Other is a vector that can be used to determine
  similarity between two candidate images. '''
  def __init__(self, img_shape=(64, 64, 3)):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    self.d_bns = [
      batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]

  def discriminator(self, inputs, training, scope='discriminator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
    
      h0 = lrelu(conv2d(inputs, 64, name='d_h0_conv'))
      h1 = lrelu(self.d_bns[0](conv2d(h0, 128, name='d_h1_conv'), training))
      h2 = lrelu(self.d_bns[1](conv2d(h1, 256, name='d_h2_conv'), training))
      h3 = lrelu(self.d_bns[2](conv2d(h2, 512, name='d_h3_conv'), training))
      h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
      
      return tf.nn.sigmoid(h4), h4





