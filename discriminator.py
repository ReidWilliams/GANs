import tensorflow as tf
import numpy as np

he_init = tf.contrib.layers.variance_scaling_initializer

class Discriminator():
  ''' Discriminator network. Takes in image and outputs predition of whether
  image is real or fake. Has two outputs. One is scalar that is probability
  that an image is fake. Other is a vector that can be used to determine
  similarity between two candidate images. '''
  def __init__(self, img_shape=(64, 64, 3), reuse=False):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    self.reuse = reuse

  def disc(self, inputs):
    with tf.variable_scope('discriminator', reuse=self.reuse):
    
      # architecture is similar to autoencoder's encoder. See that for 
      # detailed comments.
      t = tf.layers.conv2d(inputs, 32, 5, strides=2)
      # according to paper, no BN after first conv layer
      t = tf.nn.elu(t)

      t = tf.layers.conv2d(t, 128, 5, strides=2)
      t = tf.layers.batch_normalization(t, axis=-1)
      t = tf.nn.elu(t)

      t = tf.layers.conv2d(t, 256, 5, strides=2)
      t = tf.layers.batch_normalization(t, axis=-1)
      t = tf.nn.elu(t)

      t = tf.layers.conv2d(t, 256, 5, strides=2)
      t = tf.layers.batch_normalization(t, axis=-1)
      t = tf.nn.elu(t)

      # use this vector to compare similarity of two images
      self.similarity = tf.contrib.layers.flatten(t)

      t = tf.layers.dense(self.similarity, 512)
      t = tf.layers.batch_normalization(t)
      t = tf.nn.elu(t)

      # output classification: probability an image is fake
      t = tf.layers.dense(t, 1)
      classification = tf.sigmoid(t)
    return classification





