import tensorflow as tf
import numpy as np

he_init = tf.contrib.layers.variance_scaling_initializer

class Discriminator():
  ''' Discriminator network. Takes in image and outputs predition of whether
  image is real or fake. Has two outputs. One is scalar that is probability
  that an image is fake. Other is a vector that can be used to determine
  similarity between two candidate images. '''
  def __init__(self, img_shape=(64, 64, 3)):
    # Input image shape: x, y, channels
    self.img_shape = img_shape

  def disc(self, inputs, training, scope='discriminator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
    
      # architecture is similar to autoencoder's encoder. See that for 
      # detailed comments.
      t = tf.layers.conv2d(inputs, 32, 5, strides=2, name='conv2d1')
      # according to paper, no BN after first conv layer
      t = tf.nn.elu(t, name='elu1')

      t = tf.layers.conv2d(t, 128, 5, strides=2, name='conv2d2')
      t = tf.layers.batch_normalization(t, axis=-1, training=training, name='bn1')
      t = tf.nn.elu(t, name='elu2')

      t = tf.layers.conv2d(t, 256, 5, strides=2, name='conv2d3')
      t = tf.layers.batch_normalization(t, axis=-1, training=training, name='bn2')
      t = tf.nn.elu(t, name='elu3')

      t = tf.layers.conv2d(t, 256, 5, strides=2, name='conv2d4')
      t = tf.layers.batch_normalization(t, axis=-1, training=training, name='bn3')
      t = tf.nn.elu(t, name='elu4')

      # use this vector to compare similarity of two images
      self.similarity = tf.contrib.layers.flatten(t)

      t = tf.layers.dense(self.similarity, 512, name='dense1')
      t = tf.layers.batch_normalization(t, axis=-1, training=training, name='bn4')
      t = tf.nn.elu(t, name='elu5')

      # output classification: probability an image is fake
      self.logits = tf.layers.dense(t, 1, name='dense2')
      classification = tf.sigmoid(self.logits, name='sigmoid')
    return classification





