import tensorflow as tf
import numpy as np

from ops import *

he_init = tf.contrib.layers.variance_scaling_initializer

class Autoencoder():
  ''' Autoencoder including encode, decode networks. '''
  def __init__(self, img_shape=(64, 64, 3), zsize=128):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    # latent (z) vector length
    self.zsize = zsize
    self.g_bns = [
      batch_norm(name='g_bn{}'.format(i,)) for i in range(6)]

  def generator(self, inputs, training, scope='generator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
      
      # self.gf_dim = 64

      self.z_, self.h0_w, self.h0_b = linear(inputs, 8192, 'g_h0_lin', with_w=True)
      # self.z_ = tf.layers.dense(inputs, 8192, kernel_initializer=he_init())
      
      hs = [None]
      hs[0] = tf.reshape(self.z_, [-1, 4, 4, 512])
      hs[0] = tf.nn.relu(self.g_bns[0](hs[0], training))

      i = 1 # Iteration number.
      depth_mul = 8  # Depth decreases as spatial component increases.
      size = 8  # Size increases as depth decreases.

      while size < 64:
        hs.append(None)
        name = 'g_h{}'.format(i)
        hs[i], _, _ = conv2d_transpose(hs[i-1],
            [64, size, size, 64*depth_mul], name=name, with_w=True)
        hs[i] = tf.nn.relu(self.g_bns[i](hs[i], training))

        i += 1
        depth_mul //= 2
        size *= 2

      hs.append(None)
      name = 'g_h{}'.format(i)
      hs[i], _, _ = conv2d_transpose(hs[i - 1],
          [64, size, size, 3], name=name, with_w=True)
      
      return tf.nn.tanh(hs[i])



