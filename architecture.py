import tensorflow as tf
import numpy as np

from ops import BN, conv2d, conv2dtr, dense, lrelu, \
                flatten, reshape, sigmoid, tanh

class GAN:
    ''' Generative adversarial network with encoder. '''
    def __init__(self, is_training, img_shape=(64, 64, 3), zsize=128):
        # Input image shape: x, y, channels
        self.img_shape = img_shape
        # latent (z) vector length
        self.zsize = zsize
        # is the model being trained
        self.is_training = is_training

    def generator(self, inputs, scope='generator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # deconvolution mirrors convolution, start with many filters, then
            # shrink down to a base level of filters. This is lowest number of filters
            # before wiring to 3 channel image (rgb).

            minirows = self.img_shape[0] // 32
            minicols = self.img_shape[1] // 32 

            bn = BN(self.is_training)

            t = dense(inputs, minirows*minicols*512)
            t = lrelu(bn(reshape(t, (tf.shape(t)[0], minirows, minicols, 512))))

            t = lrelu(bn(conv2dtr(t, 512)))
            t = lrelu(bn(conv2dtr(t, 256)))
            t = lrelu(bn(conv2dtr(t, 128)))
            t = lrelu(bn(conv2dtr(t, 64)))

            # final conv2d  transpose to get to filter depth of 3, for rgb channels
            logits = conv2dtr(t, self.img_shape[2])
            return tanh(logits)


    def discriminator(self, inputs, scope='discriminator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):

            bn = BN(self.is_training)

            t = lrelu(conv2d(inputs, 64)) # no bn here
            t = lrelu(bn(conv2d(t, 128)))
            t = lrelu(bn(conv2d(t, 256)))
            t = lrelu(bn(conv2d(t, 512)))
            t = lrelu(bn(conv2d(t, 1024)))

            # use this vector to compare similarity of two images
            similarity = flatten(t)

            # output classification: probability an image is fake
            logits = dense(similarity, 1)
            classification = sigmoid(logits)
            return classification, logits, similarity



