import tensorflow as tf
import numpy as np

from layers import BN, conv2d, conv2dtr, dense, elu, flatten, reshape, tanh

class VAEGAN():
    ''' Generative adversarial network with encoder. '''
    def __init__(self, is_training, img_shape=(64, 64, 3), zsize=128):
        # Input image shape: x, y, channels
        self.img_shape = img_shape
        # latent (z) vector length
        self.zsize = zsize
        # is the model being trained
        self.is_training = is_training

    def encoder(self, inputs, scope='encoder', reuse=None):
        ''' Returns encoder graph. Inputs is a placeholder of size
        (None, rows, cols, channels) '''
        with tf.variable_scope(scope, reuse=reuse):

            bn = BN(self.is_training)



            t = elu(bn(conv2d(inputs, 64)))
            t = elu(bn(conv2d(inputs, 128)))
            t = elu(bn(conv2d(inputs, 256)))
            
            t = flatten(t)


            t = elu(bn(dense(t, 512)))
            
            # keep means and logsigma for computing variational loss
            self.means = elu(dense(t, self.zsize))
            self.logsigmas = elu(dense(t, self.zsize))
            
            sigmas = tf.exp(0.5 * self.logsigmas) # see Hands on machine learning, Geron, p. 435
            sample = tf.random_normal(tf.shape(sigmas), dtype=tf.float32)
            
            return sample * sigmas + self.means

    def latent_loss(self):
        with tf.variable_scope('latent_loss'):
            loss = 0.5 * tf.reduce_mean(tf.exp(self.logsigmas) + tf.square(self.means) - 1 - self.logsigmas)
            return loss

    def generator(self, inputs, scope='generator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # deconvolution mirrors convolution, start with many filters, then
            # shrink down to a base level of filters. This is lowest number of filters
            # before wiring to 3 channel image (rgb).

            # number of pixels on each side for starting 2d dimensions
            _len = int(self.img_shape[0] / 16)

            t = dense(inputs, _len*_len*512)
            t = elu(bn(reshape(t, (tf.shape(t)[0], _len, _len, 512))))

            t = elu(bn(conv2dtr(t, 512)))
            t = elu(bn(conv2dtr(t, 256)))
            t = elu(bn(conv2dtr(t, 128)))

            # final conv2d  transpose to get to filter depth of 3, for rgb channels
            logits = conv2dtr(t, self.img_shape[2])
            return tanh(logits)


    def discriminator(self, inputs, scope='discriminator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):

            t = elu(conv2d(inputs, 64)) # no bn here
            t = elu(bn(conv2d(t, 128)))
            t = elu(bn(conv2d(t, 256)))
            t = elu(bn(conv2d(t, 512)))

            # use this vector to compare similarity of two images
            self.similarity = flatten(t)

            # output classification: probability an image is fake
            logits = dense(self.similarity, 1)
            classification = sigmoid(self.logits)
            return classification, logits



