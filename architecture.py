import tensorflow as tf
import numpy as np

# abstract neural network units used into a separate
# module
from ops import BN, conv2d, conv2dtr, dense, lrelu, \
                flatten, reshape, sigmoid, tanh

class GAN:
    def __init__(self, is_training, img_shape=(64, 64, 3), zsize=128):
        # Input image shape: x, y, channels
        self.img_shape = img_shape
        # latent (z) vector length
        self.zsize = zsize
        # is the generator being trained
        # Expect this to be a TF placeholder that is set true or false
        # depending on whether model is training or generating
        self.is_training = is_training

    # latent vector of size zsize goes in, an image of size img_shape comes out
    def generator(self, inputs, scope='generator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
           
            # generator will upscale a tiny image with layers of convolution
            # until it reaches the final output image dimensions. These vars
            # are the starting tiny image dimensions.
            # This is why this arch needs images that are divisible by 32
            minirows = self.img_shape[0] // 32
            minicols = self.img_shape[1] // 32 

            # batch normalization, which needs to know whether this is training or
            # application
            bn = BN(self.is_training)

            # dense (i.e. fully connected) layer followed by reshaping into the tiny
            # image. The tiny image has a Z dim of 512 that gradually gets reduced
            # to 3 channels (r, g, b)
            t = dense(inputs, minirows*minicols*512)
            t = lrelu(bn(reshape(t, (tf.shape(t)[0], minirows, minicols, 512))))

            t = lrelu(bn(conv2dtr(t, 512)))
            t = lrelu(bn(conv2dtr(t, 256)))
            t = lrelu(bn(conv2dtr(t, 128)))
            t = lrelu(bn(conv2dtr(t, 64)))

            # final conv2d  transpose to get to filter depth of 3, for rgb channels
            logits = conv2dtr(t, self.img_shape[2])
            return tanh(logits) # common final activation in GANs

    # image goes in, and score of 0 (fake) or 1 comes out. Actually returns more
    # than just that
    def discriminator(self, inputs, scope='discriminator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):

            # Set discriminator to always be training. Reason for doing this is
            # For the WGAN gradient loss (which is not the default loss function for
            # this model, still uses this architecture), the loss function has an expression
            # which is the gradient of an instance of the discriminator. Putting that
            # into the optimizer creates a dependency on the second order gradient of the
            # disriminator. Batch normalization where the training vs running flag is itself
            # a TF variable (rather than normal python boolean) seems to break this. Easier to
            # just set to True because in this model we only ever use the discriminator for
            # training (to train the generator).
            bn = BN(True)

            t = lrelu(conv2d(inputs, 64)) # no bn here
            t = lrelu(bn(conv2d(t, 128)))
            t = lrelu(bn(conv2d(t, 256)))
            t = lrelu(bn(conv2d(t, 512)))
            t = lrelu(bn(conv2d(t, 1024)))

            # flatten 3D tensor into 1D to prepare for a dense (fully connected)
            # layer. Flattened tensor can also be treated as vector that can be
            # used for learned similarty measurements between images.
            similarity = flatten(t)

            # return logits (before sigmoid activation) because several TF
            # accumulator functions expect logits, and do the sigmoid for you
            logits = dense(similarity, 1)
            classification = sigmoid(logits)
            return classification, logits, similarity



