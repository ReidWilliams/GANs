import tensorflow as tf

# It's nice to abstract your own ops, because then you can set parameters
# that are the same for all units in an architecture and the architecture code
# stays really clean

def conv2d(inputs, filters, strides=2):
    return tf.layers.conv2d(inputs, filters, 5, strides=strides, padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.constant_initializer(0.0))

# conv2d transpose
def conv2dtr(inputs, filters, strides=2):
    return tf.layers.conv2d_transpose(inputs, filters, 5, strides=strides, padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        bias_initializer=tf.constant_initializer(0.0))

def dense(inputs, units):
    return tf.layers.dense(inputs, units, 
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.constant_initializer(0.0))

# batch normalization
# unclear how important, but use scale=True and epsilon 1e-5
# from https://github.com/carpedm20/DCGAN-tensorflow/blob/b138300623b933e2076872e7f812ba553e862355/ops.py
class BN:
    def __init__(self, is_training=True):
        self.is_training = is_training

    def __call__(self, inputs):
        return tf.contrib.layers.batch_norm(
            inputs, updates_collections=None, is_training=self.is_training,
            scale=True, epsilon=1e-5, decay=0.9)

def flatten(inputs):
    return tf.contrib.layers.flatten(inputs)

def reshape(inputs, shape):
    return tf.reshape(inputs, shape)

def lrelu(inputs, leak=0.2):
    return tf.maximum(inputs, leak*inputs)

def sigmoid(inputs):
    return tf.sigmoid(inputs)

def tanh(inputs):
    return tf.tanh(inputs)
