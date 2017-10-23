from keras import backend as K
from keras.engine.topology import Layer

class SamplingLayer(Layer):
  ''' Used by autoencoder to output gaussian sample based on
  mean and sigma values from encoder. '''
  def __init__(self, zsize, batch_size=32, **kwargs):
    super(SamplingLayer, self).__init__(**kwargs)
    self.zsize = zsize
    self.batch_size = batch_size

  def build(self, input_shape):
    super(SamplingLayer, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return (self.batch_size, self.zsize)

  def call(self, inputs):
    ''' Inputs for this layer are (mean, logsigma) (
    tuple of tensors). Given a mean and logsigma value from 
    the encoder, draw a zsize vector from gassian and return. 
    This is a defining feature of a variational autoencoder.'''
    
    mean, logsigma = inputs
    sigma = K.exp(logsigma/2.0) # see Hands on machine learning, Geron, p. 435

    # need to explicitly account for fact that input tensors include a whole
    # batch.
    sample = K.random_normal_variable((self.batch_size, self.zsize), 0., 1.)
    return sigma*sample + mean

class LatentLossLayer(Layer):
  ''' Custom layer that computes latent loss, ie loss due to
  difference between zspace distribution and gaussian. See
  Hands on machine learning, p.435 for details.

  The layer doesn't modify it's inputs but just computes a loss. Do this
  because Keras is restrictive in that custom loss functions must be a function
  of y_predicted, y_actual, but variational autoencoders have a loss term that
  comes from the distribution of the latent z vectors.
  '''
  def __init__(self, **kwargs):
    super(LatentLossLayer, self).__init__(**kwargs)
    # set to 1 or 0 to include this layer's loss
    self._use_loss = K.variable(1.0)

  def build(self, input_shape):
    super(LatentLossLayer, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape

  def _loss(self, mean, logsigma):
    return self._use_loss * 0.5 * K.sum(K.exp(logsigma) + K.square(mean) - 1 - logsigma)

  def use_loss(self, b):
    if b == True:
      self._use_loss = K.variable(1.0)
    else:
      self._use_loss = K.variable(0.0)

  def call(self, inputs):
    ''' Inputs for this layer are [mean, logsigma]'''
    mean, logsigma = inputs
    loss = self._loss(mean, logsigma)
    self.add_loss(loss, inputs=inputs)
    # add the loss and just pass inputs on as outputs
    return inputs
