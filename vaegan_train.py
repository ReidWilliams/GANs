import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers import Input
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import Model

from autoencoder import Autoencoder
from discriminator import Discriminator

class VaeganTrain():
  ''' Full autoencoder plus discriminator with methods to get different
  models for training. Implemented this way because I was fighting with
  Keras' custom loss functions and decided to put all losses into custom layers.'''
  def __init__(self, img_shape=(64, 64, 3), zsize=128, batch_size=32):
    # Input image shape: x, y, channels
    self.img_shape = img_shape
    # latent (z) vector length
    self.zsize = zsize
    self.batch_size = batch_size

    self.vae = Autoencoder(self.img_shape, self.zsize, self.batch_size)
    self.disc = Discriminator(self.img_shape, self.batch_size)

  # These methods return a model ready to be trained with
  # appropriate layers frozen. Losses are built in to model
  # so compile with loss=None
  
  # Returns model with encoder ready for training.
  def trainable_encoder(self):
    self.vae.mode('train_encoder')

    inputs = Input(shape=self.img_shape)
    t = inputs
    # encode, decode the inputs
    img_out = self.vae.model(t)

    # use discriminator to get similarity vectors for 
    # vae output and real image
    similarity_vae = self.disc.similarity_model(img_out)
    similarity_real = self.disc.similarity_model(inputs)

    # use the loss layer to record the loss from difference
    # in similarity vectors
    similarity_layer = SimilarityLossLayer()
    outputs = similarity_layer([similarity_vae, similarity_real])

    # take index 0 to give some output, otherwise keras will complain
    return Model(inputs=inputs, outputs=outputs[0])
  
  # def train_decoder():
  #   pass

  # def train_discriminator():
  #   pass

class SimilarityLossLayer(Layer):
  ''' Custom layer that computes a loss based on two instances of
  the similarity vector from the discriminator.

  The layer doesn't modify it's inputs but just computes a loss.
  '''
  def __init__(self, **kwargs):
    super(SimilarityLossLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(SimilarityLossLayer, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape

  def _loss(self, similarity1, similarity2):
    # root mean square diff
    return K.sqrt(K.mean(K.square(similarity2 - similarity1), axis=1))

  def call(self, inputs):
    ''' Inputs for this layer is list of two similarity vectors'''
    similarity1, similarity2 = inputs
    loss = self._loss(similarity1, similarity2)
    self.add_loss(loss, inputs=inputs)
    # add the loss and just pass inputs on as outputs
    return inputs







