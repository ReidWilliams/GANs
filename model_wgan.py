# Subclass model that uses the Wasserstein distance loss function (WGAN)
# Wasserstein GAN with gradient penalty
# from github.com/lilianweng/unified-gan-tensorflow

import tensorflow as tf

from model import Model

class ModelWGAN(Model):
    def build_losses(self):
        epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = epsilon * self.X + (1 - epsilon) * self.Gz

        _, Dinterpolated_logits, _ = \
            self.arch.discriminator(interpolated, reuse=True)

        # tf.gradients returns a list of sum(dy/dx) for each x in xs.
        gradients = tf.gradients(Dinterpolated_logits, [interpolated])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))

        self.D_loss = tf.reduce_mean(self.Dreal_logits - self.Dz_logits) + grad_penalty
        self.G_loss = tf.reduce_mean(self.Dz_logits)

        # This is what I've seen WGAN implemtnations and paper do
        self.D_train_iters = 5



        
        





