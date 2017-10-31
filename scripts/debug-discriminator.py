import math
import numpy as np
import scipy as sp
import os
import tensorflow as tf

from discriminator import Discriminator
from utils import imshow, resize_crop, load_img

# Setup

# img_directory = '/Users/rwilliams/Desktop/celeba/training'
img_directory = '/home/ec2-user/training-data/img_align_celeba'
# model_save_path = '/home/ec2-user/tf-checkpoints/vae-celeba/checkpoint.ckpt'
# outputs_directory = '/home/ec2-user/outputs/vaegan-celeba'
log_directory = '/home/ec2-user/tf-logs/vaegan-celeba'
batch_size = 64
training_set_size = 512
img_size = 64

# For adam optimizer
learning_rate = 2e-4
learning_beta1 = 0.5

zsize = 128

# load training data
print('loading and resizing training data')
training = np.array([resize_crop(load_img(i+1, img_directory), (img_size, img_size)) for i in range(training_set_size)])

# create models

# input images feed
X1 = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
# for feeding random draws of z (latent variable)
X2 = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
disc_trainable = tf.placeholder(tf.bool)

# discriminator attached to X1 input
# shares weights with other discriminator
disc_x1 = Discriminator(img_shape=(img_size, img_size, 3))
disc_x1_out, disc_x1_logits = disc_x1.discriminator(X1, disc_trainable)

disc_x2 = Discriminator(img_shape=(img_size, img_size, 3))
disc_x2_out, disc_x2_logits = disc_x2.discriminator(X2, disc_trainable, reuse=True)


# set up loss functions and training_ops

disc_x1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(disc_x1_logits),
    logits=disc_x1_logits))

disc_x2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(disc_x2_logits),
    logits=disc_x2_logits))

# minimize these with optimizer
disc_loss = disc_x1_loss + disc_x2_loss

# get weights to train for each of encoder, decoder, etc.
# pass this to optimizer so it only trains w.r.t the network
# we want to train and just uses other parts of the network as is
# (for example use the discriminator to compute a loss during training
# of the encoder, but don't adjust weights of the discriminator)

# need to explicitly add some of the BN variables so they get trained.
disc_update_ops = [i for i in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminator' in i.name]
disc_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=learning_beta1)

with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer.minimize(disc_loss)



# create or restore session

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# write logs for tensorboard
writer = tf.summary.FileWriter(log_directory, sess.graph)

# collect data for tensorboard

disc_x1_out_mean = tf.reduce_mean(tf.sigmoid(disc_x1_logits))
disc_x2_out_mean = tf.reduce_mean(tf.sigmoid(disc_x2_logits))

d_sum = tf.summary.merge(
    [
        tf.summary.scalar('disc_x1_out', disc_x1_out_mean),
        tf.summary.scalar('disc_x2_out', disc_x2_out_mean),
        tf.summary.scalar('disc_loss', disc_loss)
    ])


epochs = 10000
batches = int(float(training_set_size) / batch_size)
logcounter = 0

for epoch in range(epochs):
    print ('epoch %s ' % epoch, end='')
    randos = np.random.normal(size=(training_set_size, img_size, img_size, 3))
    
    for batch in range(batches):

        x1feed = training[batch*batch_size:(batch+1)*batch_size]
        x2feed = randos[batch*batch_size:(batch+1)*batch_size]

        _, summary_str = sess.run(
            [train_disc, d_sum],
            feed_dict={ X1: x1feed, X2: x2feed, disc_trainable: True })
        writer.add_summary(summary_str, logcounter)
        print('.', end='')

        logcounter += 1
       
    print('')

