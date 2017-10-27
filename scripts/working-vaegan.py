
# coding: utf-8

# In[1]:


# Setup

# img_directory = '/Users/rwilliams/Desktop/celeba/training'
img_directory = '/home/ec2-user/training-data/img_align_celeba'
model_save_path = '/home/ec2-user/tf-checkpoints/vaegan-celeba/checkpoint.ckpt'
outputs_directory = '/home/ec2-user/outputs/vaegan-celeba'
log_directory = '/home/ec2-user/tf-logs/vaegan-celeba'
batch_size = 64
training_set_size = 5000
img_size = 64

# for adam optimizer
learning_rate = 2e-4
# learning_beta1 = 0.5
learning_beta1 = 0.9

zsize = 128

# weights similarity loss term for decoder loss
# loss_gamma = 1e-2
# trying higher gamma
loss_gamma = 100.


# In[2]:


# Jupyter imports

# import matplotlib.pyplot as plt
# %matplotlib inline


# In[3]:


import numpy as np
import scipy as sp
import os
from utils import imshow, resize_crop, load_img


# In[4]:


# load training data
training = np.array([resize_crop(load_img(i+1, img_directory), (img_size, img_size)) for i in range(training_set_size)])


# # Build graph

# In[5]:


# create models

import tensorflow as tf
from autoencoder import Autoencoder
from discriminator import Discriminator
tf.reset_default_graph()
tf.set_random_seed(42.0)

# input images feed
X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])

# for feeding random draws of z (latent variable)
Z = tf.placeholder(tf.float32, [None, zsize])

# encoder, decoder that will be connected to a discriminator
vae = Autoencoder(img_shape=(img_size, img_size, 3), zsize=zsize)
encoder = vae.encoder(X)
decoder = vae.decoder(encoder)

# a second decoder for decoding samplings of z
decoder_z_obj = Autoencoder(img_shape=(img_size, img_size, 3), zsize=zsize)
decoder_z = decoder_z_obj.decoder(Z, reuse=True)

# discriminator attached to vae output
disc_vae_obj = Discriminator(img_shape=(img_size, img_size, 3))
disc_vae_obj.disc(decoder)
disc_vae_logits = disc_vae_obj.logits

# discriminator attached to X input
# shares weights with other discriminator
disc_x_obj = Discriminator(img_shape=(img_size, img_size, 3))
disc_x_obj.disc(X, reuse=True)
disc_x_logits = disc_x_obj.logits

# discriminator attached to random Zs passed through decoder
# shares weights with other discriminator
disc_z_obj = Discriminator(img_shape=(img_size, img_size, 3))
disc_z_obj.disc(decoder_z, reuse=True)
disc_z_logits = disc_z_obj.logits


# # Loss functions and optimizers

# In[6]:


# set up loss functions and training_ops

# latent loss used for training encoder
latent_loss = vae.latent_loss()

# loss that uses decoder to determine similarity between
# actual input images and output images from the vae
similarity_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=disc_x_obj.similarity, 
    logits=disc_vae_obj.similarity)
similarity_loss = tf.reduce_mean(similarity_xentropy)

# losses for the discriminator's output. Labels are real: 0, fake: 1.
# cross entropy with 1 labels, since training prob that image is fake
disc_vae_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(disc_vae_logits),
    logits=disc_vae_logits))

# cross entropy with 0 labels, since training prob that image is fake
disc_x_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(disc_x_logits),
    logits=disc_x_logits))

# cross entropy with 1 labels, since training prob that image is fake
disc_z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(disc_z_logits),
    logits=disc_z_logits))

# minimize these with optimizer
disc_loss = disc_vae_loss + disc_x_loss + disc_z_loss
encoder_loss = latent_loss + similarity_loss
decoder_loss = loss_gamma * similarity_loss - disc_loss

# get weights to train for each of encoder, decoder, etc.
# pass this to optimizer so it only trains w.r.t the network
# we want to train and just uses other parts of the network as is
# (for example use the discriminator to compute a loss during training
# of the encoder, but don't adjust weights of the discriminator)

encoder_vars = [i for i in tf.trainable_variables() if 'encoder' in i.name]
decoder_vars = [i for i in tf.trainable_variables() if 'decoder' in i.name]
disc_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]

encoder_update_ops = [i for i in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'encoder' in i.name]
decoder_update_ops = [i for i in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'decoder' in i.name]
disc_update_ops = [i for i in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminator' in i.name]

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=learning_beta1)
    
with tf.control_dependencies(encoder_update_ops):
    train_encoder = optimizer.minimize(encoder_loss, var_list=encoder_vars)
    
with tf.control_dependencies(decoder_update_ops):
    train_decoder = optimizer.minimize(decoder_loss, var_list=decoder_vars)

with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer.minimize(disc_loss, var_list=disc_vars)

saver = tf.train.Saver()


# # Init session

# In[7]:


# create or restore session

sess = tf.InteractiveSession()
try:
    print('trying to restore session')
    saver.restore(sess, model_save_path)
    print('restored session')
except:
    print('failed to restore session, creating a new one')
    tf.global_variables_initializer().run()

# write logs for tensorboard
writer = tf.summary.FileWriter(log_directory, sess.graph)


# In[8]:


# collect data for tensorboard

disc_vae_out = tf.reduce_mean(tf.sigmoid(disc_vae_logits))
disc_x_out = tf.reduce_mean(tf.sigmoid(disc_x_logits))
disc_z_out = tf.reduce_mean(tf.sigmoid(disc_z_logits))

tf.summary.scalar('encoder_loss', encoder_loss)
tf.summary.scalar('decoder_loss', decoder_loss)
tf.summary.scalar('discriminator_loss', disc_loss)
tf.summary.scalar('similarity_loss', similarity_loss)
tf.summary.scalar('disc_vae_loss', disc_vae_loss)
tf.summary.scalar('disc_x_loss', disc_x_loss)
tf.summary.scalar('disc_z_loss', disc_z_loss)
tf.summary.scalar('latent_loss', latent_loss)

tf.summary.scalar('disc_vae_out', disc_vae_out)
tf.summary.scalar('disc_x_out', disc_x_out)
tf.summary.scalar('disc_z_out', disc_z_out)

merged_summary = tf.summary.merge_all()


# In[9]:


img_idx = 823


# # Train

# In[11]:


import math
epochs = 10000
batches = int(float(training_set_size) / batch_size)
train_discriminator = True

for epoch in range(epochs):
    print ('epoch %s ' % epoch, end='')
    zdraws = np.random.normal(size=(training_set_size, zsize))
    
    # train discriminator
    if (train_discriminator):
        for batch in range(batches):
            xfeed = training[batch*batch_size:(batch+1)*batch_size]
            zfeed = zdraws[batch*batch_size:(batch+1)*batch_size]
            sess.run(train_disc, feed_dict={X: xfeed, Z: zfeed})
            print('.', end='')
         
    # train encoder
    for batch in range(batches):
        xfeed = training[batch*batch_size:(batch+1)*batch_size]
        sess.run(train_encoder, feed_dict={X: xfeed})
        print('.', end='')
        
    # train decoder
    for batch in range(batches):
        xfeed = training[batch*batch_size:(batch+1)*batch_size]
        zfeed = zdraws[batch*batch_size:(batch+1)*batch_size]
        sess.run(train_decoder, feed_dict={X: xfeed, Z: zfeed})
        print('.', end='')
        
    print('')
    
    if (epoch % 1 == 0):
        print('saving session', flush=True)
        saver.save(sess, model_save_path)
        
        xfeed = training[:batch_size]
        zfeed = zdraws[:batch_size]
        summary = merged_summary.eval(feed_dict={X: xfeed, Z: zfeed})
        writer.add_summary(summary, epoch)
        
#         disc_vae_out_val = disc_vae_out.eval(feed_dict={X: xfeed})
#         if (disc_vae_out_val >= 0.5):
#             train_discriminator = False
#             print('stopping training of discriminator')
#         else:
#             train_discriminator = True
#             print('starting training of discriminator')
            
        example = decoder.eval(feed_dict={X: training[:1]})
        img_save_path = os.path.join(outputs_directory, '%06d.jpg' % img_idx)
        img_idx += 1
        sp.misc.imsave(img_save_path, example[0])


# In[ ]:


# vae_out = decoder.eval(feed_dict={X: training[:4]})
# imshow(training[:4])
# imshow(vae_out[:4])


# In[ ]:


# r = np.random.normal(size=(8,128), scale=1.0)
# y = sess.run(decoder, feed_dict={encoder: r})
# imshow(y[0:8])

