import math
import numpy as np
import scipy as sp
import os
from joblib import Memory
import tensorflow as tf

from discriminator import Discriminator
from autoencoder import Autoencoder
from utils import imshow, resize_crop, load_img, pixels11, pixels01

# Setup

# img_directory = '/Users/rwilliams/Desktop/celeba/training'
img_directory = '/home/ec2-user/training-data/img_align_celeba'
# model_save_path = '/home/ec2-user/tf-checkpoints/vae-celeba/checkpoint.ckpt'
outputs_directory = '/home/ec2-user/outputs/vaegan-celeba'
log_directory = '/home/ec2-user/tf-logs/vaegan-celeba'
cache_directory = '/home/ec2-user/joblib-cache'
batch_size = 64
training_set_size = 10000
img_size = 64

# For adam optimizer
learning_rate = 0.0002
learning_beta1 = 0.5

zsize = 128

# load training data
print('loading and resizing training data')


# cache results of resizing and cropping on disk
memory = Memory(cachedir=cache_directory, verbose=0)
@memory.cache
def load_all_imgs(howmany, img_directory):
    training = np.array([resize_crop(load_img(i+1, img_directory), (img_size, img_size)) for i in range(howmany)])
    training = pixels11(training)
    return training

training = load_all_imgs(training_set_size, img_directory).astype('float32')

# create models

# input images feed
X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
# for feeding random draws of z (latent variable)
Z = tf.placeholder(tf.float32, [None, zsize])

is_training = tf.placeholder(tf.bool)

# generator
gen = Autoencoder().generator(Z, is_training)

# discriminator attached to X1 input
# shares weights with other discriminator
disc_x = Discriminator(img_shape=(img_size, img_size, 3))
disc_x_out, disc_x_logits = disc_x.discriminator(X, is_training)

disc_g = Discriminator(img_shape=(img_size, img_size, 3))
disc_g_out, disc_g_logits = disc_g.discriminator(gen, is_training, reuse=True)

# set up loss functions and training_ops

disc_x_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=(tf.ones_like(disc_x_logits)),
    logits=disc_x_logits))

disc_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(disc_g_logits),
    logits=disc_g_logits))

# minimize these with optimizer
disc_loss = disc_x_loss + disc_g_loss

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(disc_g_logits),
    logits=disc_g_logits))

# get weights to train for each of encoder, decoder, etc.
# pass this to optimizer so it only trains w.r.t the network
# we want to train and just uses other parts of the network as is
# (for example use the discriminator to compute a loss during training
# of the encoder, but don't adjust weights of the discriminator)

# need to explicitly add some of the BN variables so they get trained.
disc_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]
gen_vars = [i for i in tf.trainable_variables() if 'generator' in i.name]

disc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=learning_beta1)
gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=learning_beta1)

disc_train = disc_optimizer.minimize(disc_loss, var_list=disc_vars)
gen_train = gen_optimizer.minimize(gen_loss, var_list=gen_vars)

# create session

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# write logs for tensorboard
writer = tf.summary.FileWriter(log_directory, sess.graph)

# collect data for tensorboard

disc_x_out_mean = tf.reduce_mean(tf.sigmoid(disc_x_logits))
disc_g_out_mean = tf.reduce_mean(tf.sigmoid(disc_g_logits))

disc_sum = tf.summary.merge([
    tf.summary.scalar('disc_x_out', disc_x_out_mean),
    tf.summary.scalar('disc_g_out', disc_g_out_mean),
    tf.summary.scalar('disc_loss', disc_loss),
    tf.summary.histogram('x_img', tf.reshape(X, (1, batch_size*64*64*3))),
    tf.summary.histogram('gen_img', tf.reshape(gen, (1, batch_size*64*64*3)))
])

gen_sum = tf.summary.merge([
    tf.summary.scalar('gen_loss', gen_loss)
])

epochs = 10000
batches = int(float(training_set_size) / batch_size)
logcounter = 0
imgcounter = 0

def printnow(x, end='\n'): print(x, flush=True, end=end)

print('training over %s batches per epoch' % batches)

for epoch in range(epochs):
    print ('epoch %s ' % epoch, end='')
    zdraws = np.random.uniform(-1, 1, size=(training_set_size, zsize)).astype('float32')
    
    for batch in range(batches):

        xfeed = training[batch*batch_size:(batch+1)*batch_size]
        zfeed = zdraws[batch*batch_size:(batch+1)*batch_size]

        # train discriminator
        _, summary_str = sess.run(
            [disc_train, disc_sum],
            feed_dict={ 
                X: xfeed, 
                Z: zfeed, 
                is_training: True
            })
        writer.add_summary(summary_str, logcounter)
        printnow('.', end='')

        # train generator
        for _ in range(2):
            _, summary_str = sess.run(
                [gen_train, gen_sum],
                feed_dict={ 
                    Z: zfeed, 
                    is_training: True 
                })
            writer.add_summary(summary_str, logcounter)
            printnow('.', end='')

        logcounter += 1

        if (batch % 20 == 0):
            printnow('saving images', end='')
            sample_imgs = sess.run(gen,
                feed_dict={ 
                    Z: zfeed, 
                    is_training: False,
                })

            img_save_path = os.path.join(outputs_directory, '%06d.jpg' % imgcounter)
            sp.misc.imsave(img_save_path, pixels01(sample_imgs[0]))
            imgcounter += 1

    printnow('')
       


