# Model brings together the network, the loss function, the feed of 
# training images, and a training loop

import tensorflow as tf
from PIL import Image
import numpy as np
import os

from feed import Feed
from architecture import GAN
from utils import pixels01, pixels11, tile

# print and flush 
def printnow(x, end='\n'): print(x, flush=True, end=end)
# safe create directories
def makedirs(d): 
    if not os.path.exists(d): os.makedirs(d) 
          
# This model uses the same loss function as DCGAN 
class Model:
    def __init__(self, feed, batch_size=64, img_shape=(64, 64),
        G_lr=0.0004, D_lr=0.0004, G_beta1=0.5, D_beta1=0.5, 
        zsize=128, save_freq=10, output_cols=4, output_rows=4,
        sess=None, checkpoints_path=None):

        self.batch_size = batch_size

        if ((img_shape[0] % 32 != 0) or (img_shape[1] % 32 != 0)):
            raise ValueException("Image dimensions need to be divisible by 32. \
                Dimensions received was %s." % img_shape)

        self.img_shape = img_shape + (3,) # add (r,g,b) channels dimension
        
        # learning rates for Adam optimizer
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.G_beta1 = G_beta1
        self.D_beta1 = D_beta1

        # size of latent vector
        self.zsize = zsize
        # save session and examples after this many batches
        self.save_freq = int(save_freq)
        # cols and rows of output image tile
        self.output_cols = output_cols
        self.output_rows = output_rows

        pwd = os.getcwd()
        self.dirs = {
            'output':      os.path.join(pwd, 'output'),
            'logs':        os.path.join(pwd, 'logs'),
            'checkpoints': os.path.join(pwd, 'checkpoints')
        }

        # set or create tensorflow session
        self.sess = sess
        if not self.sess:
            self.sess = tf.InteractiveSession()

        # create directories if they don't exist
        makedirs(self.dirs['logs'])
        makedirs(self.dirs['output'])
        makedirs(self.dirs['checkpoints'])
        self.checkpoints_path = checkpoints_path or os.path.join(self.dirs['checkpoints'], 'checkpoint.ckpt')

        # get number of files in output so we can continue where a previous process
        # left off without overwriting
        self.output_img_idx = len([f for f in os.listdir(self.dirs['output']) \
            if os.path.isfile(os.path.join(self.dirs['output'], f))])

        # data feed for training  
        self.feed = feed
        # bool used by batch normalization. BN behavior is different when training
        # vs predicting
        self.is_training = tf.placeholder(tf.bool)
        self.arch = GAN(self.is_training, img_shape=self.img_shape, zsize=128)

        # how many times to train discriminator per minibatch
        # This is a hyperparameter that can be tuned, it's >1 in wgans
        self.D_train_iters = 2

    # Build the network
    def build_model(self):
        # real image inputs from training data feed for training the
        # discriminator
        self.X = tf.placeholder(tf.float32, (None,) + self.img_shape)
        # for feeding random draws of z (latent variable) to the generator
        self.Z = tf.placeholder(tf.float32, (None, self.zsize))

        # Instantiate a generator network. It takes an input
        # of a latent vector
        self.Gz = self.arch.generator(self.Z)

        # discriminator connected to real image input (X)
        self.Dreal, self.Dreal_logits, self.Dreal_similarity = \
            self.arch.discriminator(self.X)

        # create a second instance of the discriminator and connect to the
        # output of the generator. reuse=True means this second instance will
        # share the same network weights and biases as the first instance. We want
        # this because training the discriminator weights happens using a loss
        # function that is a function of both the discriminator applied to a generated
        # image and the discriminator applied to a real image.
        self.Dz, self.Dz_logits, _ = \
            self.arch.discriminator(self.Gz, reuse=True)

    # Build the loss function.
    def build_losses(self):
        self.Dreal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=(tf.ones_like(self.Dreal_logits) - 0.25),
            logits=self.Dreal_logits))

        self.Dz_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.Dz_logits),
            logits=self.Dz_logits))

        # discriminator loss function from DCGAN
        # discriminator wants to label real images with 1, generated with 0
        self.D_loss = self.Dreal_loss + self.Dz_loss

        # generator loss function. Make the generator think generated images
        # are real
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.Dz_logits),
            logits=self.Dz_logits))

    def build_optimizers(self):
        # explicitly grab lists of variables for each type of network. This is used
        # below to set up TF operations that train only one part of the network at
        # a time (either generator or discriminator)
        G_vars = [i for i in tf.trainable_variables() if 'generator' in i.name]
        D_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]
  
        # Create optimizers.
        G_opt = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.G_beta1)
        D_opt = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.D_beta1)
        
        # In tensor flow, you set up training by handing an optimizer object a tensor
        # this is the output of a loss function, and (in this case) a set of variables
        # that can be changed. You get back a training operation that you then run
        # (see below) to take a step in training.
        # pass var_list explicitly so that during training of (e.g.) generator, discriminator
        # weights and biases aren't updated.
        self.G_train = G_opt.minimize(self.G_loss, var_list=G_vars)
        self.D_train = D_opt.minimize(self.D_loss, var_list=D_vars)

    def setup_session(self):
        # store epoch as tf variable so we can save it in the session
        # this is nice for logging so that restarting the process doesn't reset the
        # epoch count.
        self.epoch = tf.get_variable('epoch', dtype='int32', initializer=tf.constant(0))

        # random numbers to generate outputs. Store in tf variable so it gets
        # stored in session. This is useful so that generated images that are saved
        # during training come from the same latent variable inputs. This lets you 
        # see the gradual change / improvement of outputs even if the process dies
        # and gets restarted
        self.example_noise = tf.get_variable('noise', dtype='float32', 
            initializer=tf.constant(np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')))
        
        self.saver = tf.train.Saver()
        
        try:
            print('trying to restore session from %s' % self.checkpoints_path)
            self.saver.restore(self.sess, self.checkpoints_path)
            print('restored session')
        except:
            print('failed to restore session, creating a new one')
            tf.global_variables_initializer().run()

    # log some basic data for tensorboard
    def setup_logging(self):
        self.writer = tf.summary.FileWriter(self.dirs['logs'], self.sess.graph)

        self.G_stats = tf.summary.merge([
            tf.summary.scalar('G_loss', self.G_loss)
        ])

        Dreal_mean = tf.reduce_mean(tf.sigmoid(self.Dreal_logits))
        Dz_mean = tf.reduce_mean(tf.sigmoid(self.Dz_logits))

        self.D_stats = tf.summary.merge([
            tf.summary.scalar('Dreal_out', Dreal_mean),
            tf.summary.scalar('Dz_out', Dz_mean),
            tf.summary.scalar('D_loss', self.D_loss)
        ])

    def train(self):
        batches = self.feed.nbatches()
        printnow('training with %s batches per epoch' % batches)
        printnow('saving session and examples every %s batches' % self.save_freq)
        
        # order the logged data for tensorboard
        logcounter = 0

        epoch = self.epoch.eval() # have to do this b/c self.epoch is a tensorflow var

        while True:            
            for batch in range(batches):
                # training image pixel values are [0,1] but DCGAN and it seems most
                # GAN architectures benefit from / use [-1,1]
                xfeed = pixels11(self.feed.feed(batch)) # convert to [-1, 1]
                zfeed = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')

                # train discriminator (possibly more than once) by running
                # the training operation inside the session
                for i in range(self.D_train_iters):
                    _, summary = self.sess.run(
                        [ self.D_train, self.D_stats ],
                        feed_dict={ self.X: xfeed, self.Z: zfeed, self.is_training: True })
                    self.writer.add_summary(summary, logcounter)

                # train generator
                _, summary = self.sess.run(
                    [ self.G_train, self.G_stats],
                    feed_dict={ self.X: xfeed, self.Z: zfeed, self.is_training: True })
                self.writer.add_summary(summary, logcounter)

                logcounter += 1

                if (batch % self.save_freq == 0):
                    printnow('Epoch %s, batch %s/%s, saving session and examples' % (epoch, batch, batches))
                    # update TF epoch variable so restart of process picks up at same
                    # epoch where it died
                    self.sess.run(self.epoch.assign(epoch))
                    self.save_session()
                    self.output_examples()

            epoch += 1 

    def save_session(self):
        self.saver.save(self.sess, self.checkpoints_path)

    def output_examples(self):
        cols = self.output_cols
        rows = self.output_rows
        nimgs = cols*rows
        zfeed = self.example_noise.eval() # need to eval to get value since it's a tf variable 
        imgs = self.sess.run(self.Gz, feed_dict={ self.Z: zfeed, self.is_training: False })
        imgs = imgs[:nimgs]
        # conver [-1,1] back to [0,1] before saving
        imgs = pixels01(imgs)
        path = os.path.join(self.dirs['output'], '%06d.jpg' % self.output_img_idx)
        tiled = tile(imgs, (rows, cols))
        as_ints = (tiled * 255.0).astype('uint8')
        Image.fromarray(as_ints).save(path)
        self.output_img_idx += 1 

    


        
        





