import tensorflow as tf
from PIL import Image
import numpy as np
import os

from feed import Feed
from architecture import VAEGAN
from utils import pixels01, pixels11, tile

def printnow(x, end='\n'): print(x, flush=True, end=end)
def makedirs(d): 
    if not os.path.exists(d): os.makedirs(d) 
          

class Model:
    def __init__(self, feed, batch_size=64, img_shape=(64, 64),
        G_lr=0.0004, D_lr=0.0004, G_beta1=0.5, D_beta1=0.5, 
        zsize=128, save_freq=10, epochs=10000, 
        sess=None, checkpoints_path=None):

        self.batch_size = batch_size

        if ((img_shape[0] % 32 != 0) or (img_shape[1] % 32 != 0)):
            raise ValueException("Image dimensions need to be divisible by 32. \
                Dimensions received was %s." % img_shape)

        self.img_shape = img_shape + (3,) # add channels
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.G_beta1 = G_beta1
        self.D_beta1 = D_beta1

        # size of latent vector
        self.zsize = zsize
        self.epochs = epochs
        # save session and examples after this many batches
        self.save_freq = int(save_freq)

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

        # get number of files in output so we don't overwrite
        self.output_img_idx = len([f for f in os.listdir(self.dirs['output']) \
            if os.path.isfile(os.path.join(self.dirs['output'], f))])

        # data feed    
        self.feed = feed
        # bool used by batch normalization. BN behavior is different when training
        # vs predicting
        self.is_training = tf.placeholder(tf.bool)
        self.arch = VAEGAN(self.is_training, img_shape=self.img_shape, zsize=128)

    def build_model(self):
        self.X = tf.placeholder(tf.float32, (None,) + self.img_shape)
        # for feeding random draws of z (latent variable)
        self.Z = tf.placeholder(tf.float32, (None, self.zsize))

        # generator that uses Z random draws
        self.Gz = self.arch.generator(self.Z)

        # discriminator connected to real image input (X)
        self.Dreal, self.Dreal_logits, self.Dreal_similarity = \
            self.arch.discriminator(self.X)

        # discriminator connected to Z -> generator
        self.Dz, self.Dz_logits, _ = \
            self.arch.discriminator(self.Gz, reuse=True)

    def build_losses(self):
        self.Dreal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=(tf.ones_like(self.Dreal_logits) - 0.25),
            logits=self.Dreal_logits))

        self.Dz_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.Dz_logits),
            logits=self.Dz_logits))

        # minimize these with optimizer
        self.D_loss = self.Dreal_loss + self.Dz_loss

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.Dz_logits),
            logits=self.Dz_logits))

    def build_optimizers(self):
        G_vars = [i for i in tf.trainable_variables() if 'generator' in i.name]
        D_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]
  
        G_opt = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.G_beta1)
        D_opt = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.D_beta1)
        
        self.G_train = G_opt.minimize(self.G_loss, var_list=G_vars)
        self.D_train = D_opt.minimize(self.D_loss, var_list=D_vars)

    def setup_session(self):
        self.saver = tf.train.Saver()
        
        try:
            print('trying to restore session from %s' % self.checkpoints_path)
            self.saver.restore(self.sess, self.checkpoints_path)
            print('restored session')
        except:
            print('failed to restore session, creating a new one')
            tf.global_variables_initializer().run()

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
        logcounter = 0

        example_feed = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')

        for epoch in range(self.epochs):            
            for batch in range(batches):
                xfeed = pixels11(self.feed.feed(batch)) # conver to [-1, 1]
                zfeed = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')

                # train discriminator (twice)
                for i in range(2):
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
                    self.save_session()
                    self.output_examples(example_feed)

    def save_session(self):
        self.saver.save(self.sess, self.checkpoints_path)

    def output_examples(self, feed):
        cols = 4
        # rows = self.batch_size // cols
        rows = 4
        nimgs = cols*rows
        imgs = self.sess.run(self.Gz, feed_dict={ self.Z: feed, self.is_training: False })
        imgs = imgs[:nimgs]
        imgs = pixels01(imgs)
        path = os.path.join(self.dirs['output'], '%06d.jpg' % self.output_img_idx)
        tiled = tile(imgs, (rows, cols))
        as_ints = (tiled * 255.0).astype('uint8')
        Image.fromarray(as_ints).save(path)
        self.output_img_idx += 1 

    


        
        





