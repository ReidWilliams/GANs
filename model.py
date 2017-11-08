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
    def __init__(self, training_directory, batch_size=64, img_shape=(64, 64),
        E_lr=0.0004, G_lr=0.0004, D_lr=0.0004, E_beta1=0.5, G_beta1=0.5, D_beta1=0.5, 
        gamma=0.01, zsize=128, save_freq=10, epochs=10000, 
        sess=None, checkpoints_path=None):

        self.batch_size = batch_size
        self.img_shape = img_shape + (3,) # add channels
        self.E_lr = E_lr
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.E_beta1 = E_beta1
        self.G_beta1 = G_beta1
        self.D_beta1 = D_beta1

        # weights strength of similarity loss compared 
        # to discriminator classification loss
        self.gamma = gamma
        # size of latent vector
        self.zsize = zsize
        self.epochs = epochs
        # save session and examples after this many batches
        self.save_freq = int(save_freq)

        pwd = os.getcwd()
        self.dirs = {
            'training':    training_directory,
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
        self.feed = Feed(self.dirs['training'], self.batch_size, shuffle=True)
        # bool used by batch normalization. BN behavior is different when training
        # vs predicting
        self.is_training = tf.placeholder(tf.bool)
        self.arch = VAEGAN(self.is_training, img_shape=self.img_shape, zsize=128)

    def build_model(self):
        self.X = tf.placeholder(tf.float32, (None,) + self.img_shape)
        # for feeding random draws of z (latent variable)
        self.Z = tf.placeholder(tf.float32, (None, self.zsize))

        # E for encoder
        self.E, self.E_logsigmas, self.E_means = self.arch.encoder(self.X)

        # generator that uses encoder output
        self.Genc = self.arch.generator(self.E)
        # generator that uses Z random draws
        self.Gz = self.arch.generator(self.Z, reuse=True)

        # discriminator connected to real image input (X)
        self.Dreal, self.Dreal_logits, self.Dreal_similarity = \
            self.arch.discriminator(self.X)

        # discriminator connected to X -> encoder -> generator
        self.Denc, self.Denc_logits, self.Denc_similarity = \
            self.arch.discriminator(self.Genc, reuse=True)

        # discriminator connected to Z -> generator
        self.Dz, self.Dz_logits, _ = \
            self.arch.discriminator(self.Gz, reuse=True)

    def build_losses(self):
        self.Dreal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=(tf.ones_like(self.Dreal_logits) - 0.25),
            logits=self.Dreal_logits))

        self.Denc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.Denc_logits),
            logits=self.Denc_logits))

        self.Dz_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.Dz_logits),
            logits=self.Dz_logits))

        # minimize these with optimizer
        self.D_loss = self.Dreal_loss + self.Denc_loss + self.Dz_loss

        # similarity loss according to discriminator
        self.D_similarity_loss = tf.reduce_mean(
            tf.square(self.Dreal_similarity - self.Denc_similarity))

        # pixelwise similarity loss
        self.pixel_similarity_loss = tf.reduce_mean(
            tf.square(self.X - self.Genc))

        # how much does the GAN like the decoder's output
        self.style_loss = \
            0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.Denc_logits),
                logits=self.Denc_logits)) + \
            0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.Dz_logits),
                logits=self.Dz_logits))

        # can include pixelwise, similarity, style
        # self.G_loss = self.pixel_similarity_loss
        self.G_loss = self.gamma * self.D_similarity_loss + self.style_loss
        # self.G_loss = self.D_similarity_loss

        self.latent_loss = self.arch.latent_loss(self.E_logsigmas, self.E_means)
        # can include latent loss, pixelwise loss, similarity
        # self.E_loss = self.pixel_similarity_loss + self.latent_loss
        self.E_loss = self.D_similarity_loss + self.latent_loss

    def build_optimizers(self):
        E_vars = [i for i in tf.trainable_variables() if 'encoder' in i.name]
        G_vars = [i for i in tf.trainable_variables() if 'generator' in i.name]
        D_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]
  
        E_opt = tf.train.AdamOptimizer(learning_rate=self.E_lr, beta1=self.E_beta1)
        G_opt = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.G_beta1)
        D_opt = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.D_beta1)
        
        self.E_train = E_opt.minimize(self.E_loss, var_list=E_vars)
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

        self.E_stats = tf.summary.merge([
            tf.summary.scalar('E_loss', self.E_loss),
            tf.summary.scalar('latent_loss', self.latent_loss)
        ])

        self.G_stats = tf.summary.merge([
            tf.summary.scalar('D_similarity_loss', self.D_similarity_loss),
            tf.summary.scalar('pixel_similarity_loss', self.pixel_similarity_loss),
            tf.summary.scalar('style_loss', self.style_loss),
            tf.summary.scalar('G_loss', self.G_loss)

        ])

        Dreal_mean = tf.reduce_mean(tf.sigmoid(self.Dreal_logits))
        Denc_mean = tf.reduce_mean(tf.sigmoid(self.Denc_logits))
        Dz_mean = tf.reduce_mean(tf.sigmoid(self.Dz_logits))

        self.D_stats = tf.summary.merge([
            tf.summary.scalar('Dreal_out', Dreal_mean),
            tf.summary.scalar('Denc_out', Denc_mean),
            tf.summary.scalar('Dz_out', Dz_mean),
            tf.summary.scalar('D_loss', self.D_loss)
        ])

    def train(self):
        batches = self.feed.nbatches()
        printnow('training with %s batches per epoch' % batches)
        printnow('saving session and examples every %s batches' % self.save_freq)
        logcounter = 0

        # images to encode for saving examples
        example_feed = np.copy(self.feed.feed(21))

        for epoch in range(self.epochs):            
            for batch in range(batches):
                xfeed = pixels11(self.feed.feed(batch)) # conver to [-1, 1]
                zfeed = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')

                # train discriminator
                _, summary = self.sess.run(
                    [ self.D_train, self.D_stats ],
                    feed_dict={ self.X: xfeed, self.Z: zfeed, self.is_training: True })
                self.writer.add_summary(summary, logcounter)

                # train generator
                _, summary = self.sess.run(
                    [ self.G_train, self.G_stats],
                    feed_dict={ self.X: xfeed, self.Z: zfeed, self.is_training: True })
                self.writer.add_summary(summary, logcounter)

                # train encoder
                _, summary = self.sess.run(
                    [self.E_train, self.E_stats],
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
        # feed = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')
        imgs = self.sess.run(self.Genc, feed_dict={ self.X: feed, self.is_training: False })
        path = os.path.join(self.dirs['output'], '%06d.jpg' % self.output_img_idx)
        tiled = tile(imgs, (8, 8))
        as_ints = (pixels01(tiled) * 255.0).astype('uint8')
        Image.fromarray(as_ints).save(path)
        self.output_img_idx += 1 

    


        
        





