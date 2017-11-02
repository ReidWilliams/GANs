import os

from feed import Feed
from architecture import VAEGAN

def printnow(x, end='\n'): print(x, flush=True, end=end)

class Model:
    def __init__(self, training_directory, batch_size=64, img_shape=(64, 64), \
        learning_rate=0.0002, learning_beta1=0.5, gamma=0.01, zsize=128, \
        epochs=10000):

        self.batch_size = batch_size
        self.img_shape = img_shape + (3,) # add channels
        self.learning_rate = learning_rate
        self.beta1 = learning_beta1
        
        # weights strength of similarity loss compared 
        # to discriminator classification loss
        self.gamma = gamma
        # size of latent vector
        self.zsize = zsize

        self.dirs = {
            'training':    training_directory,
            'output':      os.path.join(pwd, 'output'),
            'logs':        os.path.join(pwd, 'logs'),
            'checkpoints': os.path.join(pwd, 'checkpoints')
        }

        # create directories. Fail if logs exist
        os.makedirs(dirs['logs'])
        if not os.path.exists(dirs['output']):
            os.makedirs(dirs['output'])
        if not os.path.exists(dirs['checkpoints']):
            os.makedirs(dirs['checkpoints'])

        # get number of files in output so we don't overwrite
        self.output_img_idx = len([f for f in os.listdir(self.data_directory) \
            if os.path.isfile(os.path.join(self.data_directory, f))])

        # data feed    
        self.feed = Feed(self.dirs['training'], self.zsize, self.batch_size)
        self.arch = VAEGAN(self.is_training, img_shape=self.img_shape, zsize=128)

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
        # for feeding random draws of z (latent variable)
        self.Z = tf.placeholder(tf.float32, [None, zsize])
        self.is_training = tf.placeholder(tf.bool)

        # E for encoder
        self.E, self.E_logsigmas, self.E_means = self.arch.encoder(X)

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
            labels=(tf.ones_like(self.Dreal_logits) - 0.1),
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
                labels=tf.ones_like(Denc_logits),
                logits=Denc_logits)) + \
            0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(Dz_logits),
                logits=Dz_logits))

        # can include pixelwise, similarity, style
        self.G_loss = self.pixel_similarity_loss

        self.latent_loss = VAEGAN.latent_loss(self.E_logsigmas, self.E_means)
        # can include latent loss, pixelwise loss, similarity
        self.E_loss = self.pixel_similarity_loss + self.latent_loss

    def build_optimizers(self):
        E_vars = [i for i in tf.trainable_variables() if 'encoder' in i.name]
        G_vars = [i for i in tf.trainable_variables() if 'generator' in i.name]
        D_vars = [i for i in tf.trainable_variables() if 'discriminator' in i.name]
  
        E_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.learning_beta1)
        G_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.learning_beta1)
        D_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.learning_beta1)
        
        E_train = E_opt.minimize(self.E_loss, var_list=E_vars)
        G_train = G_opt.minimize(self.G_loss, var_list=G_vars)
        D_train = D_opt.minimize(self.D_loss, var_list=D_vars)

    def setup_session(self):
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        try:
            print('trying to restore session from %s' % self.dirs['checkpoints'])
            saver.restore(self.sess, self.dirs['checkpoints'])
            print('restored session')
        except:
            print('failed to restore session, creating a new one')
            tf.global_variables_initializer().run()

    def setup_logging(self):
        writer = tf.summary.FileWriter(self.dirs['logs'], self.sess.graph)

        E_stats = tf.summary.merge([
            tf.summary.scalar('E_loss', self.E_loss),
            tf.summary.scalar('latent_loss', self.latent_loss)
        ])

        G_stats = tf.summary.merge([
            tf.summary.scalar('D_similarity_loss', self.D_similarity_loss),
            tf.summary.scalar('pixel_similarity_loss', self.pixel_similarity_loss),
            tf.summary.scalar('style_loss', self.style_loss),
            tf.summary.scalar('G_loss', self.G_loss)

        ])

        Dreal_mean = tf.reduce_mean(tf.sigmoid(Dreal_logits))
        Denc_mean = tf.reduce_mean(tf.sigmoid(Denc_logits))
        Dz_mean = tf.reduce_mean(tf.sigmoid(Dz_logits))

        D_stats = tf.summary.merge([
            tf.summary.scalar('Dreal_out', Dreal_mean),
            tf.summary.scalar('Denc_out', Denc_mean),
            tf.summary.scalar('Dz_out', Dz_mean),
            tf.summary.scalar('D_loss', D_loss)
        ])

    def train(self):
        batches = self.feed.nbatches()
        printnow('training over %s batches per epoch' % batches)
        logcounter = 0

        # images to encode for saving examples
        example_feed = np.copy(self.feed.feed(0))

        for epoch in range(epochs):            
            for batch in range(batches):
                xfeed, zfeed = self.feed.feed(batch)

                # train discriminator
                _, summary = self.sess.run(
                    [ self.D_train, D_stats ],
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
                writer.add_summary(summary, logcounter)

                logcounter += 1

                if (batch % 20 == 0):
                    printnow('Epoch %s, batch %s/%s' % (epoch, batch, batches))
                    self.save_session()
                    self.output_examples(example_feed)

    def save_session(self):
        printnow('saving session')
        self.saver.save(sess, self.dirs['checkpoints'])

    def output_examples(feed):
        imgs = self.sess.run(self.Genc, feed_dict={ self.X: feed, self.is_training: False })
        path = os.path.join(self.dirs['output'], '%06d.jpg' % self.output_img_idx)
        sp.misc.imsave(path, pixels01(imgs[0]))
        print('saving with pixelation!???')
        self.output_img_idx += 1 

    


        
        





