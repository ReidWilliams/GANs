img_directory = '/home/ec2-user/img_align_celeba'
model_save_path = '/home/ec2-user/checkpoints/vae-celeba.ckpt'
img_save_directory = '/home/ec2-user/vae-celeba-out'
batch_size = 64
training_set_size = 5000
img_size = 128

import numpy as np
import scipy as sp
import os
from utils import imshow, resize_crop, load_img

training = np.array([resize_crop(load_img(i+1, img_directory), (img_size, img_size)) for i in range(training_set_size)])

# Create model and load weights
import tensorflow as tf

from autoencoder import Autoencoder
vae = Autoencoder(img_shape=(img_size, img_size, 3))

X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
encoder = vae.encoder(X)
decoder = vae.decoder(encoder)

latent_loss = vae.latent_loss()
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=vae.logits)
reconstruction_loss = tf.reduce_mean(xentropy)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()

# collect data for tensorboard
loss_summary = tf.summary.scalar('loss', loss)
merged_summary = tf.summary.merge_all()

sess = tf.InteractiveSession()
try:
	print('trying to restore session')
	saver.restore(sess, model_save_path)
except:
	print('failed to restore session, creating a new one')
	tf.global_variables_initializer().run()

import math
epochs = 10000
batches = int(float(training_set_size) / batch_size) 

for epoch in range(epochs):
	print ('epoch %s ' % epoch, end='', flush=True)

	for batch in range(batches):
	  feed = training[batch*batch_size:(batch+1)*batch_size]
	  sess.run(loss, feed_dict={X: feed})
	  print ('.', end='', flush=True)

	if (epoch % 10 == 0):
	  print('saving session' flush=True)
	  saver.save(sess, model_save_path)

	  xfeed = training[:batch_size]
	  summary = merged_summary.eval(feed_dict={X: xfeed})
	  writer.add_summary(summary, epoch) 

	  example = decoder.eval(feed_dict={X: training[:1]})
	  img_save_path = os.path.join(img_save_directory, '%06d.jpg' % epoch)
	  sp.misc.imsave(img_save_path, example[0])