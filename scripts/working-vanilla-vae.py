
# coding: utf-8

# In[10]:


# img_directory = '/Users/rwilliams/Desktop/celeba/training'
img_directory = '/home/ec2-user/training-data/img_align_celeba'
model_save_path = '/home/ec2-user/tf-checkpoints/vae-celeba/checkpoint.ckpt'
outputs_directory = '/home/ec2-user/outputs/vae-celeba'
log_directory = '/home/ec2-user/tf-logs/vae-celeba'
batch_size = 64
training_set_size = 5000
img_size = 128


# In[ ]:


# import packages for jupyter
# import matplotlib.pyplot as plt
# %matplotlib inline


# In[12]:


import numpy as np
import scipy as sp
import os
from utils import imshow, resize_crop, load_img


# In[9]:


# load training data
training = np.array([resize_crop(load_img(i+1, img_directory), (img_size, img_size)) for i in range(training_set_size)])


# # Create model and load weights

# In[13]:


import tensorflow as tf
from autoencoder import Autoencoder
tf.reset_default_graph()

vae = Autoencoder(img_shape=(img_size, img_size, 3))

X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
encoder = vae.encoder(X)
decoder = vae.decoder(encoder)

latent_loss = vae.latent_loss()
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=vae.logits)
reconstruction_loss = tf.reduce_mean(xentropy)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()


# In[14]:


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


# In[15]:


# collect data for tensorboard
latent_loss_summary = tf.summary.scalar('latent_loss', latent_loss)
reconstruction_loss_summary = tf.summary.scalar('reconstruction_loss', reconstruction_loss)
loss_summary = tf.summary.scalar('total_loss', loss)
merged_summary = tf.summary.merge_all()


# In[16]:


import math
epochs = 10000
batches = int(float(training_set_size) / batch_size) 

img_idx = 0
print('training', flush=True)
for epoch in range(epochs):
    
    print ('epoch %s ' % epoch, end='', flush=True)
    for batch in range(batches):
        print('.', end='', flush=True)
        feed = training[batch*batch_size:(batch+1)*batch_size]
        sess.run(training_op, feed_dict={X: feed})
        
    if (epoch % 1 == 0):
        print('saving session', flush=True)
        saver.save(sess, model_save_path)
        
        xfeed = training[:batch_size]
        summary = merged_summary.eval(feed_dict={X: xfeed})
        writer.add_summary(summary, epoch) 

        example = decoder.eval(feed_dict={X: training[:1]})
        img_save_path = os.path.join(outputs_directory, '%06d.jpg' % img_idx)
        img_idx += 1
        sp.misc.imsave(img_save_path, example[0])


# In[ ]:


# idx = 0
# y = sess.run(decoder, feed_dict={X: training[idx:idx+4]})
# y.shape
# imshow(y[0:4])
# imshow(training[idx:idx+4])


# In[ ]:


# r = np.random.normal(size=(8,128), scale=1.0)
# y = sess.run(decoder, feed_dict={encoder: r})
# imshow(y[0:8])

