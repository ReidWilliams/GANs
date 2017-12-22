import numpy as np
from PIL import Image
import os
import sys

# Loads and returns batches of training data. Abstract this into a class
# that preloads a bunch of images in advance, but doesn't preload the entire
# training set in advance.

class Feed:
	'''Feed image data to training process. '''
	def __init__(self, data_directory, batch_size, ncached_batches=100, shuffle=False):
		self.data_directory = data_directory
		self.batch_size = batch_size
		# number of batches to preload into memory
		self.ncached_batches = ncached_batches

		# filenames for all files in data dir
		self.filenames = sorted([f for f in os.listdir(self.data_directory) \
			if os.path.isfile(os.path.join(self.data_directory, f))])

		if (shuffle):
			np.random.shuffle(self.filenames)

		# index of first batch preloaded in memory
		self.cached_batch_start = -sys.maxsize

	# figure out image shape from the first image
	def get_img_shape(self):
		path = os.path.join(self.data_directory, self.filenames[0])
		img = np.asarray(self.open_image(path))
		return (img.shape[0], img.shape[1])

	# convert from global batch index (ie. between 0 and total number of 
	# batches in the entire training set) to corresponding cached batch index (number between
	# 0 and number of batches worth that get cached)
	# Also loads more data if batch_idx is outide what is currently cached
	def cidx(self, batch_idx):
		# batch_idx outside range of cached batches?
		if (batch_idx < self.cached_batch_start or 
			batch_idx >= self.cached_batch_start + self.ncached_batches):

			# new cached_batch_start
			self.cached_batch_start = self.ncached_batches * \
				int(batch_idx / float(self.ncached_batches))
			# preload more batches
			self.load_cache(self.cached_batch_start)

		# index of batch in currently preloaded data
		return batch_idx % self.ncached_batches

	# number of batches in entire directory	
	def nbatches(self):
		return int(len(self.filenames) / float(self.batch_size))

	# loads and returns np array of image, converting grayscale images
	# to RGB if necessary
	def open_image(self, f):
		img = Image.open(f)
		array = np.asarray(img)
		if (len(array.shape) == 2): # only 2 dims, no color dim, so grayscale
			rgbimg = Image.new("RGB", img.size)
			rgbimg.paste(img)
			array = np.asarray(rgbimg)
		return array


	# do the actual loading from disk	
	def load_cache(self, batch_idx):
		# end of cache
		end_batch_idx = min((batch_idx + self.ncached_batches), self.nbatches())

		start = batch_idx * self.batch_size
		end = end_batch_idx * self.batch_size

		# full paths
		cache_filepaths = [os.path.join(self.data_directory, f) for f in self.filenames[start:end]]

		self.imgs = np.asarray([self.open_image(f) for f in cache_filepaths])
		self.cached_batch_start = batch_idx

	# public method, returns the next batch_size worth of images
	def feed(self, batch_idx):
		cidx = self.cidx(batch_idx)	
		imgs = self.imgs[ cidx*self.batch_size:(cidx+1)*self.batch_size ]

		if (imgs.dtype == 'float64'):
			imgs = imgs.astype('float32')
			
		if (imgs.dtype == 'uint8'):
			imgs = imgs.astype('float32') / 255.0

		assert imgs.shape[0] > 0
		return imgs

		








