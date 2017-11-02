import numpy as np
import scipy as sp
import os
import sys

from utils import pixels11

class Feed:
	'''Feed image data to training process. '''
	def __init__(self, data_directory, zsize, batch_size, ncached_batches=100):
		self.data_directory = data_directory
		self.zsize = zsize
		self.batch_size = batch_size
		# number of batches to preload into memory
		self.ncached_batches = ncached_batches

		# filenames for all files in data dir
		self.filenames = [f for f in os.listdir(self.data_directory) \
			if os.path.isfile(os.path.join(self.data_directory, f))]

		# index of first batch preloaded in memory
		self.cached_batch_start = -sys.maxsize

	# translate from global batch index to cached batch index
	# load more data if necessary
	def cidx(self, batch_idx):
		# batch_idx outside range of cached batches?
		if (batch_idx < self.cached_batch_start or 
			batch_idx >= self.cached_batch_start + self.ncached_batches):

			# new cached_batch_start
			self.cached_batch_start = self.ncached_batches * \
				int(batch_idx / float(self.ncached_batches))
			# preload more batches
			print('cached b start: %s' % self.cached_batch_start)
			self.load_cache(self.cached_batch_start)

		# index of batch in currently preloaded data
		return batch_idx % self.ncached_batches

	def load_cache(self, batch_idx):
		# last valid index given dataset size
		last_batch_idx = int(len(self.filenames) / float(self.batch_size))
		# end of cache
		end_batch_idx = min((batch_idx + self.ncached_batches), last_batch_idx)

		start = batch_idx * self.batch_size
		end = end_batch_idx * self.batch_size

		# full paths
		cache_filepaths = [os.path.join(self.data_directory, f) for f in self.filenames[start:end]]

		self.imgs = np.array([sp.ndimage.imread(f) for f in cache_filepaths])
		self.cached_batch_start = batch_idx

	def feed(self, batch_idx):
		''' Returns images and noise. User needs to ensure that batch_idx + batch_size doesn't exceed
		last image. I.e. restrict batch idx to integer multiple of total images.'''
		cidx = self.cidx(batch_idx)

		print('batch index: %s' % cidx)
		
		imgs = self.imgs[ cidx*self.batch_size:(cidx+1)*self.batch_size ]
		noise = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')

		# make sure images are float32 between 0 and 1
		assert imgs.dtype == 'uint8' or imgs.dtype == 'float32'
		if (imgs.dtype == 'uint8'):
			imgs = imgs.astype('float32') / 255.0

		print('pixels11')
		assert imgs.shape[0] > 0

		return imgs, noise

		








