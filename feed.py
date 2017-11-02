import numpy as py
import scipy as sp
import os

from utils import pixels11

class Feed:
	'''Feed image data to training process. '''
	def __init__(self, data_directory, zsize, batch_size, ncached_batches=100):
		self.data_directory = data_directory
		self.zsize = zsize
		self.batch_size = batch_size
		# number of batches to preload into memory
		self.ncached_batches = ncached_batches

		# index of first batch preloaded in memory
		self.cached_batch_start = 0
		self.load_cache(0)


	# translate from global batch index to cached batch index
	# load more data if necessary
	def cidx(batch_idx):
		if (batch_idx > self.cached_batch_start + self.ncached_batches):
			# preload more batches
			self.load_cache(batch_idx)

		# index of batch in currently preloaded data
		return batch_idx % self.ncached_batches

	def load_cache(batch_idx):
		filenames = [f for f in os.listdir(self.data_directory) \
			if os.path.isfile(os.path.join(self.data_directory, f))]
		# full paths
		filepaths = [os.path.join(self.data_directory, f) for f in filenames]

		start = batch_idx * self.batch_size
		nfiles = self.ncached_batches * self.batch_size

		imgfiles = [sp.ndimage.imread(f) for f in filepaths]
		self.imgs = pixels11(np.array(imgfiles))

	def feed(batch_idx):
		''' Returns images and noise. User needs to ensure that batch_idx + batch_size doesn't exceed
		last image. I.e. restrict batch idx to integer multiple of total images.'''
		cidx = self.cidx(batch_idx)
		imgs = self.imgs[ batch_idx*self.batch_size:(batch_idx+1)*self.batch_size ]
		noise = np.random.normal(size=(self.batch_size, self.zsize)).astype('float32')

		return imgs, noise

		








