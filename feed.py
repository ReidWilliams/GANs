import numpy as np
from PIL import Image
import os
import sys

class Feed:
	'''Feed image data to training process. '''
	def __init__(self, data_directory, batch_size, ncached_batches=100):
		self.data_directory = data_directory
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
			self.load_cache(self.cached_batch_start)

		# index of batch in currently preloaded data
		return batch_idx % self.ncached_batches

	def nbatches(self):
		return int(len(self.filenames) / float(self.batch_size))

	def load_cache(self, batch_idx):
		# end of cache
		end_batch_idx = min((batch_idx + self.ncached_batches), self.nbatches())

		start = batch_idx * self.batch_size
		end = end_batch_idx * self.batch_size

		# full paths
		cache_filepaths = [os.path.join(self.data_directory, f) for f in self.filenames[start:end]]

		imgs = []
		for i in range(len(cache_filepaths)):
			img = Image.open(cache_filepaths[i])
			ar = np.copy(np.array(img))
			if (len(ar.shape) < 3):

				print('weird image')
				print(cache_filepaths[i])

			imgs.append()
			img.close()

		self.imgs = np.array(imgs)
		self.cached_batch_start = batch_idx

	def feed(self, batch_idx):
		''' Returns images and noise. User needs to ensure that batch_idx + batch_size doesn't exceed
		last image. I.e. restrict batch idx to integer multiple of total images.'''
		cidx = self.cidx(batch_idx)	
		imgs = self.imgs[ cidx*self.batch_size:(cidx+1)*self.batch_size ]

		if (imgs.dtype == 'float64'):
			imgs = imgs.astype('float32')
			
		if (imgs.dtype == 'uint8'):
			imgs = imgs.astype('float32') / 255.0

		assert imgs.shape[0] > 0
		return imgs

		








