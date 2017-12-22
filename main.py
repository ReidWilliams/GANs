import argparse

from model import Model
from model_wgan import ModelWGAN
from feed import Feed

parser = argparse.ArgumentParser()
# Directory for training data. Model assumes every file in the data directory
# is a training image (including hidden files)
parser.add_argument('--datadir', required=True, dest='datadir')

# How often to checkpoint the training weights and output images 
parser.add_argument('--savefreq', default=10, type=int, dest='save_freq')
parser.add_argument('--batchsize', default=64, type=int, dest='batch_size')

# Latent vector size
parser.add_argument('--zsize', default=128, type=int, dest='zsize')

# Rows and cols of output image tile. Several output images get tiled into
# one image that is saved 
parser.add_argument('--output_cols', default=8, type=int, dest='output_cols')
parser.add_argument('--output_rows', default=8, type=int, dest='output_rows')
parser.add_argument('--wgan', action='store_true', dest='wgan')
parsed = parser.parse_args()

# create data feed and get dims
feed = Feed(parsed.datadir, parsed.batch_size, shuffle=True)
img_shape = feed.get_img_shape()

model = None
args = {
	'save_freq':   parsed.save_freq, 
  	'batch_size':  parsed.batch_size, 
  	'img_shape':   img_shape,
  	'zsize':       parsed.zsize,
  	'output_cols': parsed.output_cols,
  	'output_rows': parsed.output_rows
}

if (parsed.wgan):
	# use WGAN loss (see model_wgan.py for more details)
	model = ModelWGAN(feed, **args) 
else:
	model = Model(feed, **args) 
		
model.build_model()
model.build_losses()
model.build_optimizers()
model.setup_session()
model.setup_logging()
model.train()
