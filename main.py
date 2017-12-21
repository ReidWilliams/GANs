import argparse

from model import Model
from feed import Feed

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, dest='datadir')
parser.add_argument('--savefreq', default=10, type=int, dest='save_freq')
parser.add_argument('--batchsize', default=64, type=int, dest='batch_size')
parser.add_argument('--zsize', default=128, type=int, dest='zsize')
parser.add_argument('--output_cols', default=8, type=int, dest='output_cols')
parser.add_argument('--output_rows', default=8, type=int, dest='output_rows')
parsed = parser.parse_args()

# create data feed and get dims
feed = Feed(parsed.datadir, parsed.batch_size, shuffle=True)
img_shape = feed.get_img_shape()

model = Model(
	feed, 
	save_freq=parsed.save_freq, 
  	batch_size=parsed.batch_size, 
  	img_shape=img_shape,
  	zsize=parsed.zsize,
  	output_cols=parsed.output_cols,
  	output_rows=parsed.output_rows
  	)

model.build_model()
model.build_losses()
model.build_optimizers()
model.setup_session()
model.setup_logging()
model.train()
