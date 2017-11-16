import argparse
import scipy as sp

from model import Model
from feed import Feed

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, dest='datadir')
parser.add_argument('--savefreq', default=10, type=int, dest='save_freq')
parser.add_argument('--batchsize', default=64, type=int, dest='batch_size')
parsed = parser.parse_args()

# create data feed and get dims
feed = Feed(parsed.datadir, parsed.batch_size, shuffle=True)
img_shape = feed.get_img_shape()

model = Model(feed, save_freq=parsed.save_freq, 
    batch_size=parsed.batch_size, img_shape=img_shape)

model.build_model()
model.build_losses()
model.build_optimizers()
model.setup_session()
model.setup_logging()
model.train()
