import argparse

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, dest='datadir')
parser.add_argument('--savefreq', default=10, dest='save_freq')
parser.add_argument('--batchsize', default=64, dest='batch_size')
parsed = parser.parse_args()

model = Model(parsed.datadir, save_freq=parsed.save_freq, 
    batch_size=parsed.batch_size, img_shape=(256, 448))
model.build_model()
model.build_losses()
model.build_optimizers()
model.setup_session()
model.setup_logging()
model.train()
