import argparse

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, dest='datadir')
parser.add_argument('--savefreq', default=20, dest='save_freq')
parsed = parser.parse_args()

model = Model(parsed.datadir, save_freq=parsed.save_freq)
model.build_model()
model.build_losses()
model.build_optimizers()
model.setup_session()
model.setup_logging()
model.train()
