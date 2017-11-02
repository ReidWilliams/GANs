import argparse

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--datadir')
parsed = parser.parse_args()

model = Model(parsed.datadir)
model.build_model()
model.build_losses()
model.build_optimizer()
model.setup_session()
model.setup_logging()
model.train()
