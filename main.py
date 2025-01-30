from Config import *
from train import train
from data import *

config = Config(DATASETTYPE='EMnist', alpha=1)
generate_data(config)
train()