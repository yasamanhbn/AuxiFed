from Config import *
from train import train
from data import *

if __name__=="__main__":
    config = Config(DATASETTYPE='EMnist', alpha=1)
    # generate_data(config)
    train(config)