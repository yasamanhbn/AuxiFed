from torchvision import datasets, transforms
from utils import *

def load_test_dataset(config, batch_size):
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # load the dataset
    if config.DATASETTYPE == 'Mnist':
        test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    elif config.DATASETTYPE == 'EMnist':
        test_dataset = datasets.EMNIST(root=DATA_DIR, split='letters', train=False, download=True, transform=transform)

    print("Test Set size: ", len(test_dataset))
    return test_dataset



