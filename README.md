# AuxiFed: Resilient Federated Adversarial Learning

This repository contains the implementation of **AuxiFed**, a resilient federated adversarial learning framework using **Auxiliary-Classifier GANs (AC-GANs)** and probabilistic synthesis for heterogeneous environments. The framework is designed to enhance model robustness against adversarial attacks while improving generalization in federated learning (FL) settings.

## Features
- **Federated Learning** with adversarial robustness techniques (FGSM, PGD).
- **AC-GAN based synthetic data generation** for better model generalization.
- **Configurable training parameters** through a configuration file.
- **Support for MNIST and EMNIST datasets** in homogeneous and heterogeneous settings.

## Installation
To set up the environment, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Running the main script
```bash
python main.py --dataset Mnist --alpha 0.2
```

### Configuration File
The configuration settings are handled using `config.py`. Modify the values within `Config` class to adjust parameters for training, dataset selection, model settings, and attack parameters.

Example usage in `main.py`:
```python
from config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mnist', help='Dataset type: Mnist or EMnist')
parser.add_argument('--alpha', type=float, default=0.2, help='Dirichlet distribution parameter for data heterogeneity')
parser.add_argument('--local_iter', type=int, default=4, help='Number of local training iterations per round')
parser.add_argument('--rounds', type=int, default=4, help='Number of global communication rounds in federated learning')
parser.add_argument('--gan_epochs', type=int, default=50, help='Number of epochs for GAN training')
parser.add_argument('--adverseial_training', type=bool, default=False, help='Enable adversarial training (True/False)')
parser.add_argument('--attack', type=str, default='fgsm', help='Type of adversarial attack (fgsm, pgd, etc.)')
parser.add_argument('--adverseial_training_probs', type=float, default=0.2, help='Probability of using adversarial training samples')
parser.add_argument('--replace_probs', type=float, default=16, help='Probability of replacing real samples with synthetic samples')

args = parser.parse_args()

config = Config(DATASETTYPE=args.dataset, alpha=args.alpha, local_iter=args.local_iter, rounds=args.rounds, gan_epochs=args.gan_epochs, adverseial_training=args.adverseial_training, adverseial_training_probs=args.adverseial_training_probs, replace_probs=args.replace_probs)
```

## Dependencies
All dependencies are listed in `requirements.txt` and include:
```text
matplotlib
numpy
torch
torchvision
tqdm
argparse
json
os
``` 

## Citation
If you use this code, please cite our paper:
> **Resilient Federated Adversarial Learning with AC-GANs and Probabilistic Synthesis**

## License
This project is licensed under the MIT License.
