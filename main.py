from Config import *
from train import train
from data import *
import argparse

if __name__=="__main__":

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

    generate_data(config)
    train(config)