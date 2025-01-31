import torch.nn as nn
import torch.nn.functional as F
from utils import *

class CNN(nn.Module):
    def __init__(self, config):
      super(CNN, self).__init__()
      self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
      self.bn1 =  nn.BatchNorm2d(8)
      self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
      self.bn2 =  nn.BatchNorm2d(16)
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.fc1 = nn.Linear(FLATTEN_SIZE, 16)
      self.fc2 = nn.Linear(16, config.CLASS_NUM)
      self.dropout = nn.Dropout(0.25)

    def forward(self, input, original_data=[], labels=[], class_lens=[], type="train"):
        """
        Form the Feed Forward Network by combininig all the layers
        :param x: the input image for the network
        """
        x = input.clone()
        if type == 'train':
          original_data.requires_grad = True
          for idx, i,  in enumerate(labels):
            if class_lens[i] < 800:
                x[idx] = original_data[idx]

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        pred = F.log_softmax(x, dim=1)
        return pred