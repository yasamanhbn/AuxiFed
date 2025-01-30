import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
class ACGenerator(nn.Module):
  def __init__(self):
    super(ACGenerator,self).__init__()

    self.label_emb = nn.Embedding(config.CLASS_NUM, config.latent_dim)

    self.init_size = config.img_size // 4  # Initial size before upsampling
    self.l1 = nn.Sequential(nn.Linear(config.latent_dim, 128 * self.init_size ** 2))

    self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, config.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

  def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels.clone().detach().to(torch.int64)), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
