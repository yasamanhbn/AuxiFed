import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
class ACDiscriminator(nn.Module):
  def __init__(self, config):

    super(ACDiscriminator,self).__init__()
    def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

    self.conv_blocks = nn.Sequential(
            *discriminator_block(config.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = math.ceil(config.img_size / (2 ** 4))

    # Output layers
    self.adv_layer = nn.Sequential(nn.Linear(128 * (ds_size ** 2), 1), nn.Sigmoid())
    self.aux_layer = nn.Sequential(nn.Linear(128 * (ds_size ** 2), config.CLASS_NUM), nn.Softmax(dim=1))

  def forward(self, img):
      out = self.conv_blocks(img)
      out = out.view(out.shape[0], -1)
      validity = self.adv_layer(out)
      label = self.aux_layer(out)
      return validity, label