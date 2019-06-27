import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, image_size, hidden_size):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(nn.Linear(image_size, hidden_size),
                                nn.LeakyReLU(0.2),
                                nn.Linear(hidden_size, hidden_size),
                                nn.LeakyReLU(0.2),
                                nn.Linear(hidden_size, 1),
                                nn.Sigmoid())


    def forward(self, x):
        out = self.net(x)
        return out

