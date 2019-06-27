import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, image_size, latent_size, hidden_size):
        super(Generator, self).__init__()

        self.net = nn.Sequential(nn.Linear(latent_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, image_size),
                        nn.Tanh())


    def forward(self, x):#1ページ分のオブジェクトパラメータを入力する
        out = self.net(x)
        return out
        