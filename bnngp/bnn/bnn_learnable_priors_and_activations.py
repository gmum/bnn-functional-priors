import torch
import numpy as np


class SingleHiddenLayerWideNNWithLearnablePriorsAndActivation(torch.nn.Module):
    def __init__(self, width=1000, indim=1, outdim=1, learnable_activation=None):
        super().__init__()
        self.width = width
        self.fc1 = torch.nn.Linear(indim, width)
        self.fc2 = torch.nn.Linear(width, outdim)
        self.learnable_activation = learnable_activation

    def forward(self, x, learnable_activation=None):
        learnable_activation = self.learnable_activation or learnable_activation

        x = self.fc1(x)
        x = learnable_activation(x)
        x = self.fc2(x)
        return x
