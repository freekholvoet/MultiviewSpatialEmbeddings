import torch
from torch import nn
import numpy as np
import pyproj

"""
Encode coordinates using W3034
"""
class W3034(nn.Module):
    def __init__(self):
        super(W3034, self).__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 2

    def forward(self, coords):
        
        return coords