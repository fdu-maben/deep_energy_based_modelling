import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random as random
import torchvision

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class JEMClassifier(nn.Module):
    def __init__(self, hidden_features=32):
        super().__init__()
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),  # [16x16]
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  # [8x8]
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),  # [2x2]
            Swish(),
            nn.Flatten()
        )
        self.fc_class = nn.Linear(c_hid3 * 4, num_classes)

    def forward(self, x):
        features = self.cnn_layers(x)
        logits = self.fc_class(features)
        energy = torch.log(torch.sum(torch.exp(logits),dim=-1))
        return energy, logits

