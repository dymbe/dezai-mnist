from abc import ABC

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module, ABC):
    def __init__(self):
        super(Net, self).__init__()
        self.fltn = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fltn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
