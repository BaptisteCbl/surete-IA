import torch
import torch.nn as nn
import torch.nn.functional as F


class fcn(torch.nn.Module):
    """Basic fcn architecture."""

    def __init__(self, in_channels):
        super().__init__()
        self.linear1 = nn.Linear(3 * 32 * 32, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten images into vectors
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        return out
