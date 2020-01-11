import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnNet(nn.Module):
    def __init__(self, config):
        super(CnnNet, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            in_channels=len(self.config.features),
            out_channels=32,
            kernel_size=5
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5
        )
        self.linear1 = torch.nn.Linear(
            in_features=1792,
            out_features=len(self.config.targets)
        )


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.linear1(x)
        return x
