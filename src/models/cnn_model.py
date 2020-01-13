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
        self.conv3 = torch.nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5
        )
        self.conv4 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5
        )
        self.conv5 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=5
        )
        self.conv6 = torch.nn.Conv1d(
            in_channels=512,
            out_channels=1024,
            kernel_size=5
        )
        self.linear1 = torch.nn.Linear(
            in_features=2560,
            out_features=len(self.config.targets)
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.linear1(x)
        return x
