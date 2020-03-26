import torch
from torch.nn import functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=5,
            out_channels=32,
            kernel_size=3
        )
        self.batchnorm1 = torch.nn.BatchNorm1d(
            num_features=32
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.batchnorm2 = torch.nn.BatchNorm1d(
            num_features=64
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.batchnorm3 = torch.nn.BatchNorm1d(
            num_features=128
        )
        self.conv4 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.batchnorm4 = torch.nn.BatchNorm1d(
            num_features=256
        )
        self.conv5 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=3
        )
        self.batchnorm5 = torch.nn.BatchNorm1d(
            num_features=512
        )
        self.linear1 = torch.nn.Linear(
            in_features=2048,
            out_features=4096
        )
        self.batchnorm6 = torch.nn.BatchNorm1d(
            num_features=4096
        )
        self.linear2 = torch.nn.Linear(
            in_features=4096,
            out_features=2048
        )
        self.batchnorm7 = torch.nn.BatchNorm1d(
            num_features=2048
        )
        self.linear3 = torch.nn.Linear(
            in_features=2048,
            out_features=1024
        )
        self.batchnorm8 = torch.nn.BatchNorm1d(
            num_features=1024
        )
        self.linear4 = torch.nn.Linear(
            in_features=1024,
            out_features=8
        )

    def forward(self, x):
        x = F.max_pool1d(F.leaky_relu(self.conv1(x)), 2)
        x = self.batchnorm1(x)
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = self.batchnorm2(x)
        x = F.max_pool1d(F.leaky_relu(self.conv3(x)), 2)
        x = self.batchnorm3(x)
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)
        x = self.batchnorm4(x)
        x = F.max_pool1d(F.leaky_relu(self.conv5(x)), 2)
        x = self.batchnorm5(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = F.leaky_relu(self.linear1(x))
        x = self.batchnorm6(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.batchnorm7(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.batchnorm8(x)
        x = F.dropout(x, p=0.5)
        x = self.linear4(x)
        return x
