import torch


def logcosh_loss(y_hat, y):
    return torch.mean(torch.log(torch.cosh((y_hat - y) + 1e-12)))
