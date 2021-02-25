import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    loss = nn.MSELoss()
    return loss(output, target)


def bce_loss(output, target):
    loss = nn.BCELoss()
    return loss(output, target)
