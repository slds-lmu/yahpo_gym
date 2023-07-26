import torch
import torch.nn as nn


class MultiMaeLoss(nn.Module):
    """Multi-target MAE loss"""

    def __init__(self):
        super(MultiMaeLoss, self).__init__()

    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.mean(torch.abs(output - target), axis=0))
        if torch.isnan(loss):
            loss = torch.tensor(9999999999)  # if loss is nan, set to high value
        return loss


class MultiMseLoss(nn.Module):
    """Multi-target MSE loss"""

    def __init__(self):
        super(MultiMseLoss, self).__init__()

    @staticmethod
    def forward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.mean(torch.pow(output - target, 2), axis=0))
        if torch.isnan(loss):
            loss = torch.tensor(9999999999)  # if loss is nan, set to high value
        return loss
