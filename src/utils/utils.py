import torch
import torch.nn as nn
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets, shuffled_targets, lam]

    return data, targets


def mixup_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)
