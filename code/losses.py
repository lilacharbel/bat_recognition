import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        log_prob = F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        # Gather the log_prob based on target labels
        log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = self.alpha.gather(0, targets)

        # Calculate the focal loss
        loss = - alpha * ((1 - prob) ** self.gamma) * log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss