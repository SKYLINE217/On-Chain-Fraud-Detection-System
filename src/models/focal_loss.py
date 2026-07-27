"""
Focal Loss for extreme class imbalance.

Reduces loss weight for easy examples, focuses training on hard examples.
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

gamma=2, alpha=0.25 is standard starting point.

Usage:
    Switch to Focal Loss if after 50 epochs val PR-AUC < 0.6 with class
    weighting. Try gamma=2, alpha=0.25 first.

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    Reduces loss weight for easy examples, focuses training on hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2, alpha=0.25 is standard starting point.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) unnormalized scores
            targets: (N,) class indices 0 or 1 (no -1 here -- mask before calling)
        """
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
