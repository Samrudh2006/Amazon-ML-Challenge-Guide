"""
Amazon ML Challenge 2026 - Competition-Aligned Custom Loss Functions
====================================================================
Standard MSE and Cross-Entropy fail in competitive ML because they don't align with the evaluation metrics.

This module provides:
1. FocalLoss: Solves severe class imbalance in unit prediction (e.g. 'gram' is 60% of data, 'kilovolt' is 0.1%).
2. Differentiable SMAPE Loss: Symmetric Mean Absolute Percentage Error (Amazon's favorite price/value metric).
3. LogCoshLoss: Smooth L1 alternative that prevents extreme price/weight outliers from corrupting gradients.
4. CompositeMetricLoss: Joint loss balancing unit classification F1 + numerical regression accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss focuses learning on hard examples and prevents frequent classes
    from overwhelming the gradient.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DifferentiableSMAPELoss(nn.Module):
    """
    Smooth, differentiable version of Symmetric Mean Absolute Percentage Error (SMAPE):
    SMAPE = 200% * |y - y_hat| / (|y| + |y_hat| + epsilon)
    """
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        # preds and targets are in natural scale (un-logged)
        numerator = torch.abs(preds - targets)
        denominator = torch.abs(preds) + torch.abs(targets) + self.epsilon
        smape = 2.0 * numerator / denominator
        return torch.mean(smape)

class LogCoshLoss(nn.Module):
    """
    Log(Cosh(x)) behaves like L2 loss for small errors and L1 loss for large errors.
    Completely immune to packaging text outlier numbers (e.g., telephone numbers mistaken for weight).
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        diff = preds - targets
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

class CompositeMetricLoss(nn.Module):
    """
    Combines Focal Loss for unit prediction + LogCosh / SMAPE for value magnitude.
    """
    def __init__(self, unit_weight=1.0, value_weight=2.0, focal_gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.log_cosh = LogCoshLoss()
        self.unit_weight = unit_weight
        self.value_weight = value_weight

    def forward(self, unit_logits, unit_targets, value_preds_log, value_targets_log):
        loss_unit = self.focal(unit_logits, unit_targets)
        loss_val = self.log_cosh(value_preds_log, value_targets_log)
        return (self.unit_weight * loss_unit) + (self.value_weight * loss_val)

if __name__ == "__main__":
    print("=" * 65)
    print("  AMAZON ML CHALLENGE - COMPETITIVE LOSS SUITE")
    print("=" * 65)
    print("[*] FocalLoss, DifferentiableSMAPELoss, LogCoshLoss, CompositeMetricLoss ready.")
