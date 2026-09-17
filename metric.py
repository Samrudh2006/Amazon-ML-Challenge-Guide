import numpy as np
from scipy.optimize import minimize
# Optional torch import for PyTorch environments
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def calculate_smape(y_true, y_pred):
    """
    Official Symmetric Mean Absolute Percentage Error (SMAPE)
    Formula: SMAPE = (100% / n) * sum( |actual - predicted| / ((|actual| + |predicted|) / 2) )
    Range: 0% to 200% (Lower is better)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    
    # Handle edge case where both actual and predicted are zero
    mask = denominator != 0
    smape_array = np.zeros_like(diff)
    smape_array[mask] = diff[mask] / denominator[mask]
    
    return float(np.mean(smape_array) * 100.0)

def optimize_smape_multiplier(y_true, oof_preds, test_preds=None):
    """
    Post-Processing Secret Weapon:
    Finds scalar multiplier alpha that minimizes SMAPE on Out-Of-Fold (OOF) predictions.
    Typically alpha is in [0.90, 0.98] due to SMAPE's denominator asymmetry.
    """
    def objective(alpha):
        scaled = oof_preds * alpha[0]
        return calculate_smape(y_true, scaled)
    
    initial_alpha = [1.0]
    res = minimize(objective, initial_alpha, method='Nelder-Mead')
    best_alpha = float(res.x[0])
    raw_score = calculate_smape(y_true, oof_preds)
    optimized_score = float(res.fun)
    gain = raw_score - optimized_score
    
    print(f"[SMAPE Optimizer] Raw OOF: {raw_score:.3f}% | Optimized: {optimized_score:.3f}% (Gain: -{gain:.3f}%)")
    print(f"[SMAPE Optimizer] Best Scaling Factor alpha: {best_alpha:.4f}")
    
    if test_preds is not None:
        scaled_test = test_preds * best_alpha
        return best_alpha, optimized_score, scaled_test
    return best_alpha, optimized_score

if HAS_TORCH:
    class PyTorchSMAPELoss(nn.Module):
        """
        Differentiable SMAPE Loss for PyTorch Neural Networks.
        Used for fine-tuning DeBERTa, Qwen2.5, or Multimodal MLPs.
        """
        def __init__(self, eps=1e-8):
            super().__init__()
            self.eps = eps

        def forward(self, y_pred, y_true):
            y_pred = torch.clamp(y_pred, min=self.eps)
            y_true = torch.clamp(y_true, min=self.eps)
            denominator = (torch.abs(y_true) + torch.abs(y_pred) + self.eps) / 2.0
            diff = torch.abs(y_pred - y_true)
            return torch.mean(diff / denominator) * 100.0
else:
    class PyTorchSMAPELoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use PyTorchSMAPELoss.")

