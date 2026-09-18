"""
SciPy Mathematical SMAPE Ensemble Blender
Amazon ML Challenge 2026

Solves the convex optimization problem:
    min_{w} SMAPE(y_true, sum(w_i * y_pred_i))
    subject to: w_i >= 0, sum(w_i) == 1

Finds the exact mathematical blending weights for LightGBM, CatBoost, XGBoost,
DeBERTa, and Vision models to maximize leaderboard rank.
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates official competition SMAPE metric percentage."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(100.0 * np.abs(y_pred - y_true) / denom))


class ScipySmapeBlender:
    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or []
        self.weights = None
        self.best_smape = None

    def fit(self, oof_matrix: np.ndarray, y_true: np.ndarray, model_names: List[str] = None) -> np.ndarray:
        """
        oof_matrix: shape (N_samples, M_models)
        y_true: shape (N_samples,)
        """
        if model_names:
            self.model_names = model_names

        n_samples, n_models = oof_matrix.shape
        if not self.model_names or len(self.model_names) != n_models:
            self.model_names = [f"Model_{i+1}" for i in range(n_models)]

        # Benchmark: Equal weights
        equal_weights = np.ones(n_models) / n_models
        equal_pred = np.dot(oof_matrix, equal_weights)
        equal_smape = calculate_smape(y_true, equal_pred)

        # Objective function to minimize
        def objective(weights):
            weights = np.asarray(weights)
            # Normalize to ensure sum is 1 during evaluation
            weights = weights / (np.sum(weights) + 1e-12)
            blend_pred = np.dot(oof_matrix, weights)
            return calculate_smape(y_true, blend_pred)

        # Constraints & Bounds
        # Non-negative weights
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        # Sum of weights == 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        # Initial guess: equal weights
        init_guess = np.ones(n_models) / n_models

        # Run SLSQP optimization
        res = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        if not res.success:
            # Fallback to Nelder-Mead on softmax representation
            def objective_unconstrained(raw_w):
                exp_w = np.exp(raw_w - np.max(raw_w))
                w = exp_w / np.sum(exp_w)
                return calculate_smape(y_true, np.dot(oof_matrix, w))

            res_nm = minimize(objective_unconstrained, np.zeros(n_models), method='Nelder-Mead')
            exp_w = np.exp(res_nm.x - np.max(res_nm.x))
            optimal_weights = exp_w / np.sum(exp_w)
        else:
            optimal_weights = res.x / np.sum(res.x)

        self.weights = optimal_weights
        optimal_pred = np.dot(oof_matrix, self.weights)
        self.best_smape = calculate_smape(y_true, optimal_pred)

        print("\n" + "=" * 55)
        print("   SCIPY MATHEMATICAL SMAPE OPTIMIZER RESULTS")
        print("=" * 55)
        print(f"  Equal Weights SMAPE:   {equal_smape:.4f}%")
        print(f"  Optimal Weights SMAPE: {self.best_smape:.4f}%")
        print(f"  Absolute Delta Gain:   -{(equal_smape - self.best_smape):.4f}%\n")
        print("  Learned Convex Weights:")
        for name, w in zip(self.model_names, self.weights):
            print(f"    - {name:<20}: {w*100:6.2f}% (weight={w:.4f})")
        print("=" * 55 + "\n")

        return self.weights

    def predict(self, test_matrix: np.ndarray) -> np.ndarray:
        """Applies learned optimal weights to test predictions."""
        if self.weights is None:
            raise ValueError("Blender has not been fitted yet! Call fit() first.")
        return np.dot(test_matrix, self.weights)

    def save_weights(self, filepath: str = "optimal_blend_weights.json"):
        """Saves learned weights to JSON configuration."""
        data = {
            "model_names": self.model_names,
            "weights": self.weights.tolist(),
            "best_smape": self.best_smape
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SAVE] Optimal weights saved to {filepath}")


if __name__ == "__main__":
    print("[TEST] Running SciPy SMAPE Blender on synthetic multi-model OOF predictions...")
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.uniform(200, 5000, n_samples)

    # 4 distinct models with different error characteristics
    # Model 1: LightGBM (good baseline, slight overprediction)
    lgb_oof = y_true * np.random.normal(1.05, 0.18, n_samples)
    # Model 2: CatBoost (unbiased, balanced)
    cat_oof = y_true * np.random.normal(0.98, 0.16, n_samples)
    # Model 3: XGBoost (sharp, slight underprediction on high tail)
    xgb_oof = y_true * np.random.normal(0.95, 0.17, n_samples)
    # Model 4: DeBERTa (text semantic strength)
    deb_oof = y_true * np.random.normal(1.02, 0.22, n_samples)

    oof_mat = np.column_stack([lgb_oof, cat_oof, xgb_oof, deb_oof])
    names = ["LightGBM", "CatBoost", "XGBoost", "DeBERTa-v3"]

    blender = ScipySmapeBlender()
    learned_weights = blender.fit(oof_mat, y_true, model_names=names)
    blender.save_weights()
