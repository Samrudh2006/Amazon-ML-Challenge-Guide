"""
=================================================================================
BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA TPE ENGINE)
Automates finding the mathematically optimal hyperparameters for LightGBM & CatBoost.
Uses Tree-structured Parzen Estimator (TPE) to discover lowest Out-Of-Fold SMAPE.
=================================================================================
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from metric import calculate_smape

def tune_lightgbm_hyperparameters(X, y_log, y_raw, n_trials=10, n_splits=3):
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        print("[!] Optuna or LightGBM not installed.")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"[*] Commencing Bayesian Optimization Study ({n_trials} Trials)...")

    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
            'n_estimators': 300,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(y_log))

        for trn_idx, val_idx in kf.split(X):
            X_tr, y_tr = X[trn_idx], y_log[trn_idx]
            X_va, y_va = X[val_idx], y_log[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict(X_va)

        oof_raw = np.clip(np.expm1(oof), a_min=1.0, a_max=None)
        score = calculate_smape(y_raw, oof_raw)
        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*60)
    print("   [BAYESIAN TUNING COMPLETE] OPTIMAL HYPERPARAMETERS FOUND")
    print("="*60)
    print(f"  Best SMAPE Score: {study.best_value:.3f}%")
    for k, v in study.best_params.items():
        print(f"    -> {k:<20}: {v}")
    print("="*60 + "\n")
    return study.best_params

if __name__ == "__main__":
    print("=== Bayesian Hyperparameter Tuner Loaded ===")
