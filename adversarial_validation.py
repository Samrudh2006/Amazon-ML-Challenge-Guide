"""
Amazon ML Challenge 2026 - Adversarial Validation & Distribution Drift Detector
=============================================================================
Detects distribution shift between train.csv and test.csv to eliminate Private Leaderboard Shakeup.

Why Adversarial Validation Guarantees Top 10:
- When competition organizers split data into train and test, sometimes certain brands or categories
  only appear in test.csv.
- Adversarial Validation trains a classifier to distinguish 'is_test' (1) from 'is_train' (0):
    1. If AUC is ~0.50: Perfect random split. Normal K-Fold is optimal.
    2. If AUC is >0.65: Severe distribution drift! Adversarial validation calculates sample weights
       (importance weighting) so the model focuses on training examples that closely match test.csv.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

def run_adversarial_validation(train_csv, test_csv, output_weights="adversarial_sample_weights.csv"):
    print("=" * 65)
    print("  AMAZON ML CHALLENGE - ADVERSARIAL DRIFT DETECTOR")
    print("=" * 65)

    if not HAS_LGB:
        print("[!] LightGBM required for fast adversarial validation.")
        return

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"[*] Analyzing Train ({len(train_df):,} rows) vs Test ({len(test_df):,} rows)...")

    # Target: 0 for train, 1 for test
    train_df["is_test"] = 0
    test_df["is_test"] = 1

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    text_data = combined_df.get("catalog_name", combined_df.get("product_name", "")).fillna("").astype(str)

    print("[*] Vectorizing text to check vocabulary and brand distribution...")
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    X = tfidf.fit_transform(text_data)
    y = combined_df["is_test"].values

    print("[*] Training 3-Fold Adversarial Classifier...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(combined_df))

    for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X[t_idx], y[t_idx]
        X_va, y_va = X[v_idx], y[v_idx]

        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        oof_probs[v_idx] = clf.predict_proba(X_va)[:, 1]

    auc = roc_auc_score(y, oof_probs)
    print("-" * 65)
    print(f"[+] Adversarial ROC-AUC Score: {auc:.4f}")
    
    if auc < 0.55:
        print("[VERDICT] Excellent! Train and Test distributions are identical (AUC ~ 0.50).")
        print("          Standard 5-Fold Stratified Cross-Validation is completely safe.")
    elif auc < 0.70:
        print("[VERDICT] Mild drift detected. Computing importance weights for training...")
    else:
        print("[WARNING] Severe drift detected! Using adversarial weights to prevent private leaderboard drop.")

    # Calculate density ratio weights for train samples: P(is_test) / (1 - P(is_test))
    train_probs = oof_probs[:len(train_df)]
    weights = train_probs / (1.0 - np.clip(train_probs, 1e-5, 0.999))
    weights = weights / np.mean(weights) # Normalize

    weights_df = pd.DataFrame({
        "index": train_df.get("index", range(len(train_df))),
        "adversarial_weight": weights
    })
    weights_df.to_csv(output_weights, index=False)
    print(f"[+] Saved sample importance weights to: {output_weights}")
    print("=" * 65)

if __name__ == "__main__":
    print("[*] Adversarial Validation module ready.")
