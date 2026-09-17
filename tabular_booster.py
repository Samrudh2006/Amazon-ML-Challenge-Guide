"""
Amazon ML Challenge 2026 - High-Speed Tabular Gradient Boosting Engine
======================================================================
Trains LightGBM / CatBoost models on text TF-IDF, engineered regex metrics, and OCR candidates.

Why Tabular Boosting is Critical for TOP 10:
1. Orthogonal Predictions: Decision trees split on non-linear threshold boundaries completely differently from Neural Nets.
2. Robust to Missing Data: Automatically handles missing OCR or missing image features.
3. Fast 5-Fold CV: Trains 5 folds in minutes, providing rapid feature engineering feedback.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.sparse import hstack

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def engineer_tabular_features(df):
    """
    Extracts high-signal domain features from catalog descriptions and OCR text.
    """
    features = pd.DataFrame(index=df.index)
    
    # Text length signals
    text = df.get("catalog_name", df.get("product_name", "")).fillna("").astype(str)
    ocr = df.get("ocr_text", "").fillna("").astype(str)

    features["text_len"] = text.str.len()
    features["text_word_count"] = text.str.split().apply(len)
    features["ocr_len"] = ocr.str.len()
    features["ocr_word_count"] = ocr.str.split().apply(len)
    
    # Digit and punctuation density
    features["digit_count_text"] = text.str.count(r'\d')
    features["digit_count_ocr"] = ocr.str.count(r'\d')
    features["has_digits"] = (features["digit_count_text"] + features["digit_count_ocr"] > 0).astype(int)

    # Unit indicators
    common_keywords = ["pack", "ml", "gm", "kg", "litre", "volt", "watt", "cm", "inch", "set"]
    for kw in common_keywords:
        features[f"has_{kw}"] = (
            text.str.contains(rf'\b{kw}\b', case=False, regex=True) |
            ocr.str.contains(rf'\b{kw}\b', case=False, regex=True)
        ).astype(int)

    return features

class TabularBooster:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
        self.models = []

    def train_cv(self, df, target_col="unit_class"):
        print("=" * 65)
        print("  AMAZON ML CHALLENGE - TABULAR GRADIENT BOOSTING ENGINE")
        print("=" * 65)
        
        if not HAS_LGB and not HAS_CATBOOST:
            print("[!] Neither LightGBM nor CatBoost is installed. Run 'pip install lightgbm catboost'.")
            return None

        # 1. Feature Engineering
        print("[*] Generating domain-specific tabular features...")
        num_features = engineer_tabular_features(df)
        
        # 2. Text TF-IDF
        text_series = (df.get("catalog_name", "").fillna("") + " " + df.get("ocr_text", "").fillna("")).astype(str)
        print("[*] Computing N-Gram TF-IDF representation (2,500 features)...")
        tfidf_features = self.tfidf.fit_transform(text_series)
        
        # Combine dense tabular + sparse TF-IDF
        from scipy.sparse import csr_matrix
        dense_num_sparse = csr_matrix(num_features.values)
        X_all = hstack([dense_num_sparse, tfidf_features]).tocsr()
        y_all = df[target_col].values

        print(f"[*] Total Features Dimension: {X_all.shape[1]:,}")
        print(f"[*] Starting {self.n_folds}-Fold Stratified Cross-Validation...")

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(df))
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state + fold,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            self.models.append(model)

            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds
            fold_f1 = f1_score(y_val, val_preds, average="macro")
            cv_scores.append(fold_f1)
            print(f"    -> Fold {fold + 1} Macro F1: {fold_f1 * 100:.2f}%")

        mean_cv = np.mean(cv_scores)
        print("-" * 65)
        print(f"[+] Overall 5-Fold Cross-Validation Macro F1: {mean_cv * 100:.2f}%")
        print("=" * 65)
        return oof_preds

if __name__ == "__main__":
    print("[*] Tabular Booster module ready.")
