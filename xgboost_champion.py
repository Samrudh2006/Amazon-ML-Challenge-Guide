"""
=================================================================================
TITAN 3: DMLC XGBOOST CHAMPION PIPELINE (AMAZON ML CHALLENGE 2026)
Completes the Holy Trinity of GBDT:
  1. Microsoft LightGBM (Leaf-wise trees)
  2. Yandex CatBoost (Symmetric trees)
  3. DMLC XGBoost (Depth-wise exact histogram trees)
Trained strictly on log1p(price) with L1 objective to minimize SMAPE.
=================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from metric import calculate_smape, optimize_smape_multiplier
from features import build_tabular_features
from brand_category_clusterer import build_cluster_and_brand_features
from memory_optimizer import reduce_mem_usage

def run_xgboost_training(train_csv="mock_train.csv", test_csv="mock_test.csv"):
    import xgboost as xgb
    print("\n" + "="*70)
    print("   >>> TITAN 3: DMLC XGBOOST 5-FOLD REGRESSION PIPELINE <<<")
    print("="*70 + "\n")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"[*] Train Samples: {len(train_df)} | Test Samples: {len(test_df)}")

    # 1. Memory Optimization & Tabular Extraction
    print("\n[Step 1/4] Extracting domain tabular features & optimizing RAM...")
    train_tab = build_tabular_features(train_df)
    test_tab = build_tabular_features(test_df)
    train_tab = reduce_mem_usage(train_tab, verbose=False)
    test_tab = reduce_mem_usage(test_tab, verbose=False)

    # 2. Semantic Cluster Features
    print("[Step 2/4] Mining semantic clusters and out-of-fold target encodings...")
    train_clust, test_clust = build_cluster_and_brand_features(train_df, test_df)

    # 3. TF-IDF Text Vectors
    print("[Step 3/4] Extracting TF-IDF text features...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_test_tfidf = tfidf.transform(test_df['catalog_content'].fillna('').astype(str))

    # Combine All Feature Streams
    X_train = hstack([X_train_tfidf, train_tab.values, train_clust.values]).tocsr()
    X_test = hstack([X_test_tfidf, test_tab.values, test_clust.values]).tocsr()

    y_train_raw = train_df['price'].values.astype(float)
    y_train_log = np.log1p(y_train_raw)

    # 4. 5-Fold Stratified Cross-Validation
    print("\n[Step 4/4] Training 5-Fold XGBoost on log1p(price)...")
    n_splits = min(5, len(train_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_log = np.zeros(len(train_df))
    test_preds_log = np.zeros(len(test_df))

    xgb_params = {
        'objective': 'reg:absoluteerror',  # L1 loss maps to percentage error
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 600,
        'random_state': 42,
        'n_jobs': -1
    }

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[trn_idx], y_train_log[trn_idx]
        X_va, y_va = X_train[val_idx], y_train_log[val_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        val_pred = model.predict(X_va)
        oof_preds_log[val_idx] = val_pred
        fold_smape = calculate_smape(np.expm1(y_va), np.expm1(val_pred))
        print(f"  Fold {fold+1}/{n_splits} Local SMAPE: {fold_smape:.3f}%")

        test_preds_log += model.predict(X_test) / n_splits

    oof_raw = np.expm1(oof_preds_log)
    test_raw = np.expm1(test_preds_log)

    total_smape = calculate_smape(y_train_raw, oof_raw)
    print(f"\n[XGBoost Complete] Raw OOF SMAPE: {total_smape:.3f}%")

    np.save("xgboost_oof.npy", oof_raw)
    np.save("xgboost_test.npy", test_raw)
    print("[+] Saved predictions to 'xgboost_oof.npy' and 'xgboost_test.npy'.")
    return oof_raw, test_raw

if __name__ == "__main__":
    train_f = sys.argv[1] if len(sys.argv) > 1 else "mock_train.csv"
    test_f = sys.argv[2] if len(sys.argv) > 2 else "mock_test.csv"
    run_xgboost_training(train_f, test_f)
