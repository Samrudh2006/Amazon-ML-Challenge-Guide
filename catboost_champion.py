"""
=================================================================================
CATBOOST CHAMPION MODEL (Model Diversity for Grandmaster Ensemble)
Task:
1. Native handling of categorical clusters & brand signals
2. Trains on log1p(price) using Symmetric Trees (fast & resistant to overfitting)
3. Generates Out-of-Fold (OOF) and Test predictions for ensemble stacking
=================================================================================
"""

import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from metric import calculate_smape, optimize_smape_multiplier
from features import build_tabular_features
from brand_category_clusterer import build_cluster_and_brand_features

def run_catboost_training(train_csv, test_csv, out_oof="catboost_oof.npy", out_test="catboost_test.npy"):
    print("=== Amazon ML Challenge - CatBoost Champion Pipeline ===")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # 1. Feature Engineering
    print("[Step 1/4] Extracting tabular & regex features...")
    X_train_num = build_tabular_features(train_df)
    X_test_num = build_tabular_features(test_df)
    
    # 2. Semantic Clustering & Target Encoding
    print("[Step 2/4] Extracting semantic clusters and brand encodings...")
    X_train_clust, X_test_clust = build_cluster_and_brand_features(train_df, test_df)
    
    # Combine tabular with cluster encodings
    X_train_dense = pd.concat([X_train_num, X_train_clust], axis=1)
    X_test_dense = pd.concat([X_test_num, X_test_clust], axis=1)
    
    # 3. TF-IDF Reduced Features
    print("[Step 3/4] Extracting text TF-IDF...")
    tfidf = TfidfVectorizer(max_features=1500, stop_words='english', sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_test_tfidf = tfidf.transform(test_df['catalog_content'].fillna('').astype(str))
    
    X_train = hstack([X_train_tfidf, X_train_dense.values]).tocsr()
    X_test = hstack([X_test_tfidf, X_test_dense.values]).tocsr()
    
    y_train_raw = train_df['price'].values.astype(float)
    y_train_log = np.log1p(y_train_raw)
    
    # 4. 5-Fold Cross Validation
    print("\n[Step 4/4] Training 5-Fold CatBoost Regressor on log1p(price)...")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_log_preds = np.zeros(len(train_df))
    test_log_preds = np.zeros(len(test_df))
    
    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'iterations': 1000,
        'learning_rate': 0.06,
        'depth': 6,
        'random_seed': 42,
        'verbose': 200
    }
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"--- Training Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X_train[trn_idx], y_train_log[trn_idx]
        X_va, y_va = X_train[val_idx], y_train_log[val_idx]
        
        model = CatBoostRegressor(**cat_params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=40,
            verbose=False
        )
        
        val_pred_log = model.predict(X_va)
        oof_log_preds[val_idx] = val_pred_log
        
        fold_smape = calculate_smape(np.expm1(y_va), np.expm1(val_pred_log))
        print(f"  Fold {fold + 1} Local SMAPE: {fold_smape:.3f}%")
        
        test_log_preds += model.predict(X_test) / n_splits
        
    oof_preds_raw = np.clip(np.expm1(oof_log_preds), a_min=1.0, a_max=None)
    test_preds_raw = np.clip(np.expm1(test_log_preds), a_min=1.0, a_max=None)
    
    raw_oof_smape = calculate_smape(y_train_raw, oof_preds_raw)
    print(f"\n[CatBoost Complete] Raw OOF SMAPE: {raw_oof_smape:.3f}%")
    
    # Save predictions for ensemble
    np.save(out_oof, oof_preds_raw)
    np.save(out_test, test_preds_raw)
    print(f"Saved predictions to {out_oof} and {out_test}.")
    
    return oof_preds_raw, test_preds_raw

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_catboost_training(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python catboost_champion.py <train_csv> <test_csv>")
