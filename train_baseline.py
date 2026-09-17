import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from metric import calculate_smape, optimize_smape_multiplier
from features import build_tabular_features

def run_training_pipeline(train_csv, test_csv, output_sub="submission.csv"):
    print(f"=== Amazon ML Challenge Baseline Pipeline ===")
    print(f"Loading data from {train_csv} and {test_csv}...")
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")
    
    # 1. Feature Engineering
    print("\n[Step 1/5] Extracting regex & tabular features (IPQ, Weights, Quantities)...")
    train_feats = build_tabular_features(train_df, text_column='catalog_content')
    test_feats = build_tabular_features(test_df, text_column='catalog_content')
    
    # 2. TF-IDF Text Features
    print("[Step 2/5] Extracting TF-IDF text features from catalog_content...")
    tfidf = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_test_tfidf = tfidf.transform(test_df['catalog_content'].fillna('').astype(str))
    
    # Combine TF-IDF with tabular features
    X_train = hstack([X_train_tfidf, train_feats.values]).tocsr()
    X_test = hstack([X_test_tfidf, test_feats.values]).tocsr()
    
    # Target transformation: log1p
    y_train_raw = train_df['price'].values.astype(float)
    y_train_log = np.log1p(y_train_raw)
    
    # 3. 5-Fold Cross Validation
    print("\n[Step 3/5] Training 5-Fold LightGBM Regressor on log1p(price)...")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_log_preds = np.zeros(len(train_df))
    test_log_preds = np.zeros(len(test_df))
    
    lgb_params = {
        'objective': 'regression_l1',   # MAE on log space maps well to percentage error
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 45,
        'n_estimators': 800,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[trn_idx], y_train_log[trn_idx]
        X_va, y_va = X_train[val_idx], y_train_log[val_idx]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        
        val_pred_log = model.predict(X_va)
        oof_log_preds[val_idx] = val_pred_log
        
        fold_smape = calculate_smape(np.expm1(y_va), np.expm1(val_pred_log))
        print(f"  Fold {fold + 1}/{n_splits} - Local SMAPE: {fold_smape:.3f}%")
        
        test_log_preds += model.predict(X_test) / n_splits
        
    # Convert predictions back from log space
    oof_raw_preds = np.expm1(oof_log_preds)
    test_raw_preds = np.expm1(test_log_preds)
    
    # Floor prices to prevent negative or zero
    oof_raw_preds = np.clip(oof_raw_preds, a_min=1.0, a_max=None)
    test_raw_preds = np.clip(test_raw_preds, a_min=1.0, a_max=None)
    
    raw_oof_smape = calculate_smape(y_train_raw, oof_raw_preds)
    print(f"\n[Step 4/5] Full Out-Of-Fold Raw SMAPE: {raw_oof_smape:.3f}%")
    
    # 4. Post-Processing Multiplier Optimization
    print("Running Nelder-Mead Post-Processing Calibration...")
    best_alpha, opt_smape, final_test_preds = optimize_smape_multiplier(
        y_train_raw, oof_raw_preds, test_raw_preds
    )
    
    # 5. Build Final Submission
    print(f"\n[Step 5/5] Generating submission file -> {output_sub}...")
    sub = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.round(final_test_preds, 2)
    })
    
    # Safety assertions
    assert len(sub) == len(test_df), "Mismatch in submission rows!"
    assert sub['price'].isnull().sum() == 0, "Submission contains NaN values!"
    assert (sub['price'] <= 0).sum() == 0, "Submission contains non-positive prices!"
    
    sub.to_csv(output_sub, index=False)
    print(f"SUCCESS! Submission saved to {output_sub} ({len(sub)} rows).")
    print(sub.head())
    
    return opt_smape

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_training_pipeline(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python train_baseline.py <train_csv> <test_csv>")
