"""
=================================================================================
SUBMISSION VERIFIER & ZERO-DISQUALIFICATION AUDITOR
Checks:
1. Exact row count match with test.csv
2. Row-by-row sample_id order alignment (prevents scrambled submission disaster)
3. Zero NaN, null, or inf values
4. Strictly positive price values (> 0)
5. Distribution Drift Check: Compares predicted prices vs train set percentiles
=================================================================================
"""

import sys
import pandas as pd
import numpy as np

def audit_submission(submission_path, test_csv_path, train_csv_path=None):
    print(f"[*] Starting rigorous audit for: {submission_path}")
    print(f"[*] Comparing against official test set: {test_csv_path}")
    
    sub = pd.read_csv(submission_path)
    test = pd.read_csv(test_csv_path)
    
    errors = []
    warnings = []
    
    # Check 1: Required columns
    required_cols = ['sample_id', 'price']
    if list(sub.columns) != required_cols:
        errors.append(f"Column mismatch! Expected {required_cols}, got {list(sub.columns)}")
        
    # Check 2: Row Count
    if len(sub) != len(test):
        errors.append(f"Row count mismatch! Test set has {len(test)} rows, but submission has {len(sub)} rows.")
        
    # Check 3: Row-by-Row ID Alignment (CRITICAL!)
    if not np.array_equal(sub['sample_id'].values, test['sample_id'].values):
        errors.append("CRITICAL ERROR: sample_id order does not match test.csv! Your IDs may be scrambled!")
        
    # Check 4: Nulls or NaNs
    nan_count = sub['price'].isnull().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN/Null values in price column!")
        
    inf_count = np.isinf(sub['price']).sum()
    if inf_count > 0:
        errors.append(f"Found {inf_count} Infinite values in price column!")
        
    # Check 5: Non-positive prices
    invalid_prices = (sub['price'] <= 0).sum()
    if invalid_prices > 0:
        errors.append(f"Found {invalid_prices} zero or negative prices! Prices must be strictly > 0.")
        
    # Check 6: Distribution Drift Analysis
    print("\n--- Predicted Price Distribution Stats ---")
    preds = sub['price'].values
    print(f"  Min Price:    Rs. {np.min(preds):.2f}")
    print(f"  10th %ile:    Rs. {np.percentile(preds, 10):.2f}")
    print(f"  Median (p50): Rs. {np.percentile(preds, 50):.2f}")
    print(f"  Mean Price:   Rs. {np.mean(preds):.2f}")
    print(f"  90th %ile:    Rs. {np.percentile(preds, 90):.2f}")
    print(f"  Max Price:    Rs. {np.max(preds):.2f}")
    
    if train_csv_path:
        train = pd.read_csv(train_csv_path)
        if 'price' in train.columns:
            train_prices = train['price'].values
            print("\n--- Training Price Distribution Comparison ---")
            print(f"  Train Median: Rs. {np.percentile(train_prices, 50):.2f} | Pred Median: Rs. {np.percentile(preds, 50):.2f}")
            print(f"  Train Mean:   Rs. {np.mean(train_prices):.2f} | Pred Mean:   Rs. {np.mean(preds):.2f}")

            
            ratio = np.mean(preds) / (np.mean(train_prices) + 1e-5)
            if ratio < 0.60 or ratio > 1.40:
                warnings.append(f"Potential Distribution Drift: Predicted mean is {ratio:.2f}x of training mean!")
                
    # Final Verdict
    print("\n" + "="*50)
    if errors:
        print("[FAILED] AUDIT FAILED! DO NOT SUBMIT THIS FILE!")
        for err in errors:
            print(f"   [!] {err}")
        return False
    else:
        if warnings:
            print("[WARNING] AUDIT PASSED WITH WARNINGS:")
            for w in warnings:
                print(f"   [*] {w}")
        else:
            print("[SUCCESS] 100% AUDIT PASSED! READY FOR LEADERBOARD SUBMISSION!")
            print("   - No NaNs")
            print("   - Correct IDs & Row Order")
            print("   - Realistic Price Range")
        print("="*50)
        return True


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        train_path = sys.argv[3] if len(sys.argv) > 3 else None
        audit_submission(sys.argv[1], sys.argv[2], train_path)
    else:
        print("Usage: python submission_verifier.py <submission.csv> <test.csv> [train.csv]")
