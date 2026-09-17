"""
Amazon ML Challenge 2026 - Auto-Adaptive Competition Engine (The 100% Automator)
================================================================================
Eliminates manual code changes on competition day.

When Amazon releases the dataset on Sept 25th, you simply place 'train.csv' and 'test.csv' here.
This script automatically:
1. Detects ID column (index, id, product_id, sample_id, etc.)
2. Detects Image URL column (image_link, url, image_url, img, etc.)
3. Detects Text columns (title, catalog_name, description, item_name, etc.)
4. Detects Task Type & Target Column:
   - 'ENTITY_EXTRACTION' (e.g. '500 gram', '100 ml')
   - 'PRICE_REGRESSION' (e.g. float numbers 499.99, 1200.0)
   - 'PRODUCT_CLASSIFICATION' (e.g. category labels)
5. Selects Optimal Loss Function:
   - Entity -> CompositeMetricLoss (Focal + LogCosh)
   - Regression -> DifferentiableSMAPELoss
   - Classification -> FocalLoss
6. Writes a global 'competition_config.json' that powers the entire pipeline!
"""

import os
import sys
import re
import json
import argparse
import pandas as pd
import numpy as np

def detect_column(df, candidates, default=None):
    """Finds best matching column name case-insensitively."""
    col_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        cand_lower = cand.lower().strip()
        if cand_lower in col_map:
            return col_map[cand_lower]
        # Partial match
        for col_l, col_orig in col_map.items():
            if cand_lower in col_l:
                return col_orig
    return default

def detect_task_type(series):
    """
    Analyzes sample target values to infer task type.
    """
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return "UNKNOWN"

    sample = non_null.head(100).tolist()
    
    # Check for Entity pattern: '<number> <unit>'
    entity_regex = r'^\d+(?:\.\d+)?\s+[a-zA-Z\s]+$'
    entity_matches = sum(1 for val in sample if re.match(entity_regex, val))
    if entity_matches / len(sample) >= 0.5:
        return "ENTITY_EXTRACTION"

    # Check for numeric regression
    try:
        float_sample = pd.to_numeric(non_null.head(100), errors='coerce')
        if float_sample.notnull().sum() / len(sample) >= 0.8:
            unique_ratio = len(float_sample.unique()) / len(float_sample)
            if unique_ratio > 0.3:
                return "REGRESSION"
    except Exception:
        pass

    # Check for classification (few unique string/int categories)
    n_unique = series.nunique()
    if n_unique < 200:
        return "CLASSIFICATION"

    return "TEXT_OR_ENTITY"

def auto_configure_competition(train_csv="train.csv", test_csv="test.csv", config_path="competition_config.json"):
    print("=" * 70)
    print("  AMAZON ML CHALLENGE - ZERO-TOUCH AUTO-ADAPTIVE ENGINE")
    print("=" * 70)

    if not os.path.exists(train_csv):
        print(f"[!] Error: '{train_csv}' not found. Please ensure train.csv is in the project root.")
        return None

    train_df = pd.read_csv(train_csv, nrows=1000)
    test_df = pd.read_csv(test_csv, nrows=1000) if os.path.exists(test_csv) else None

    print(f"[*] Analyzing Train columns: {list(train_df.columns)}")
    if test_df is not None:
        print(f"[*] Analyzing Test columns: {list(test_df.columns)}")

    # 1. Detect ID Column
    id_candidates = ["index", "id", "sample_id", "product_id", "item_id", "uid"]
    id_col = detect_column(train_df, id_candidates, default=train_df.columns[0])

    # 2. Detect Image URL Column
    image_candidates = ["image_link", "image_url", "url", "image", "img_url", "link"]
    image_col = detect_column(train_df, image_candidates, default=None)

    # 3. Detect Target Column (in train but not in test, or typical names)
    target_col = None
    if test_df is not None:
        diff_cols = [c for c in train_df.columns if c not in test_df.columns]
        if len(diff_cols) == 1:
            target_col = diff_cols[0]
            
    if not target_col:
        target_candidates = ["entity_value", "price", "target", "label", "value", "category", "prediction"]
        target_col = detect_column(train_df, target_candidates, default=None)

    # 4. Detect Text Columns
    text_cols = []
    text_candidates = ["catalog_name", "product_name", "title", "description", "item_name", "text", "bullet_point"]
    for cand in text_candidates:
        matched = detect_column(train_df, [cand])
        if matched and matched not in text_cols and matched != target_col and matched != image_col and matched != id_col:
            text_cols.append(matched)

    # 5. Determine Task Type & Optimal Loss Function
    task_type = "ENTITY_EXTRACTION"
    if target_col and target_col in train_df.columns:
        task_type = detect_task_type(train_df[target_col])

    loss_map = {
        "ENTITY_EXTRACTION": "CompositeMetricLoss(FocalLoss + LogCoshLoss)",
        "REGRESSION": "DifferentiableSMAPELoss",
        "CLASSIFICATION": "FocalLoss"
    }
    recommended_loss = loss_map.get(task_type, "CompositeMetricLoss")

    config = {
        "id_column": id_col,
        "image_column": image_col,
        "target_column": target_col,
        "text_columns": text_cols,
        "task_type": task_type,
        "recommended_loss": recommended_loss,
        "train_csv_path": train_csv,
        "test_csv_path": test_csv if test_df is not None else None,
        "auto_configured": True
    }

    # Save to JSON
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n" + "-" * 70)
    print("  [100% FULLY CONFIGURED] AUTO-ADAPTATION COMPLETE!")
    print("-" * 70)
    print(f"  [+] Identified ID Column    : '{id_col}'")
    print(f"  [+] Identified Image Column : '{image_col}'")
    print(f"  [+] Identified Target Column: '{target_col}'")
    print(f"  [+] Identified Text Columns : {text_cols}")
    print(f"  [+] Detected Task Type      : {task_type}")
    print(f"  [+] Auto-Selected SOTA Loss : {recommended_loss}")
    print(f"  [+] Configuration Saved To  : {os.path.abspath(config_path)}")
    print("-" * 70)
    print("  [SUCCESS] Zero manual coding needed! The entire pipeline will now adapt automatically.")
    print("=" * 70)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Adaptive Competition Configurator")
    parser.add_argument("--train", type=str, default="train.csv")
    parser.add_argument("--test", type=str, default="test.csv")
    parser.add_argument("--out", type=str, default="competition_config.json")
    
    args = parser.parse_args()
    auto_configure_competition(train_csv=args.train, test_csv=args.test, config_path=args.out)
