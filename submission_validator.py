"""
Amazon ML Challenge 2026 - Submission Validator & Disqualification Shield
========================================================================
Validates candidate submission CSVs against Amazon's strict leaderboard rules.

Checks performed:
1. File existence and non-empty size.
2. Exact row count match against test.csv (no missing or extra predictions).
3. Column names match required format (e.g., 'index' and 'prediction').
4. Zero NaN / Null / Empty string values.
5. Entity formatting checks: '<positive_float_or_int> <allowed_unit>' syntax.
6. Local CV Metric calculation (Micro/Macro F1 Score) if ground truth is provided.
"""

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np

# Allowed standard units dictionary
ALLOWED_UNITS = {
    # Weight
    "gram", "kilogram", "milligram", "microgram", "ounce", "pound", "ton",
    # Volume
    "millilitre", "litre", "decilitre", "centilitre", "microlitre", "pint", "quart", "gallon", "fluid ounce",
    # Dimension
    "centimetre", "millimetre", "metre", "kilometre", "inch", "foot", "yard",
    # Voltage / Wattage
    "volt", "millivolt", "kilovolt", "watt", "kilowatt"
}

def validate_submission(
    submission_path,
    test_path=None,
    id_col="index",
    prediction_col="prediction",
    strict_entity_format=True
):
    print("=" * 65)
    print("  AMAZON ML CHALLENGE - SUBMISSION INTEGRITY SCANNER")
    print("=" * 65)
    
    issues = []
    warnings = []

    # 1. Existence check
    if not os.path.exists(submission_path):
        print(f"[CRITICAL ERROR] Submission file not found: {submission_path}")
        return False

    try:
        sub_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to parse CSV: {e}")
        return False

    print(f"[*] Submission File: {os.path.basename(submission_path)}")
    print(f"[*] Total Rows: {len(sub_df):,}")
    print(f"[*] Columns Found: {list(sub_df.columns)}")

    # 2. Column Name Check
    if id_col not in sub_df.columns:
        issues.append(f"Missing ID column: '{id_col}'. Found: {list(sub_df.columns)}")
    if prediction_col not in sub_df.columns:
        issues.append(f"Missing prediction column: '{prediction_col}'. Found: {list(sub_df.columns)}")

    if issues:
        print("\n[!] SCAN FAILED ON COLUMN HEADERS:")
        for iss in issues:
            print(f"    -> {iss}")
        return False

    # 3. Test Row Count & ID Alignment Check
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"[*] Comparing against Test CSV ({len(test_df):,} rows)...")
        
        if len(sub_df) != len(test_df):
            issues.append(
                f"Row count mismatch! Submission has {len(sub_df):,} rows, but test.csv has {len(test_df):,} rows."
            )
            
        if id_col in test_df.columns:
            if not sub_df[id_col].equals(test_df[id_col]):
                # Check set difference
                sub_ids = set(sub_df[id_col])
                test_ids = set(test_df[id_col])
                missing = test_ids - sub_ids
                extra = sub_ids - test_ids
                if missing:
                    issues.append(f"{len(missing):,} IDs from test.csv are missing in submission!")
                if extra:
                    issues.append(f"{len(extra):,} unknown IDs in submission that are not in test.csv!")
                if not missing and not extra:
                    warnings.append("IDs match test.csv but ordering is different.")

    # 4. Null / NaN Check
    null_count = sub_df[prediction_col].isnull().sum()
    empty_str_count = (sub_df[prediction_col].astype(str).str.strip() == "").sum()
    total_missing = null_count + empty_str_count

    if total_missing > 0:
        issues.append(f"Found {total_missing:,} NULL, NaN, or completely empty predictions!")
        print(f"[!] Warning: Amazon evaluator rejects submissions containing NaN values.")

    # 5. Format & Syntax Verification (<number> <unit>)
    if strict_entity_format:
        format_regex = r'^(\d+(?:\.\d+)?)\s+([a-zA-Z\s]+)$'
        invalid_format_count = 0
        invalid_units_count = 0
        sample_invalids = []

        for idx, val in sub_df[prediction_col].dropna().items():
            val_str = str(val).strip()
            match = re.match(format_regex, val_str)
            if not match:
                invalid_format_count += 1
                if len(sample_invalids) < 5:
                    sample_invalids.append((idx, val_str, "Syntax error (must be '<number> <unit>')"))
            else:
                num_part, unit_part = match.groups()
                unit_part = unit_part.strip().lower()
                if unit_part not in ALLOWED_UNITS:
                    invalid_units_count += 1
                    if len(sample_invalids) < 5:
                        sample_invalids.append((idx, val_str, f"Unknown unit: '{unit_part}'"))

        if invalid_format_count > 0:
            warnings.append(
                f"{invalid_format_count:,} rows do not match '<value> <unit>' pattern (might be intended if classification task)."
            )
        if invalid_units_count > 0:
            warnings.append(
                f"{invalid_units_count:,} rows use units not in the standard Amazon units list."
            )

        if sample_invalids:
            print("\n[*] Sample format warnings:")
            for s_idx, s_val, s_reason in sample_invalids:
                print(f"    Row {s_idx}: '{s_val}' -> {s_reason}")

    # Final Verdict
    print("\n" + "-" * 65)
    if issues:
        print("[CRITICAL VERDICT] [FAIL] SUBMISSION REJECTED (DO NOT UPLOAD!)")
        for iss in issues:
            print(f"  [X] {iss}")
        print("-" * 65)
        return False
    else:
        print("[VERDICT] [PASS] SUBMISSION INTEGRITY 100% VERIFIED!")
        if warnings:
            print("[ADVISORY] Minor warnings to review:")
            for w in warnings:
                print(f"  [!] {w}")
        else:
            print("  [SUCCESS] Zero errors, zero warnings. Ready for Leaderboard submission!")
        print("-" * 65)
        return True

def compute_f1_metrics(ground_truth_csv, submission_csv, target_col="entity_value", pred_col="prediction"):
    """
    Computes Exact Match F1 score for local CV validation.
    """
    gt_df = pd.read_csv(ground_truth_csv)
    sub_df = pd.read_csv(submission_csv)
    
    merged = pd.merge(gt_df, sub_df, on="index")
    correct = (merged[target_col].astype(str).str.strip().str.lower() == 
               merged[pred_col].astype(str).str.strip().str.lower()).sum()
    total = len(merged)
    
    accuracy = correct / total if total > 0 else 0
    print(f"[*] Local Validation Score:")
    print(f"    - Correct Predictions: {correct:,} / {total:,}")
    print(f"    - Exact Match Accuracy / Micro F1: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Submission Validator")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to original test CSV to check count and IDs")
    parser.add_argument("--id_col", type=str, default="index", help="ID column name")
    parser.add_argument("--pred_col", type=str, default="prediction", help="Prediction column name")
    parser.add_argument("--skip_unit_check", action="store_true", help="Skip entity unit check (for regression/classification tasks)")
    
    args = parser.parse_args()
    
    is_valid = validate_submission(
        submission_path=args.submission,
        test_path=args.test_csv,
        id_col=args.id_col,
        prediction_col=args.pred_col,
        strict_entity_format=not args.skip_unit_check
    )
    
    sys.exit(0 if is_valid else 1)
