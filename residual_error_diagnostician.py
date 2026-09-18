"""
Residual Error Diagnostician (The Grandmaster Post-Mortem Brain)
Amazon ML Challenge 2026

Analyzes Out-of-Fold (OOF) predictions, extracts the highest SMAPE failure cases,
classifies the root causes (Multi-Pack Blindness, Accessory Confusion, Luxury Mismatch),
and outputs automated tactical recommendations to push SMAPE down into Top 10.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple


def calculate_sample_smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates sample-level SMAPE percentage."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return 100.0 * np.abs(y_pred - y_true) / denominator


class ResidualErrorDiagnostician:
    def __init__(self, top_k: int = 20, smape_threshold: float = 60.0):
        self.top_k = top_k
        self.smape_threshold = smape_threshold

    def diagnose(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, text_col: str = "catalog_content") -> pd.DataFrame:
        """
        Takes raw dataframe, true prices, and predicted prices.
        Returns a diagnostic dataframe with classified error types and remediation advice.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        smapes = calculate_sample_smape(y_true, y_pred)
        ratio = y_pred / np.clip(y_true, 1e-3, None)

        diag_df = df.copy()
        diag_df["y_true"] = y_true
        diag_df["y_pred"] = np.round(y_pred, 2)
        diag_df["smape"] = np.round(smapes, 2)
        diag_df["pred_to_true_ratio"] = np.round(ratio, 2)

        # Classify Root Causes
        error_types = []
        action_advice = []

        for idx, row in diag_df.iterrows():
            txt = str(row.get(text_col, "")).lower()
            yt = row["y_true"]
            yp = row["y_pred"]
            s = row["smape"]
            r = row["pred_to_true_ratio"]

            if s < self.smape_threshold:
                error_types.append("NORMAL_VARIANCE")
                action_advice.append("Prediction within acceptable confidence boundary.")
                continue

            # Case 1: Accessory / Cover mistaken for Main Device
            if r > 3.0 and any(w in txt for w in ["case", "cover", "glass", "protector", "stand", "cable", "strap"]):
                error_types.append("ACCESSORY_OVERPREDICTION")
                action_advice.append("Model confused accessory with main product. Add negative accessory penalty feature.")

            # Case 2: Multi-Pack Blindness
            elif r < 0.35 and any(w in txt for w in ["pack of", "set of", "combo", "pcs", "count", "pieces"]):
                error_types.append("MULTIPACK_UNDERPREDICTION")
                action_advice.append("Model failed to scale price with Item Pack Quantity (IPQ). Ensure IPQ multiplier is active.")

            # Case 3: Luxury / Flagship Brand Underprediction
            elif r < 0.4 and any(w in txt for w in ["apple", "dyson", "sony", "bose", "rolex", "samsung galaxy s", "pro max"]):
                error_types.append("LUXURY_BRAND_UNDERPREDICTION")
                action_advice.append("Model compressed extreme high-end price. Increase brand target encoding weight.")

            # Case 4: Extreme High Price skew (Underpredicted)
            elif yt > 20000 and r < 0.5:
                error_types.append("HIGH_VALUE_COMPRESSION")
                action_advice.append("Tree regressors struggle on extreme tail. Add Log-Huber or Box-Cox power transform.")

            # Case 5: Budget Item Overpredicted
            elif yt < 200 and r > 2.5:
                error_types.append("BUDGET_FLOOR_COLLAPSE")
                action_advice.append("Low value item overpredicted. Add floor clamping or minimum price threshold.")

            else:
                error_types.append("UNSTRUCTURED_RESIDUAL")
                action_advice.append("Inspect multimodal image and DINOv2 visual embedding for visual signals.")

        diag_df["root_cause"] = error_types
        diag_df["remediation"] = action_advice

        # Sort by worst SMAPE
        worst_cases = diag_df.sort_values(by="smape", ascending=False).head(self.top_k)
        return worst_cases

    def generate_executive_summary(self, worst_cases: pd.DataFrame) -> Dict[str, any]:
        """Produces aggregate statistics on model failure modes."""
        counts = worst_cases["root_cause"].value_counts().to_dict()
        avg_smape = float(worst_cases["smape"].mean())
        max_smape = float(worst_cases["smape"].max())

        summary = {
            "worst_sample_count": len(worst_cases),
            "worst_mean_smape": round(avg_smape, 2),
            "worst_peak_smape": round(max_smape, 2),
            "failure_mode_breakdown": counts,
            "primary_weakness": max(counts, key=counts.get) if counts else "NONE"
        }
        return summary


if __name__ == "__main__":
    print("[DIAGNOSTICIAN] Running Residual Error Analysis on simulated predictions...")
    try:
        train_df = pd.read_csv("mock_train.csv")
        y_true = train_df["price"].values
        # Simulate imperfect model predictions with classic failure modes
        np.random.seed(42)
        noise = np.random.normal(1.0, 0.25, len(y_true))
        y_pred = y_true * noise
        # Inject deliberate multi-pack blindness and accessory overprediction
        y_pred[1] = 15000.0  # iPhone 15 Pro Max true 129k -> predicted 15k (Luxury underprediction)
        y_pred[0] = 120.0    # Cadbury pack of 3 true 485 -> predicted 120 (Multipack blindness)

        diag = ResidualErrorDiagnostician(top_k=5, smape_threshold=40.0)
        report = diag.diagnose(train_df, y_true, y_pred)
        summary = diag.generate_executive_summary(report)

        print("\n[EXECUTIVE SUMMARY OF WEAKNESSES]")
        for k, v in summary.items():
            print(f"  - {k}: {v}")

        print("\n[TOP FAILING SAMPLES & REMEDIATIONS]")
        for idx, r in report.iterrows():
            print(f"Sample #{r.get('sample_id', idx)} | True: Rs.{r['y_true']} | Pred: Rs.{r['y_pred']} | SMAPE: {r['smape']}%")
            print(f"  Root Cause:  {r['root_cause']}")
            print(f"  Remediation: {r['remediation']}\n")

    except Exception as e:
        print(f"[ERROR] Diagnostician test failed: {e}")
