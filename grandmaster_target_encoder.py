"""
Grandmaster Out-of-Fold (OOF) Target Encoder
Amazon ML Challenge 2026

Calculates Bayesian-smoothed Out-of-Fold target statistics (Mean, Median, Std, Count)
for high-cardinality categorical variables (brand, category, unit, domain).
Completely leak-free using K-Fold cross-validation partitioning.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import KFold


class GrandmasterTargetEncoder:
    """
    Leak-Free Bayesian Smoothed Target Statistics.
    Computes prior-weighted smoothed means:
    Smoothed_Mean = (count * group_mean + smoothing * global_mean) / (count + smoothing)
    """

    def __init__(self, cat_cols: List[str], target_col: str = "price", smoothing: float = 15.0, n_splits: int = 5, random_state: int = 42):
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean = 0.0
        self.global_median = 0.0
        self.global_std = 0.0
        self.fitted_mappings = {}

    def _prepare_cats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "catalog_content" in df.columns:
            if "brand" not in df.columns:
                df["brand"] = df["catalog_content"].str.extract(r"Brand:\s*([^\|\n,]+)", flags=re.IGNORECASE)[0].fillna("Unknown").str.strip()
            if "unit" not in df.columns:
                df["unit"] = df["catalog_content"].str.extract(r"\b(kg|g|gm|ml|l|ltr|pack|pcs)\b", flags=re.IGNORECASE)[0].fillna("unit").str.lower()
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates out-of-fold target statistics for train dataset.
        """
        df = self._prepare_cats(train_df)
        target = np.log1p(df[self.target_col].clip(lower=0.1))
        self.global_mean = float(target.mean())
        self.global_median = float(target.median())
        self.global_std = float(target.std())

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in self.cat_cols:
            if col not in df.columns:
                continue

            oof_mean_col = f"{col}_oof_target_mean"
            oof_std_col = f"{col}_oof_target_std"
            oof_count_col = f"{col}_count"

            df[oof_mean_col] = np.nan
            df[oof_std_col] = np.nan
            df[oof_count_col] = 0

            # K-Fold loop for leak-free training
            for train_idx, val_idx in kf.split(df):
                tr_part = df.iloc[train_idx]
                val_part = df.iloc[val_idx]
                tr_target = target.iloc[train_idx]

                # Group statistics on train fold
                grp = tr_target.groupby(tr_part[col])
                counts = grp.count()
                means = grp.mean()
                stds = grp.std().fillna(0.0)

                # Smoothed mean formula
                smoothed_means = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)

                # Map to validation slice
                df.loc[df.index[val_idx], oof_mean_col] = val_part[col].map(smoothed_means).fillna(self.global_mean)
                df.loc[df.index[val_idx], oof_std_col] = val_part[col].map(stds).fillna(0.0)
                df.loc[df.index[val_idx], oof_count_col] = val_part[col].map(counts).fillna(0)

            # Fit entire train for test transformation
            full_grp = target.groupby(df[col])
            full_counts = full_grp.count()
            full_means = full_grp.mean()
            full_stds = full_grp.std().fillna(0.0)
            full_smoothed = (full_counts * full_means + self.smoothing * self.global_mean) / (full_counts + self.smoothing)

            self.fitted_mappings[col] = {
                "smoothed_mean": full_smoothed,
                "std": full_stds,
                "count": full_counts
            }

        return df

    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test dataframe using the fitted Bayesian smoothed statistics.
        """
        df = self._prepare_cats(test_df)

        for col in self.cat_cols:
            if col not in self.fitted_mappings:
                continue

            mapping = self.fitted_mappings[col]
            oof_mean_col = f"{col}_oof_target_mean"
            oof_std_col = f"{col}_oof_target_std"
            oof_count_col = f"{col}_count"

            df[oof_mean_col] = df[col].map(mapping["smoothed_mean"]).fillna(self.global_mean)
            df[oof_std_col] = df[col].map(mapping["std"]).fillna(0.0)
            df[oof_count_col] = df[col].map(mapping["count"]).fillna(0)

        return df


if __name__ == "__main__":
    print("[TEST] Initializing Grandmaster Target Encoder on mock data...")
    # Load mock train/test
    try:
        train_df = pd.read_csv("mock_train.csv")
        test_df = pd.read_csv("mock_test.csv")

        encoder = GrandmasterTargetEncoder(
            cat_cols=["brand", "category", "unit"],
            target_col="price",
            smoothing=10.0
        )

        train_encoded = encoder.fit_transform(train_df)
        test_encoded = encoder.transform(test_df)

        print("[SUCCESS] Grandmaster Target Encoding successful!")
        print(f"Train columns added: {[c for c in train_encoded.columns if 'target' in c]}")
        print(f"Sample encoded rows:\n{train_encoded[['brand', 'brand_oof_target_mean', 'brand_count']].head(3)}")
    except Exception as e:
        print(f"[ERROR] Mock test failed: {e}")
