"""
=================================================================================
GRANDMASTER ENSEMBLE & POST-PROCESSING STACKER
Task:
1. Optimal Non-negative Weight Blending (Hill-Climbing / Nelder-Mead)
2. Packaging MRP Calibration & Hybrid Rule Fusion
3. Target Price Floor & Nelder-Mead Global Multiplier Optimization
=================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from metric import calculate_smape, optimize_smape_multiplier

def find_optimal_blend_weights(y_true, oof_predictions_list):
    """
    Finds optimal weights w_1, w_2, ..., w_k such that
    sum(w_i * pred_i) minimizes SMAPE.
    """
    n_models = len(oof_predictions_list)
    initial_weights = np.ones(n_models) / n_models
    
    def objective(weights):
        # Normalize weights to sum to 1
        w = np.maximum(0, weights)
        if np.sum(w) == 0:
            return 999.0
        w = w / np.sum(w)
        
        blend = np.zeros_like(y_true)
        for i in range(n_models):
            blend += w[i] * oof_predictions_list[i]
            
        return calculate_smape(y_true, blend)
        
    bounds = [(0, 1) for _ in range(n_models)]
    res = minimize(objective, initial_weights, method='Nelder-Mead')
    
    best_weights = np.maximum(0, res.x)
    best_weights /= np.sum(best_weights)
    
    raw_scores = [calculate_smape(y_true, p) for p in oof_predictions_list]
    best_score = res.fun
    
    print("\n=== Model Blending Report ===")
    for idx, (score, weight) in enumerate(zip(raw_scores, best_weights)):
        print(f"  Model {idx + 1} Individual SMAPE: {score:.3f}% | Blend Weight: {weight:.4f}")
    print(f"[*] Blended Ensemble SMAPE: {best_score:.3f}% (Improvement: {min(raw_scores) - best_score:.3f}%)")
    
    return best_weights, best_score

def sanitize_model_outliers(predictions_list, y_train_ref=None):
    """
    SINGLE-MODEL DEFENSE SHIELD:
    Protects the ensemble against a rogue model that outputs extreme wild predictions.
    1. Bounds predictions within realistic catalog percentiles (e.g. 0.1% to 99.9% * 2.0).
    2. Clips any individual model prediction that diverges > 4x from the cross-model median.
    """
    sanitized = []
    min_bound = 1.0
    max_bound = np.percentile(y_train_ref, 99.9) * 2.0 if y_train_ref is not None else 500000.0
    
    # Calculate cross-model median per sample
    stacked = np.vstack(predictions_list) # shape: (n_models, n_samples)
    model_median = np.median(stacked, axis=0) # shape: (n_samples,)
    
    for i, pred in enumerate(predictions_list):
        p_clean = np.clip(pred, a_min=min_bound, a_max=max_bound)
        # Pull extreme divergent predictions back toward cross-model consensus
        divergence = p_clean / np.maximum(model_median, 1.0)
        outlier_mask_high = divergence > 4.0
        outlier_mask_low = divergence < 0.25
        
        if np.sum(outlier_mask_high) > 0 or np.sum(outlier_mask_low) > 0:
            p_clean[outlier_mask_high] = model_median[outlier_mask_high] * 2.5
            p_clean[outlier_mask_low] = model_median[outlier_mask_low] * 0.4
            
        sanitized.append(p_clean)
        
    return sanitized

def blend_test_predictions(test_predictions_list, weights):
    """Applies optimal weights to test predictions"""
    blended_test = np.zeros_like(test_predictions_list[0])
    for pred, w in zip(test_predictions_list, weights):
        blended_test += w * pred
    return blended_test

def apply_packaging_mrp_override(predicted_prices, detected_mrp_list, discount_ratio=0.82):
    """
    If packaging clearly shows MRP (e.g. Rs 500), retail e-commerce price
    is tightly clustered around (MRP * 0.82). We use soft bayesian fusion.
    """
    final_prices = np.copy(predicted_prices)
    mrp_count = 0
    
    for i in range(len(final_prices)):
        mrp = detected_mrp_list[i]
        if mrp > 10.0:  # Valid detected MRP
            expected_price = mrp * discount_ratio
            # Blend 70% model + 30% expected retail MRP
            final_prices[i] = 0.70 * final_prices[i] + 0.30 * expected_price
            mrp_count += 1
            
    print(f"[*] Applied Packaging MRP Calibration to {mrp_count} samples.")
    return final_prices

def run_grandmaster_ensemble(y_true, oof_list, test_list, detected_mrp_test=None, train_texts=None, test_texts=None):
    """
    End-to-End Grandmaster Pipeline with 35% SMAPE Targeting:
    1. Optimal Convex Weight Blending
    2. Catalog Twin Refinement (FAISS / NearestNeighbors)
    3. Packaging MRP Calibration
    4. Nelder-Mead Post-Processing Multiplier
    """
    # 0. Sanitize rogue model predictions
    oof_list = sanitize_model_outliers(oof_list, y_true)
    test_list = sanitize_model_outliers(test_list, y_true)

    # 1. Optimal Weight Blending
    weights, blended_score = find_optimal_blend_weights(y_true, oof_list)
    blended_test = blend_test_predictions(test_list, weights)
    
    # 2. Catalog Twin Refinement (if text series provided)
    if train_texts is not None and test_texts is not None:
        from faiss_similarity_matcher import CatalogSimilarityMatcher
        matcher = CatalogSimilarityMatcher(similarity_threshold=0.88)
        matcher.fit(train_texts, y_true)
        blended_test = matcher.apply_catalog_neighbor_override(blended_test, test_texts)
        
    # 3. Packaging MRP Calibration (if available)
    if detected_mrp_test is not None:
        blended_test = apply_packaging_mrp_override(blended_test, detected_mrp_test)
        
    # 4. Nelder-Mead Post-Processing Multiplier
    best_alpha, final_oof_score, calibrated_test = optimize_smape_multiplier(
        y_true, 
        blend_test_predictions(oof_list, weights), 
        blended_test
    )
    
    # 5. Floor & Ceiling clipping
    max_ceil = np.percentile(y_true, 99.9) * 2.0
    calibrated_test = np.clip(calibrated_test, a_min=1.0, a_max=max_ceil)
    
    print(f"\n[GRANDMASTER ENSEMBLE COMPLETE] Final Calibrated OOF SMAPE: {final_oof_score:.3f}%")
    return calibrated_test


