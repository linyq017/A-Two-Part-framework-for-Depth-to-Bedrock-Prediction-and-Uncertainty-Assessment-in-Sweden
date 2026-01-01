"""Evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import *

def pinball_loss(y_true, y_pred, quantile):
    """Pinball loss."""
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))


def picp(y_test, y_lower, y_upper):
    """Prediction Interval Coverage Probability."""
    return np.mean((y_test >= y_lower) & (y_test <= y_upper))

def quantile_coverage_probability(y_true, y_pred_quantile, alpha=0.9):
    """
    Calculate the Quantile Coverage Probability (QCP).
    
    QCP measures the fraction of observations in the test set that fall below
    the predicted quantile. It indicates how well-calibrated a quantile prediction is.
    
    Parameters:
    -----------
    y_true : array-like
        Actual observed values from the test set.
    
    y_pred_quantile : array-like
        Predicted quantile values (must be same length as y_true).
    
    alpha : float, optional (default=0.9)
        The target quantile level (between 0 and 1).
        - alpha=0.5 for median predictions
        - alpha=0.9 for 90th percentile predictions
        - etc.
    
    Returns:
    --------
    qcp : float
        The fraction of observations below the predicted quantile (between 0 and 1).
        Ideally, QCP should be close to alpha.
    """
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred_quantile = np.asarray(y_pred_quantile)
    
    # Validate inputs
    if len(y_true) != len(y_pred_quantile):
        raise ValueError("y_true and y_pred_quantile must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    
    # Calculate QCP: fraction of observations below the predicted quantile
    coverage = np.mean(y_true <= y_pred_quantile)
    
    return coverage

def ccc(y_true, y_pred):
    """Concordance Correlation Coefficient."""
    if len(y_true) < 2:
        return np.nan
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(cor):
        return 0.0
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    denominator = var_t + var_p + (mean_t - mean_p)**2
    return (2 * cor * np.sqrt(var_t * var_p)) / denominator if denominator != 0 else 0.0


def interval_score(y_true, y_lower, y_upper, alpha=0.1):
    """Interval score."""
    width = y_upper - y_lower
    penalty_below = (2/alpha) * (y_lower - y_true) * (y_true < y_lower)
    penalty_above = (2/alpha) * (y_true - y_upper) * (y_true > y_upper)
    return np.mean(width + penalty_below + penalty_above)


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_binary_component(y_true, outcrop_proba, threshold=0.5):
    """
    Evaluate the binary classification component separately.
    Returns metrics and predictions.
    """
    y_binary_true = (y_true == 0).astype(int)
    is_outcrop_pred = (outcrop_proba >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_binary_true, is_outcrop_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'threshold': threshold,
        'AUC': roc_auc_score(y_binary_true, outcrop_proba),
        'Brier_Score': brier_score_loss(y_binary_true, outcrop_proba),
        'Log_Loss': log_loss(y_binary_true, outcrop_proba),
        'Accuracy': accuracy_score(y_binary_true, is_outcrop_pred),
        'Precision': precision_score(y_binary_true, is_outcrop_pred, zero_division=0),
        'Recall': recall_score(y_binary_true, is_outcrop_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1_Score': f1_score(y_binary_true, is_outcrop_pred, zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'n_true_outcrops': y_binary_true.sum(),
        'n_pred_outcrops': is_outcrop_pred.sum(),
    }
    
    return metrics, is_outcrop_pred

def evaluate_regression_component(y_true, y_pred_all, quantiles):
    """
    Evaluate the QRF regression component on TRUE DEPTH samples only.
    This is the "ideal" performance if binary classification were perfect.
    """
    # Only evaluate on samples with actual depth
    mask = y_true > 0
    y_true_depth = y_true[mask]
    y_pred_depth = y_pred_all[mask]
    
    if len(y_true_depth) == 0:
        return {'error': 'No depth samples in test set'}
    
    q_map = {q: i for i, q in enumerate(quantiles)}
    y_pred_median = y_pred_depth[:, q_map[0.5]]
    
    metrics = {
        'n_samples': len(y_true_depth),
        'MAE': mean_absolute_error(y_true_depth, y_pred_median),
        'RMSE': np.sqrt(mean_squared_error(y_true_depth, y_pred_median)),
        'R2': r2_score(y_true_depth, y_pred_median),
        'CCC': ccc(y_true_depth, y_pred_median),
        'Bias': np.mean(y_true_depth - y_pred_median),
    }
    
    # Add uncertainty metrics for common intervals
    for i, lower_q in enumerate(quantiles):
        upper_q = quantiles[-(i+1)]
        if lower_q >= upper_q:
            break
        
        y_lower = y_pred_depth[:, i]
        y_upper = y_pred_depth[:, -(i+1)]
        nominal_cov = int(round((upper_q - lower_q) * 100))
        
        metrics[f'PICP_{nominal_cov}'] = picp(y_true_depth, y_lower, y_upper)
        metrics[f'PI_Width_{nominal_cov}'] = np.mean(y_upper - y_lower)
    return metrics

def evaluate_fused_model(y_true, outcrop_proba, y_pred_all, quantiles, threshold=0.5):
    """
    Comprehensive evaluation of the fused two-stage model.
    """
    is_outcrop_pred = (outcrop_proba >= threshold).astype(int)
    y_binary_true = (y_true == 0).astype(int)
    q_map = {q: i for i, q in enumerate(quantiles)}
    y_pred_median = y_pred_all[:, q_map[0.5]]

    # Create the "Final Pipeline Prediction"
    # If binary says Outcrop (1), set depth to 0. Otherwise keep QRF median.
    y_pred_pipeline = y_pred_median.copy()
    y_pred_pipeline[is_outcrop_pred == 1] = 0
    # --- CRITICAL FIX END ---

    # 1. Binary Classification Performance (Unchanged)
    cm = confusion_matrix(y_binary_true, is_outcrop_pred)
    tn, fp, fn, tp = cm.ravel()
    
    binary_metrics = {
        'threshold': threshold,
        'AUC': roc_auc_score(y_binary_true, outcrop_proba),
        'Brier_Score': brier_score_loss(y_binary_true, outcrop_proba),
        'Log_Loss': log_loss(y_binary_true, outcrop_proba),
        'Accuracy': accuracy_score(y_binary_true, is_outcrop_pred),
        'Precision': precision_score(y_binary_true, is_outcrop_pred, zero_division=0),
        'Recall': recall_score(y_binary_true, is_outcrop_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1_Score': f1_score(y_binary_true, is_outcrop_pred, zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }
    
    # 2. Clean Regression Performance (Unchanged - Evaluates QRF on correct soil only)
    # This answers: "When we correctly find soil, how accurate is the depth?"
    mask_correct = (~is_outcrop_pred.astype(bool)) & (y_true > 0)
    
    if mask_correct.sum() > 0:
        y_true_correct = y_true[mask_correct]
        y_pred_correct = y_pred_median[mask_correct]
        y_pred_all_correct = y_pred_all[mask_correct]
        
        regression_clean = {
            'n_samples': mask_correct.sum(),
            'MAE': mean_absolute_error(y_true_correct, y_pred_correct),
            'RMSE': np.sqrt(mean_squared_error(y_true_correct, y_pred_correct)),
            'R2': r2_score(y_true_correct, y_pred_correct),
            'CCC': ccc(y_true_correct, y_pred_correct),
            'Bias': np.mean(y_true_correct - y_pred_correct),
        }
        
        if 0.05 in quantiles and 0.95 in quantiles:
            y_pred_05 = y_pred_all_correct[:, q_map[0.05]]
            y_pred_95 = y_pred_all_correct[:, q_map[0.95]]
            regression_clean['PICP_90'] = picp(y_true_correct, y_pred_05, y_pred_95)
            regression_clean['PI_Width_90'] = np.mean(y_pred_95 - y_pred_05)
    else:
        regression_clean = {'n_samples': 0, 'MAE': np.nan, 'RMSE': np.nan, 
                           'R2': np.nan, 'CCC': np.nan, 'Bias': np.nan}
    
    # 3. Full Pipeline Performance
    # This answers: "What is the error of the final map vs reality?"
    # (Includes penalty for misclassifying soil as rock)
    mask_true_depth = y_true > 0
    
    if mask_true_depth.sum() > 0:
        y_true_depth = y_true[mask_true_depth]
        
        # Use the PIPELINE prediction (which includes zeros), not the raw QRF
        y_pred_depth_pipeline = y_pred_pipeline[mask_true_depth]
        
        # Note: We don't really have "PIs" for the rock predictions (they are point estimates of 0)
        # So for PIs, we usually still look at the raw QRF or we set PI width to 0
        y_pred_all_depth = y_pred_all[mask_true_depth] 
        
        n_false_outcrop = (mask_true_depth & is_outcrop_pred.astype(bool)).sum()
        
        full_pipeline = {
            'n_samples': mask_true_depth.sum(),
            'n_misclassified_as_outcrop': n_false_outcrop,
            'pct_misclassified': 100 * n_false_outcrop / mask_true_depth.sum(),
            
            # These metrics will now likely be WORSE (higher error) because 
            # we are comparing True Depth (e.g. 1.5) vs Predicted (0.0) for misclassifications
            'MAE': mean_absolute_error(y_true_depth, y_pred_depth_pipeline),
            'RMSE': np.sqrt(mean_squared_error(y_true_depth, y_pred_depth_pipeline)),
            'R2': r2_score(y_true_depth, y_pred_depth_pipeline),
            'CCC': ccc(y_true_depth, y_pred_depth_pipeline),
            'Bias': np.mean(y_true_depth - y_pred_depth_pipeline),
        }
        
        # Add PI metrics
        # For PIs, it's debatable. If we predicted Rock, our PI is technically [0, 0].
        # But usually, we track the QRF's uncertainty even if we zeroed it out, 
        # OR we just accept the raw QRF stats here to see "potential" uncertainty.
        # Let's keep the raw QRF stats for PIs to avoid breaking calculations with zeros.
        if 0.05 in quantiles and 0.95 in quantiles:
            y_pred_05_all = y_pred_all_depth[:, q_map[0.05]]
            y_pred_95_all = y_pred_all_depth[:, q_map[0.95]]
            full_pipeline['PICP_90'] = picp(y_true_depth, y_pred_05_all, y_pred_95_all)
            full_pipeline['PI_Width_90'] = np.mean(y_pred_95_all - y_pred_05_all)
    else:
        full_pipeline = {'n_samples': 0, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    
    return {
        'binary': binary_metrics,
        'regression_clean': regression_clean,
        'full_pipeline': full_pipeline
    }

def threshold_sensitivity_analysis(y_true, outcrop_proba, y_pred_all, quantiles, 
                                   thresholds=None):
    """
    Evaluate model performance across different outcrop probability thresholds.
    Shows the trade-off between binary classification and regression performance.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    for thresh in thresholds:
        eval_results = evaluate_fused_model(y_true, outcrop_proba, y_pred_all, 
                                           quantiles, threshold=thresh)
        
        row = {
            'threshold': thresh,
            'binary_accuracy': eval_results['binary']['Accuracy'],
            'binary_recall': eval_results['binary']['Recall'],
            'binary_precision': eval_results['binary']['Precision'],
            'binary_f1': eval_results['binary']['F1_Score'],
            'pct_misclassified': eval_results['full_pipeline']['pct_misclassified'],
        }
        results.append(row)
    
    return pd.DataFrame(results)

# ==============================================================================
# Helper function to add pipeline metrics to existing sensitivity_df
# ==============================================================================

def add_pipeline_metrics_to_sensitivity(sensitivity_df, y_true, outcrop_proba, 
                                       y_pred_depth, quantiles=None):
    """
    Add full pipeline metrics to existing threshold sensitivity DataFrame.
    
    This calculates the TRUE final map error including FP penalties.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        Existing threshold sensitivity results
    y_true : array-like
        True depth values (0 = outcrop, >0 = depth)
    outcrop_proba : array-like
        Predicted outcrop probabilities
    y_pred_depth : array-like
        Predicted depths from QRF (median or specific quantile)
    quantiles : list, optional
        If provided, will also calculate PI metrics for pipeline
    
    Returns:
    --------
    pd.DataFrame with added columns: pipeline_mae, pipeline_rmse, pipeline_r2
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    print("\nAdding full pipeline metrics to sensitivity analysis...")
    
    pipeline_results = []
    
    for threshold in sensitivity_df['threshold']:
        # Create binary predictions
        is_outcrop_pred = (outcrop_proba >= threshold).astype(int)
        
        # Create final pipeline prediction
        y_final = y_pred_depth.copy()
        y_final[is_outcrop_pred == 1] = 0  # Set predicted outcrops to 0
        
        # Evaluate on true depth samples only
        mask_true_depth = y_true > 0
        
        if mask_true_depth.sum() > 0:
            y_true_depth = y_true[mask_true_depth]
            y_final_depth = y_final[mask_true_depth]
            
            pipeline_results.append({
                'threshold': threshold,
                'pipeline_mae': mean_absolute_error(y_true_depth, y_final_depth),
                'pipeline_rmse': np.sqrt(mean_squared_error(y_true_depth, y_final_depth)),
                'pipeline_r2': r2_score(y_true_depth, y_final_depth)
            })
        else:
            pipeline_results.append({
                'threshold': threshold,
                'pipeline_mae': np.nan,
                'pipeline_rmse': np.nan,
                'pipeline_r2': np.nan
            })
    
    pipeline_df = pd.DataFrame(pipeline_results)
    
    # Merge with original
    enhanced_df = sensitivity_df.merge(pipeline_df, on='threshold', how='left')
    
    print(f"✅ Added pipeline metrics")
    print(f"   MAE range: {enhanced_df['pipeline_mae'].min():.3f} - {enhanced_df['pipeline_mae'].max():.3f}m")
    print(f"   Clean vs Pipeline MAE difference: {(enhanced_df['pipeline_mae'] - enhanced_df['regression_mae']).mean():.3f}m avg")
    
    return enhanced_df

def print_evaluation_summary(evaluation):
    """Print evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\n1. BINARY CLASSIFICATION:")
    print("-"*70)
    b = evaluation['binary']
    print(f"  AUC: {b['AUC']:.3f} | Accuracy: {b['Accuracy']:.3f} | F1: {b['F1_Score']:.3f}")
    
    print("\n2. REGRESSION (Clean):")
    print("-"*70)
    r = evaluation['regression_clean']
    print(f"  MAE: {r['MAE']:.2f}m | RMSE: {r['RMSE']:.2f}m | R²: {r['R2']:.3f}")
    
    print("\n3. FULL PIPELINE:")
    print("-"*70)
    p = evaluation['full_pipeline']
    print(f"  MAE: {p['MAE']:.2f}m | RMSE: {p['RMSE']:.2f}m | R²: {p['R2']:.3f}")
    print(f"  Misclassified: {p['n_misclassified_as_outcrop']} ({p['pct_misclassified']:.1f}%)")
    print("="*70)

