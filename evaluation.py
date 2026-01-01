"""Evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import *
from pathlib import Path
# from metrics import quantile_coverage_probability, ccc, picp 
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
        
        # FIXED: Add QCP for specific quantiles (not intervals)
        # For 90% PI, we check QCP at 5th and 95th percentiles
        if lower_q == 0.05 and upper_q == 0.95:
            metrics[f'QCP_0.05'] = quantile_coverage_probability(
                y_true_depth, y_lower, alpha=0.05
            )
            metrics[f'QCP_0.95'] = quantile_coverage_probability(
                y_true_depth, y_upper, alpha=0.95
            )
    
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
    y_pred_pipeline = y_pred_median.copy()
    y_pred_pipeline[is_outcrop_pred == 1] = 0

    # 1. Binary Classification Performance
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
        'n_true_outcrops': y_binary_true.sum(),
        'n_pred_outcrops': is_outcrop_pred.sum(),
    }
    
    # 2. Clean Regression Performance
    # "When we correctly find soil, how accurate is the depth?"
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
        
        # if 0.05 in quantiles and 0.95 in quantiles:
        #     y_pred_05 = y_pred_all_correct[:, q_map[0.05]]
        #     y_pred_95 = y_pred_all_correct[:, q_map[0.95]]
        #     regression_clean['PICP_90'] = picp(y_true_correct, y_pred_05, y_pred_95)
        #     regression_clean['PI_Width_90'] = np.mean(y_pred_95 - y_pred_05)
    else:
        regression_clean = {'n_samples': 0, 'MAE': np.nan, 'RMSE': np.nan, 
                           'R2': np.nan, 'CCC': np.nan, 'Bias': np.nan}
    # --- NEW: PICP and QCP Calculation ---
    for i, lower_q in enumerate(quantiles):
        # Calculate QCP for every single quantile in the model
        q_val = quantiles[i]
        y_q_pred = y_pred_all_correct[:, i]
        regression_clean[f'QCP_{q_val:.3f}'] = quantile_coverage_probability(
            y_true_correct, y_q_pred, alpha=q_val
        )
        
        # Calculate PICP for symmetrical pairs (e.g., 0.05 and 0.95)
        upper_idx = -(i + 1)
        upper_q = quantiles[upper_idx]
        
        if lower_q < upper_q:
            y_lower = y_pred_all_correct[:, i]
            y_upper = y_pred_all_correct[:, upper_idx]
            
            nominal_cov = int(round((upper_q - lower_q) * 100))
            
            # PICP: % of true values between the two bounds
            regression_clean[f'PICP_{nominal_cov}'] = picp(
                y_true_correct, y_lower, y_upper
            )
            # Sharpness: Average width of the interval
            regression_clean[f'PI_Width_{nominal_cov}'] = np.mean(y_upper - y_lower)
    # 3. Full Pipeline Performance
    # "What is the error of the final map vs reality?"
    mask_true_depth = y_true > 0
    
    if mask_true_depth.sum() > 0:
        y_true_depth = y_true[mask_true_depth]
        y_pred_depth_pipeline = y_pred_pipeline[mask_true_depth]
        y_pred_all_depth = y_pred_all[mask_true_depth]
        
        n_false_outcrop = (mask_true_depth & is_outcrop_pred.astype(bool)).sum()
        
        full_pipeline = {
            'n_samples': mask_true_depth.sum(),
            'n_misclassified_as_outcrop': n_false_outcrop,
            'pct_misclassified': 100 * n_false_outcrop / mask_true_depth.sum(),
            'MAE': mean_absolute_error(y_true_depth, y_pred_depth_pipeline),
            'RMSE': np.sqrt(mean_squared_error(y_true_depth, y_pred_depth_pipeline)),
            'R2': r2_score(y_true_depth, y_pred_depth_pipeline),
            'CCC': ccc(y_true_depth, y_pred_depth_pipeline),
            'Bias': np.mean(y_true_depth - y_pred_depth_pipeline),
        }
        
        # Add PI metrics (using raw QRF predictions)
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
    
    UPDATED: Now includes final pipeline MAE/RMSE/R² that accounts for 
    misclassifications (depth samples wrongly predicted as outcrop = 0m)
    
    Parameters:
    -----------
    y_true : array-like
        True depth values (0 = outcrop, >0 = depth)
    outcrop_proba : array-like
        Predicted probabilities for outcrop class
    y_pred_all : array-like, shape (n_samples, n_quantiles)
        Predicted quantiles from QRF
    quantiles : list
        List of quantiles (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
    thresholds : array-like, optional
        Custom threshold values to test (default: 0.1 to 0.9 in steps of 0.1)
    
    Returns:
    --------
    pd.DataFrame with comprehensive metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    # Get median prediction from QRF
    q_map = {q: i for i, q in enumerate(quantiles)}
    y_pred_median = y_pred_all[:, q_map[0.5]]
    
    results = []
    for thresh in thresholds:
        # Binary predictions
        is_outcrop_pred = (outcrop_proba >= thresh).astype(int)
        y_binary_true = (y_true == 0).astype(int)
        
        # Create final pipeline prediction
        # If binary says outcrop, set depth to 0; otherwise use QRF prediction
        y_pred_final = y_pred_median.copy()
        y_pred_final[is_outcrop_pred == 1] = 0
        
        # Evaluate fused model
        eval_results = evaluate_fused_model(
            y_true, outcrop_proba, y_pred_all, quantiles, threshold=thresh
        )
        
        row = {
            'threshold': thresh,
            
            # Binary classification metrics
            'binary_accuracy': eval_results['binary']['Accuracy'],
            'binary_recall': eval_results['binary']['Recall'],
            'binary_precision': eval_results['binary']['Precision'],
            'binary_f1': eval_results['binary']['F1_Score'],
            
            # Regression metrics (clean - only on correctly classified depth samples)
            'regression_clean_mae': eval_results['regression_clean']['MAE'],
            'regression_clean_rmse': eval_results['regression_clean']['RMSE'],
            'regression_clean_r2': eval_results['regression_clean']['R2'],
            
            # Full pipeline metrics (includes penalty for misclassifications)
            'pipeline_mae': eval_results['full_pipeline']['MAE'],
            'pipeline_rmse': eval_results['full_pipeline']['RMSE'],
            'pipeline_r2': eval_results['full_pipeline']['R2'],
            
            # Error analysis
            'pct_misclassified': eval_results['full_pipeline']['pct_misclassified'],
            'n_misclassified': eval_results['full_pipeline']['n_misclassified_as_outcrop'],
        }
        
        # Add specificity (TN / (TN + FP))
        tn = eval_results['binary']['TN']
        fp = eval_results['binary']['FP']
        row['binary_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate combined score (for finding optimal threshold)
        # 60% final MAE + 30% FP minimization + 10% recall maintenance
        row['combined_score'] = (
            row['pipeline_mae'] / min([r['pipeline_mae'] for r in results + [row]]) * 0.6 +
            (row['pct_misclassified'] / 100) * 0.3 +
            (1 - row['binary_recall']) * 0.1
        )
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Recalculate combined_score with proper normalization
    min_mae = df['pipeline_mae'].min()
    df['combined_score'] = (
        df['pipeline_mae'] / min_mae * 0.6 +
        (df['pct_misclassified'] / 100) * 0.3 +
        (1 - df['binary_recall']) * 0.1
    )
    
    return df

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


def export_all_plot_data(y_test, fused_results, quantiles, evaluation, output_dir):
    """
    Export all data used in plots to CSV files for reproducibility.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    plots_data_dir = output_dir / 'plot_data'
    plots_data_dir.mkdir(exist_ok=True)
    
    q_map = {q: i for i, q in enumerate(quantiles)}
    
    # 1. Predictions data
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_median': fused_results['fused_predictions'][:, q_map[0.5]],
        'outcrop_proba': fused_results['outcrop_proba'],
        'is_outcrop_pred': fused_results['is_outcrop'],
        'is_outcrop_true': (y_test == 0)
    })
    
    # Add quantile predictions
    for q in quantiles:
        predictions_df[f'q_{q:.3f}'] = fused_results['fused_predictions'][:, q_map[q]]
    
    predictions_df.to_csv(plots_data_dir / 'predictions.csv', index=False)
    
    # 2. Evaluation metrics
    eval_flat = {}
    for key, metrics in evaluation.items():
        for metric_name, value in metrics.items():
            eval_flat[f'{key}_{metric_name}'] = value
    
    eval_df = pd.DataFrame([eval_flat])
    eval_df.to_csv(plots_data_dir / 'evaluation_metrics.csv', index=False)
    
    print(f"  ✓ Exported plot data to {plots_data_dir}")



# ==============================================================================
# HELPER: Define Depth Classes  
# ==============================================================================
def _get_depth_masks(y_test):
    """
    Internal helper to generate depth class masks.
    Returns a list of (class_name, boolean_mask).
    """
    # Ensure numpy array for boolean indexing
    y = np.array(y_test)
    
    classes = [
        ('0-2m',   (y > 0)  & (y <= 2)),
        ('2-5m',   (y > 2)  & (y <= 5)),
        ('5-10m',  (y > 5)  & (y <= 10)),
        ('10-15m', (y > 10) & (y <= 15)),
        ('15-20m', (y > 15) & (y <= 20)),
        ('20-30m', (y > 20) & (y <= 30)),
        ('>30m',    y > 30),
        ('Overall', np.ones_like(y, dtype=bool)) # Includes Overall automatically
    ]
    return classes
# ============================================================================
# MAIN EVALUATION WORKFLOW
# ============================================================================

def evaluate_by_depth_class(y_true, outcrop_proba, y_pred_all, quantiles, threshold=0.5):
    """
    Stratified evaluation by depth class for the fused model.
    """
    is_outcrop_pred = (outcrop_proba >= threshold).astype(int)
    q_map = {q: i for i, q in enumerate(quantiles)}
    y_pred_median = y_pred_all[:, q_map[0.5]]
    
    results = []
    for name, mask in _get_depth_masks(y_true):
        if mask.sum() == 0:
            continue
        
        # For depth classes (not outcrop), evaluate regression
        if name != 'Overall' or np.any(y_true[mask] > 0):
            depth_mask = mask & (y_true > 0)
            if depth_mask.sum() > 0:
                y_t = y_true[depth_mask]
                y_p = y_pred_median[depth_mask]
                y_p_all = y_pred_all[depth_mask]
                
                row = {
                    'depth_class': name,
                    'n_samples': depth_mask.sum(),
                    'MAE': mean_absolute_error(y_t, y_p),
                    'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
                    'Bias': np.mean(y_t - y_p),
                }
                
                # Add PI metrics if available
                if 0.05 in quantiles and 0.95 in quantiles:
                    y_05 = y_p_all[:, q_map[0.05]]
                    y_95 = y_p_all[:, q_map[0.95]]
                    row['PICP_90'] = picp(y_t, y_05, y_95)
                    row['PI_Width_90'] = np.mean(y_95 - y_05)
                
                if name == 'Overall':
                    row['R2'] = r2_score(y_t, y_p)
                    row['CCC'] = ccc(y_t, y_p)
                
                results.append(row)
    
    return pd.DataFrame(results)

# # # ==============================================================================
# # # 2. CALIBRATION EVALUATION
# # # ==============================================================================
def evaluate_calibration_by_depth(y_test, y_pred_all, quantiles):
    """
    Evaluate uncertainty (PICP, PI Width, QCP) by depth class.
    """
    y_test = np.array(y_test)
    y_pred_all = np.array(y_pred_all)
    
    # Map quantile values to their column index: e.g. {0.05: 0, 0.5: 1, 0.95: 2}
    q_map = {q: i for i, q in enumerate(quantiles)}

    results = []
    
    # Iterate through shared masks
    for name, mask in _get_depth_masks(y_test):
        n_samples = mask.sum()
        
        if n_samples > 0:
            row = {'class': name, 'n': n_samples}
            
            yt_subset = y_test[mask]
            yp_subset = y_pred_all[mask]

            # A. Interval Metrics (PICP, PI Width)
            # Pairs quantiles from outside in: (0.05, 0.95), (0.1, 0.9), etc.
            for i, lower_q in enumerate(quantiles):
                upper_q = quantiles[-(i+1)]
                if lower_q >= upper_q: 
                    break # Stop when we cross the middle
                
                y_lower = yp_subset[:, i]
                y_upper = yp_subset[:, -(i+1)]
                
                nominal_cov = int(round((upper_q - lower_q) * 100))
                
                # Assuming picp() is imported
                row[f'PICP_{nominal_cov}'] = picp(yt_subset, y_lower, y_upper)
                row[f'PI_Width_{nominal_cov}'] = np.mean(y_upper - y_lower)

            # B. Quantile Metrics (QCP)
            for q in quantiles:
                q_idx = q_map[q]
                # Assuming quantile_coverage_probability is imported
                row[f'QCP_{q:.3f}'] = quantile_coverage_probability(
                    yt_subset, yp_subset[:, q_idx], alpha=q
                )
            
            results.append(row)
            
    return pd.DataFrame(results)

def comprehensive_fused_evaluation(y_test, outcrop_proba, y_pred_all, quantiles, 
                                   output_dir, threshold=0.5):
    """
    Complete evaluation workflow for the fused two-stage model.
    
    Parameters:
    -----------
    y_test : array-like
        True depth values (0 = outcrop, >0 = depth in meters)
    outcrop_proba : array-like
        Predicted probabilities from binary model (0-1)
    y_pred_all : array-like, shape (n_samples, n_quantiles)
        Predicted quantiles from QRF model
    quantiles : list
        List of quantiles predicted (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
    output_dir : Path or str
        Directory to save results
    threshold : float
        Outcrop probability threshold (default 0.5)
    
    Returns:
    --------
    dict : All evaluation results
    """
    from visualization.plots import plot_threshold_sensitivity, plot_fused_model_summary, plot_calibration_by_depth
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE FUSED MODEL EVALUATION")
    print("="*80)
    

    # 3. Evaluate Fused Model (three perspectives)
    print(f"\n### 3. Fused Model Evaluation (threshold={threshold}) ###")
    fused_results = evaluate_fused_model(y_test, outcrop_proba, y_pred_all, 
                                        quantiles, threshold)
    
    # Save each perspective
    for key, metrics in fused_results.items():
        df = pd.DataFrame([metrics])
        df.to_csv(output_dir / f'fused_{key}_metrics.csv', index=False)
        print(f"\n{key.upper()}:")
        print(df.T)
    
    # 4. Threshold Sensitivity Analysis
    print("\n### 4. Threshold Sensitivity Analysis ###")
    sensitivity_df = threshold_sensitivity_analysis(y_test, outcrop_proba, 
                                                    y_pred_all, quantiles)
    sensitivity_df.to_csv(output_dir / 'threshold_sensitivity.csv', index=False)
    plot_threshold_sensitivity(sensitivity_df, output_dir / 'threshold_sensitivity.png')
    
    # 5. Stratified Evaluation by Depth Class
    print("\n### 5. Performance by Depth Class ###")
    depth_class_df = evaluate_by_depth_class(y_test, outcrop_proba, y_pred_all, 
                                             quantiles, threshold)
    depth_class_df.to_csv(output_dir / 'performance_by_depth_class.csv', index=False)
    print(depth_class_df.to_string(index=False))
    
    if 'PICP_90' in depth_class_df.columns:
        plot_calibration_by_depth(depth_class_df, 
                                 output_dir / 'calibration_by_depth_class.png')
    
    # 6. Create comprehensive summary plot
    print("\n### 6. Generating Summary Visualizations ###")
    plot_fused_model_summary(fused_results, output_dir / 'fused_model_summary.png')
    
    print("\n" + "="*80)
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print("="*80)
    print("\nGenerated Files:")
    print("  - binary_component_metrics.csv")
    print("  - regression_component_metrics.csv")
    print("  - fused_binary_metrics.csv")
    print("  - fused_regression_clean_metrics.csv")
    print("  - fused_full_pipeline_metrics.csv")
    print("  - threshold_sensitivity.csv")
    print("  - performance_by_depth_class.csv")
    print("  - threshold_sensitivity.png")
    print("  - calibration_by_depth_class.png")
    print("  - fused_model_summary.png")
    print("="*80)
    
    return {
        # 'binary': binary_metrics,
        # 'regression': regression_metrics,
        'fused': fused_results,
        'sensitivity': sensitivity_df,
        'by_depth': depth_class_df,
        #'optimal_threshold': optimal_row['threshold']
    }

