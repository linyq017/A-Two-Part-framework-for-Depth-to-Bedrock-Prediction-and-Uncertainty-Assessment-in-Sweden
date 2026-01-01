"""
Plotting functions for two-part model visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc
)
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec
from metrics import threshold_sensitivity_analysis, evaluate_fused_model, evaluate_binary_component, evaluate_regression_component
from evaluation import evaluate_calibration_by_depth

def plot_twopart_evaluation(y_true, fused_results, quantiles, 
                            evaluate_calibration_by_depth_func, # Renamed to avoid shadowing
                            evaluation, save_path=None, title_prefix=""):
    """
    Create a 4x4 comprehensive evaluation figure.
    """
    
    # -------------------------------------------------------------------------
    # 1. SETUP & DATA EXTRACTION
    # -------------------------------------------------------------------------
    y_true = np.array(y_true)
    y_pred_all = fused_results['fused_predictions']
    outcrop_proba = fused_results['outcrop_proba']
    is_outcrop_pred = fused_results['is_outcrop']
    
    # Identify Quantile Indices
    q_map = {q: i for i, q in enumerate(quantiles)}
    y_pred_median = y_pred_all[:, q_map[0.5]]
    
    # Define Masks
    mask_soil_true = y_true > 0                # Actually Soil
    mask_soil_pred = ~is_outcrop_pred          # Predicted Soil
    mask_clean = mask_soil_true & mask_soil_pred # True Soil AND Predicted Soil
    
    # Initialize Figure
    fig = plt.figure(figsize=(20, 24)) # Taller figure for 4 rows
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Styles
    text_box_style = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
 
    # -------------------------------------------------------------------------
    # ROW 1: BINARY CLASSIFICATION
    # -------------------------------------------------------------------------
    binary_eval = evaluation['binary']
    
    # [0,0] Confusion Matrix
    ax_cm = fig.add_subplot(gs[0, 0])
    
    # *** FIX: Reconstruct Confusion Matrix from components ***
    tn, fp, fn, tp = binary_eval['TN'], binary_eval['FP'], binary_eval['FN'], binary_eval['TP']
    cm = np.array([[tn, fp], [fn, tp]])
    
    ConfusionMatrixDisplay(cm, display_labels=['Soil', 'Outcrop']).plot(
        ax=ax_cm, cmap='Blues', values_format='d', colorbar=False
    )
    ax_cm.set_title(f'{title_prefix}Confusion Matrix')

    # [0,1] ROC Curve
    ax_roc = fig.add_subplot(gs[0, 1])
    y_binary_true = (y_true == 0).astype(int)
    fpr, tpr, _ = roc_curve(y_binary_true, outcrop_proba)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {binary_eval['AUC']:.3f}")
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title('ROC Curve')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # [0,2] Probability Distribution
    ax_prob = fig.add_subplot(gs[0, 2])
    # Handle edge case where one class might be empty in a fold
    if (y_true > 0).sum() > 0:
        ax_prob.hist(outcrop_proba[y_true > 0], bins=30, alpha=0.5, density=True, label='True Soil', color='green')
    if (y_true == 0).sum() > 0:
        ax_prob.hist(outcrop_proba[y_true == 0], bins=30, alpha=0.5, density=True, label='True Outcrop', color='brown')
    
    ax_prob.set_title('Outcrop Probability')
    ax_prob.set_xlabel('Predicted Probability')
    ax_prob.legend()

    # [0,3] Binary evaluation Text
    ax_bmet = fig.add_subplot(gs[0, 3])
    ax_bmet.axis('off')
    bin_text = (
        f"BINARY evaluation\n{'-'*20}\n"
        f"Accuracy:  {binary_eval['Accuracy']:.3f}\n"
        f"Precision: {binary_eval['Precision']:.3f}\n"
        f"Recall:    {binary_eval['Recall']:.3f}\n"
        f"F1-Score:  {binary_eval['F1_Score']:.3f}\n\n"
        f"Counts:\nTrue Soil: {mask_soil_true.sum():,}\nTrue Rock: {(~mask_soil_true).sum():,}"
    )
    ax_bmet.text(0.1, 0.5, bin_text, fontsize=10, family='monospace', va='center', bbox=text_box_style)

    # -------------------------------------------------------------------------
    # ROW 2: REGRESSION (CLEAN / CORRECT ONLY)
    # -------------------------------------------------------------------------
    reg_eval = evaluation['regression_clean']
    
    if mask_clean.sum() > 0:
        y_c_true = y_true[mask_clean]
        y_c_pred = y_pred_median[mask_clean]
        res_clean = y_c_true - y_c_pred
        
        # [1,0] Obs vs Pred (Hexbin)
        ax_obs = fig.add_subplot(gs[1, 0])
        hb = ax_obs.hexbin(y_c_true, y_c_pred, gridsize=30, cmap='twilight_shifted', mincnt=1, bins='log')
        max_v = max(y_c_true.max(), y_c_pred.max())
        ax_obs.plot([0, max_v], [0, max_v], 'k--', lw=1.5)
        ax_obs.set_title('Predicted vs. Observed (CORRECT ONLY)')
        ax_obs.set_xlabel('Observed (m)')
        ax_obs.set_ylabel('Predicted (m)')
        plt.colorbar(hb, ax=ax_obs, label='log(Count)') 
        ax_obs.text(0.05, 0.95, 
                   f"RMSE: {reg_eval['RMSE']:.2f}m\n$R^2$: {reg_eval['R2']:.2f}\n$CCC$: {reg_eval['CCC']:.2f}", 
                   transform=ax_obs.transAxes, va='top', fontsize=9, bbox=text_box_style)
      
        # [1,1] Residuals Hist
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_hist.hist(res_clean, bins=50, color='steelblue', edgecolor='k', alpha=0.7)
        ax_hist.axvline(0, color='r', ls='--',label='Zero')
        ax_hist.set_title(f'Residuals (Bias: {reg_eval["Bias"]:.2f}m)')
        ax_hist.legend()
        
        # [1,2] Residuals vs Pred
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(y_c_pred, res_clean, alpha=0.3, s=10, c='steelblue', edgecolors='none')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Perfect predictions', alpha=0.7)

        # Trend line
        if len(y_c_pred) > 1:
            z = np.polyfit(y_c_pred, res_clean, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(y_c_pred.min(), y_c_pred.max(), 100)
            ax.plot(x_trend, p(x_trend), 'orange', linewidth=2, label=f'Trend (slope={z[0]:.3f})')

        # Confidence bands (±2 std)
        std_residual = np.std(res_clean)
        ax.axhspan(-2*std_residual, 2*std_residual, alpha=0.1, color='gray', label='±2σ')

        ax.set_xlabel('Predicted Depth (m)', fontsize=10)
        ax.set_ylabel('Residual (Observed - Predicted) (m)', fontsize=10)
        ax.set_title('Residual Plot\n(Check for patterns)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Add diagnostic text
        correlation = np.corrcoef(y_c_pred, res_clean)[0, 1] if len(y_c_pred) > 1 else 0
        mean_res = np.mean(res_clean)
        text = f'Mean: {mean_res:.2f}m\nCorr: {correlation:.3f}\nStd: {std_residual:.2f}m'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # [1,3] Regression evaluation
        ax_rmet = fig.add_subplot(gs[1, 3])
        ax_rmet.axis('off')
        reg_text = (
            f"REGRESSION (Clean)\n{'-'*20}\n"
            f"MAE:  {reg_eval['MAE']:.3f} m\n"
            f"RMSE: {reg_eval['RMSE']:.3f} m\n"
            f"R2:   {reg_eval['R2']:.3f}\n"
            f"Bias: {reg_eval['Bias']:.3f} m\n\n"
            f"n_samples: {mask_clean.sum():,}"
        )
        ax_rmet.text(0.1, 0.5, reg_text, fontsize=10, family='monospace', va='center', bbox=text_box_style)
    else:
        for i in range(4): fig.add_subplot(gs[1, i]).text(0.5,0.5,"No Clean Samples", ha='center')

    # -------------------------------------------------------------------------
    # ROW 3: CALIBRATION (ALL TRUE SOIL, y > 0)
    # -------------------------------------------------------------------------
    if mask_soil_true.sum() > 0:
        # Calculate calibration on the fly
        calib_df = evaluate_calibration_by_depth_func(
            y_true[mask_soil_true], 
            y_pred_all[mask_soil_true], 
            quantiles
        )
        overall_calib = calib_df[calib_df['class'] == 'Overall'].iloc[0]
        depth_calib = calib_df[calib_df['class'] != 'Overall']

        # [2,0] PICP Reliability Plot (Scatter)
        ax_picp = fig.add_subplot(gs[2, 0])
        nom_cov, obs_picp = [], []
        for i, low_q in enumerate(quantiles):
            up_q = quantiles[-(i+1)]
            if low_q >= up_q: break
            nom = int(round((up_q - low_q) * 100))
            if f'PICP_{nom}' in overall_calib:
                nom_cov.append(nom)
                obs_picp.append(overall_calib[f'PICP_{nom}'] * 100)
        
        ax_picp.plot(nom_cov, obs_picp, 'o-', color='blue', label='Observed')
        ax_picp.plot([0,100], [0,100], 'k--', alpha=0.5, label='Ideal')
        ax_picp.set_title('PICP Reliability')
        ax_picp.set_xlabel('Nominal Coverage (%)')
        ax_picp.set_ylabel('Observed PICP (%)')
        ax_picp.grid(True, alpha=0.3)
 
        for i, (nom, picp_val) in enumerate(zip(nom_cov, obs_picp)):
            ax_picp.annotate(f'{picp_val:.1f}', xy=(nom, picp_val), xytext=(-10, 5),
                    textcoords='offset points', fontsize=8, color='blue', alpha=0.8)

        # Highlight 90%
        idx_90 = np.argmin(np.abs(np.array(nom_cov) - 90))
        if len(nom_cov) > 0:
            ax_picp.plot(nom_cov[idx_90], obs_picp[idx_90], 'ro', markersize=10, 
                    markeredgewidth=2, markerfacecolor='none', label=f'90% PI')
        ax_picp.legend()
        
        # [2,1] QCP Reliability (Scatter)
        ax_qcp = fig.add_subplot(gs[2, 1])
        q_levs, q_obs = [], []
        for q in quantiles:
            if f'QCP_{q:.3f}' in overall_calib:
                q_levs.append(q)
                q_obs.append(overall_calib[f'QCP_{q:.3f}'])
        
        ax_qcp.plot(q_levs, q_obs, 'o-', color='purple')
        ax_qcp.plot([0,1], [0,1], 'k--', alpha=0.5)
        
        for q, qcp in zip(q_levs, q_obs):
            if q in [0.05, 0.50, 0.95]: 
                ax_qcp.annotate(f'{qcp*100:.1f}%', xy=(q, qcp), xytext=(-13, 8),
                        textcoords='offset points', fontsize=8, color='blue')
        ax_qcp.set_title('QCP Reliability')
        ax_qcp.set_xlabel('Quantile Level')
        ax_qcp.set_ylabel('Observed Frequency')
        ax_qcp.grid(True, alpha=0.3)
        
        # [2,2] PICP 90% by Depth Class (Bar)
        ax_pdep = fig.add_subplot(gs[2, 2])
        if 'PICP_90' in depth_calib.columns:
            cols = ['green' if abs(x-0.9)<0.05 else 'orange' if abs(x-0.9)<0.1 else 'darksalmon' 
                    for x in depth_calib['PICP_90']]
            ax_pdep.bar(depth_calib['class'], depth_calib['PICP_90']*100, color=cols, alpha=0.7, edgecolor='k')
            ax_pdep.axhline(90, color='r', ls='--', label='Target 90%')
            ax_pdep.set_title('90% Interval Coverage by Depth')
            ax_pdep.tick_params(axis='x', rotation=45)
            ax_pdep.legend()
        
        # [2,3] QCP by Depth (Lines for select quantiles)
        ax_qdep = fig.add_subplot(gs[2, 3])
        target_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        key_qs = [q for q in target_quantiles if q in quantiles]
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(key_qs)))

        for q, color in zip(key_qs, colors):
            col_name = f'QCP_{q:.3f}'
            if col_name in depth_calib.columns:
                ax_qdep.plot(depth_calib['class'], depth_calib[col_name], marker='o', linestyle='-',
                    color=color, markeredgecolor='black', alpha=0.8, label=f'{q:.2f}')

        ax_qdep.set_yticks([0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
        ax_qdep.set_ylim(-0.05, 1.05)
        ax_qdep.grid(which='major', color='grey', linestyle='--', alpha=0.5)
        for q in key_qs:
            ax_qdep.axhline(y=q, color='gray', linestyle=':', alpha=0.3, zorder=0)

        ax_qdep.set_title('QCP Stability by Depth', fontsize=10)
        ax_qdep.set_xlabel('Depth Class', fontsize=10)
        ax_qdep.legend(title="Quantile", fontsize='small', edgecolor='white')
        ax_qdep.tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
    
def plot_threshold_sensitivity(sensitivity_df, save_path, optimal_threshold=None):
    """
    Plot how performance metrics change with different outcrop probability thresholds.
    Includes both regression-only and full pipeline metrics.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        DataFrame with threshold sensitivity results. Expected columns:
        - threshold
        - binary_accuracy, binary_recall, binary_precision, binary_f1
        - regression_mae, regression_rmse, regression_r2 (clean regression only)
        - pct_misclassified
        Optional (if available):
        - pipeline_mae, pipeline_rmse, pipeline_r2 (full pipeline with FP penalty)
    save_path : str or Path
        Path to save the plot
    optimal_threshold : float, optional
        If provided, highlight this threshold as optimal
    """
    
    # Check if we have full pipeline metrics
    has_pipeline = 'pipeline_mae' in sensitivity_df.columns
    
    if has_pipeline:
        # Create 3x2 grid for all metrics
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    else:
        # Use original 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes = axes.flatten()
    
    # Find optimal threshold if not provided
    if optimal_threshold is None and 'combined_score' in sensitivity_df.columns:
        optimal_threshold = sensitivity_df.loc[sensitivity_df['combined_score'].idxmin(), 'threshold']
    
    # =========================================================================
    # Plot 1: Binary Classification Metrics
    # =========================================================================
    ax = axes[0]
    ax.plot(sensitivity_df['threshold'], sensitivity_df['binary_accuracy'], 
            'o-', label='Accuracy', linewidth=2, markersize=4)
    ax.plot(sensitivity_df['threshold'], sensitivity_df['binary_recall'], 
            's-', label='Recall', linewidth=2, markersize=4)
    ax.plot(sensitivity_df['threshold'], sensitivity_df['binary_precision'], 
            '^-', label='Precision', linewidth=2, markersize=4)
    ax.plot(sensitivity_df['threshold'], sensitivity_df['binary_f1'], 
            'd-', label='F1-Score', linewidth=2, markersize=4)
    ax.set_xlabel('Outcrop Probability Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Binary Classification Metrics', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5, label='Default (0.5)')
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='green', linestyle='-', alpha=0.7, linewidth=2.5, 
                   label=f'Optimal ({optimal_threshold:.2f})')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 2: MAE Comparison (Clean vs Pipeline if available)
    # =========================================================================
    ax = axes[1]
    
    if has_pipeline:
        # Plot both clean regression and full pipeline
        ax.plot(sensitivity_df['threshold'], sensitivity_df['regression_clean_mae'], 
                'o-', color='steelblue', linewidth=2, markersize=5, 
                label='Clean Regression (correct classifications only)')
        ax.plot(sensitivity_df['threshold'], sensitivity_df['pipeline_mae'], 
                's-', color='orangered', linewidth=2.5, markersize=5, 
                label='Full Pipeline (includes FP penalty)')
        ax.set_ylabel('MAE (meters)', fontsize=11)
        ax.set_title('MAE: Clean Regression vs Full Pipeline', fontsize=12, fontweight='bold')
        
        if optimal_threshold is not None:
            optimal_pipeline_mae = sensitivity_df.loc[
                sensitivity_df['threshold'] == optimal_threshold, 'pipeline_mae'
            ].values[0]
            ax.plot(optimal_threshold, optimal_pipeline_mae, 'g*', markersize=20, 
                   markeredgecolor='darkgreen', markeredgewidth=2, 
                   label=f'Optimal Pipeline MAE: {optimal_pipeline_mae:.2f}m')
    else:
        # Only clean regression available
        ax.plot(sensitivity_df['threshold'], sensitivity_df['regression_mae'], 
                'o-', color='forestgreen', linewidth=2, markersize=6,
                label='Regression MAE')
        ax.set_ylabel('MAE (meters)', fontsize=11)
        ax.set_title('Regression MAE vs Threshold', fontsize=12, fontweight='bold')
        
        if optimal_threshold is not None:
            optimal_mae = sensitivity_df.loc[
                sensitivity_df['threshold'] == optimal_threshold, 'regression_mae'
            ].values[0]
            ax.plot(optimal_threshold, optimal_mae, 'g*', markersize=20, 
                   markeredgecolor='darkgreen', markeredgewidth=2, 
                   label=f'Optimal: {optimal_mae:.2f}m')
    
    ax.set_xlabel('Outcrop Probability Threshold', fontsize=11)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='green', linestyle='-', alpha=0.7, linewidth=2.5)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 3: Misclassification Rate (False Positive Rate)
    # =========================================================================
    ax = axes[2]
    ax.plot(sensitivity_df['threshold'], sensitivity_df['pct_misclassified'], 
            'o-', color='orangered', linewidth=2, markersize=6)
    ax.set_xlabel('Outcrop Probability Threshold', fontsize=11)
    ax.set_ylabel('% Depth Misclassified as Outcrop', fontsize=11)
    ax.set_title('False Positive Rate (FP)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='green', linestyle='-', alpha=0.7, linewidth=2.5)
        optimal_misc = sensitivity_df.loc[
            sensitivity_df['threshold'] == optimal_threshold, 'pct_misclassified'
        ].values[0]
        ax.plot(optimal_threshold, optimal_misc, 'g*', markersize=20, 
               markeredgecolor='darkgreen', markeredgewidth=2, 
               label=f'Optimal FP: {optimal_misc:.1f}%')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 4: R²
    # =========================================================================
    ax = axes[3]
    
    if has_pipeline:
        ax.plot(sensitivity_df['threshold'], sensitivity_df['regression_clean_r2'], 
                'o-', color='steelblue', linewidth=2, markersize=5, 
                label='Clean Regression R²')
        ax.plot(sensitivity_df['threshold'], sensitivity_df['pipeline_r2'], 
                's-', color='purple', linewidth=2.5, markersize=5, 
                label='Full Pipeline R²')
        ax.set_title('R²: Clean vs Pipeline', fontsize=12, fontweight='bold')
    else:
        ax.plot(sensitivity_df['threshold'], sensitivity_df['regression_r2'], 
                'o-', color='purple', linewidth=2, markersize=6)
        ax.set_title('Regression R²', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Outcrop Probability Threshold', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='green', linestyle='-', alpha=0.7, linewidth=2.5)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    
    # =========================================================================
    # Additional plots if we have full pipeline metrics
    # =========================================================================
    if has_pipeline:
        # Plot 5: RMSE Comparison
        ax = axes[4]
        ax.plot(sensitivity_df['threshold'], sensitivity_df['regression_clean_rmse'], 
                'o-', color='steelblue', linewidth=2, markersize=5, 
                label='Clean Regression RMSE')
        ax.plot(sensitivity_df['threshold'], sensitivity_df['pipeline_rmse'], 
                's-', color='orangered', linewidth=2.5, markersize=5, 
                label='Full Pipeline RMSE')
        ax.set_xlabel('Outcrop Probability Threshold', fontsize=11)
        ax.set_ylabel('RMSE (meters)', fontsize=11)
        ax.set_title('RMSE: Clean vs Pipeline', fontsize=12, fontweight='bold')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
        if optimal_threshold is not None:
            ax.axvline(x=optimal_threshold, color='green', linestyle='-', alpha=0.7, linewidth=2.5)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        
        # Plot 6: Impact of Threshold on Error Components
        ax = axes[5]
        
        # Calculate error increase from FP
        mae_increase = sensitivity_df['pipeline_mae'] - sensitivity_df['regression_clean_mae']
        
        ax.plot(sensitivity_df['threshold'], mae_increase, 
                'o-', color='darkred', linewidth=2.5, markersize=6)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.set_xlabel('Outcrop Probability Threshold', fontsize=11)
        ax.set_ylabel('MAE Increase from FP (meters)', fontsize=11)
        ax.set_title('Penalty from False Positives', fontsize=12, fontweight='bold')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
        if optimal_threshold is not None:
            ax.axvline(x=optimal_threshold, color='green', linestyle='-', alpha=0.7, linewidth=2.5)
            optimal_increase = mae_increase[sensitivity_df['threshold'] == optimal_threshold].values[0]
            ax.plot(optimal_threshold, optimal_increase, 'g*', markersize=20, 
                   markeredgecolor='darkgreen', markeredgewidth=2, 
                   label=f'Optimal penalty: {optimal_increase:.2f}m')
            ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.fill_between(sensitivity_df['threshold'], 0, mae_increase, 
                        alpha=0.2, color='red', label='FP Error Penalty')
    
    # Overall title
    if optimal_threshold is not None:
        title = f'Threshold Sensitivity Analysis | Optimal: {optimal_threshold:.2f}'
        if has_pipeline:
            opt_row = sensitivity_df[sensitivity_df['threshold'] == optimal_threshold].iloc[0]
            title += f' (Pipeline MAE: {opt_row["pipeline_mae"]:.2f}m, FP: {opt_row["pct_misclassified"]:.1f}%)'
    else:
        title = 'Threshold Sensitivity Analysis for Fused Model'
    
    plt.suptitle(title, fontsize=14, y=0.998, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved enhanced threshold sensitivity plot to {save_path}")


def plot_calibration_by_depth(calibration_df, save_path):
    """
    Plot prediction interval coverage by depth class.
    """
    depth_classes = calibration_df['depth_class'].values
    picp_90 = calibration_df['PICP_90'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(depth_classes, picp_90 * 100, color='steelblue', alpha=0.7)
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Nominal 90% Coverage')
    ax.set_xlabel('Depth Class', fontsize=12)
    ax.set_ylabel('Prediction Interval Coverage Probability (%)', fontsize=12)
    ax.set_title('90% Prediction Interval Coverage by Depth Class', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration by depth plot to {save_path}")

def plot_fused_model_summary(eval_results, save_path):
    """
    Create a comprehensive summary plot of the fused model evaluation.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # Extract metrics
    binary = eval_results['binary']
    reg_clean = eval_results['regression_clean']
    full_pipe = eval_results['full_pipeline']
    
    # Plot 1: Binary Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([[binary['TN'], binary['FP']], 
                   [binary['FN'], binary['TP']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Non Outcrop', 'Outcrop'],
                yticklabels=['Non Outcrop', 'Outcrop'])
    ax1.set_title('Binary Classification\nConfusion Matrix', fontsize=10)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot 2: Binary Metrics Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    binary_metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    binary_values = [binary[m] for m in binary_metrics]
    bars = ax2.bar(range(len(binary_metrics)), binary_values, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(binary_metrics)))
    ax2.set_xticklabels([m.replace('_', ' ') for m in binary_metrics], rotation=0, ha='right')
    ax2.set_ylim([0, 1])
    ax2.set_title('Binary Classification Metrics', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Probabilistic Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    prob_metrics = ['AUC', 'Brier_Score', 'Log_Loss']
    prob_values = [binary[m] for m in prob_metrics]
    ax3.bar(range(len(prob_metrics)), prob_values, color='coral', alpha=0.7)
    ax3.set_xticks(range(len(prob_metrics)))
    ax3.set_xticklabels(prob_metrics, rotation=0, ha='right')
    ax3.set_title('Probabilistic Binary Metrics', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Regression Clean Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    reg_metrics = ['MAE', 'RMSE', 'Bias']
    reg_values = [reg_clean.get(m, np.nan) for m in reg_metrics]
    bars = ax4.bar(range(len(reg_metrics)), reg_values, color='forestgreen', alpha=0.7)
    ax4.set_xticks(range(len(reg_metrics)))
    ax4.set_xticklabels(reg_metrics, rotation=0)
    ax4.set_title(f'Regression (Clean)\nn={reg_clean.get("n_samples", 0)}', fontsize=10)
    ax4.set_ylabel('Error (meters)')
    ax4.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Regression Clean R² and CCC
    ax5 = fig.add_subplot(gs[1, 1])
    quality_metrics = ['R2', 'CCC']
    quality_values = [reg_clean.get(m, np.nan) for m in quality_metrics]
    bars = ax5.bar(range(len(quality_metrics)), quality_values, color='forestgreen', alpha=0.7)
    ax5.set_xticks(range(len(quality_metrics)))
    ax5.set_xticklabels(['R²', 'CCC'], rotation=0)
    ax5.set_ylim([0, 1])
    ax5.set_title('Regression Quality (Clean)', fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Uncertainty (Clean)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'PICP_90' in reg_clean:
        picp_val = reg_clean['PICP_90'] * 100
        width_val = reg_clean['PI_Width_90']
        ax6.bar([0, 1], [picp_val, width_val], color=['steelblue', 'coral'], alpha=0.7)
        ax6.set_xticks([0, 1])
        ax6.set_xticklabels(['PICP 90%', 'PI Width'], rotation=0)
        ax6.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax6.set_title('Uncertainty (Clean)', fontsize=10)
        ax6.text(0, picp_val, f'{picp_val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax6.text(1, width_val, f'{width_val:.1f}m', ha='center', va='bottom', fontsize=9)
    
    # Plot 7: Full Pipeline Metrics
    ax7 = fig.add_subplot(gs[2, 0])
    full_metrics = ['MAE', 'RMSE', 'Bias']
    full_values = [full_pipe.get(m, np.nan) for m in full_metrics]
    bars = ax7.bar(range(len(full_metrics)), full_values, color='purple', alpha=0.7)
    ax7.set_xticks(range(len(full_metrics)))
    ax7.set_xticklabels(full_metrics, rotation=0)
    ax7.set_title(f'Full Pipeline\nn={full_pipe.get("n_samples", 0)}', fontsize=10)
    ax7.set_ylabel('Error (meters)')
    ax7.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 8: Full Pipeline R² and CCC
    ax8 = fig.add_subplot(gs[2, 1])
    full_quality = ['R2', 'CCC']
    full_quality_values = [full_pipe.get(m, np.nan) for m in full_quality]
    bars = ax8.bar(range(len(full_quality)), full_quality_values, color='purple', alpha=0.7)
    ax8.set_xticks(range(len(full_quality)))
    ax8.set_xticklabels(['R²', 'CCC'], rotation=0)
    ax8.set_ylim([0, 1])
    ax8.set_title('Full Pipeline Quality', fontsize=10)
    ax8.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 9: Misclassification Info
    ax9 = fig.add_subplot(gs[2, 2])
    n_misc = full_pipe.get('n_misclassified_as_outcrop', 0)
    pct_misc = full_pipe.get('pct_misclassified', 0)
    n_total = full_pipe.get('n_samples', 0)
    
    sizes = [n_misc, n_total - n_misc]
    colors = ['#ff6b6b', '#51cf66']
    labels = [f'Misclassified\n({n_misc})', f'Correct\n({n_total - n_misc})']
    
    wedges, texts, autotexts = ax9.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10})
    ax9.set_title(f'Depth Sample Classification\n({pct_misc:.1f}% misclassified)', fontsize=10)
    
    plt.suptitle('Fused Two-Stage Model: Comprehensive Evaluation Summary', 
                 fontsize=12, y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved fused model summary plot to {save_path}")

def plot_pipeline_evaluation_enhanced(y_true, fused_results, quantiles, evaluation, 
                                      save_path=None, title_prefix=""):
    """
    Enhanced pipeline evaluation plots with annotations and TXT/CSV outputs,
    using a 2x3 grid layout (2 rows, 3 columns). The summary table is removed 
    from the plot and saved to a separate .txt file.
    """
    # Extract components
    y_pred_all = fused_results['fused_predictions']
    is_outcrop_pred = fused_results['is_outcrop']
    
    q_map = {q: i for i, q in enumerate(quantiles)}
    y_pred_median = y_pred_all[:, q_map[0.5]]
    
    # Masks
    mask_soil_true = y_true > 0
    mask_soil_pred = ~is_outcrop_pred
    
    # Get evaluation results
    pipeline_eval = evaluation['full_pipeline']
    reg_eval = evaluation['regression_clean']
    
    # Create figure with 2 rows, 3 columns (6 plots)
    fig = plt.figure(figsize=(18, 12)) 
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3) 
    
    # --- Data preparation common to multiple plots ---
    if mask_soil_true.sum() > 0:
        y_soil_true = y_true[mask_soil_true]
        y_soil_pred = y_pred_median[mask_soil_true]
        is_misclassified = ~mask_soil_pred[mask_soil_true]#mask shows true soil samples predicted as soil (True), negating it results in true soil samples that were NOT predicted as soil.
        
        # Data for Error Analysis by Depth
        depth_bins = [0, 2, 5, 10, 15, 20, 30, np.inf]
        labels = ['0-2m', '2-5m', '5-10m', '10-15m', '15-20m', '20-30m', '>30m']
        y_cuts = pd.cut(y_soil_true, bins=depth_bins, labels=labels)
        
        maes, bias, mis_rates, n_samples = [], [], [], []
        
        for lbl in labels:
            m = y_cuts == lbl
            if m.sum() > 0:
                pipeline_preds = y_soil_pred.copy()
                pipeline_preds[is_misclassified] = 0
                
                abs_err = np.abs(y_soil_true[m] - pipeline_preds[m])
                maes.append(np.mean(abs_err))
                bias.append(np.mean(y_soil_true[m] - pipeline_preds[m]))
                mis_rates.append(is_misclassified[m].mean() * 100)
                n_samples.append(m.sum())
            else:
                maes.append(0)
                bias.append(0)      # Added this to keep list lengths consistent
                mis_rates.append(0)
                n_samples.append(0)
                
        # Data for Calibration Plots
        calib_y_true = y_true[mask_soil_true]
        calib_y_pred_all = y_pred_all[mask_soil_true]
        
        calib_df = evaluate_calibration_by_depth(
            calib_y_true, 
            calib_y_pred_all, 
            quantiles
        )
        overall_calib = calib_df[calib_df['class'] == 'Overall'].iloc[0]
        depth_calib = calib_df[calib_df['class'] != 'Overall']
        
        # --- CSV SAVES (placed here since data is ready) ---
        if save_path:
            # Scatter data
            scatter_df = pd.DataFrame({
                'observed': y_soil_true,
                'predicted': y_soil_pred,
                'is_misclassified': is_misclassified,
                'residual': y_soil_true - y_soil_pred,
                'abs_error': np.abs(y_soil_true - y_soil_pred)
            })
            csv_path = save_path.replace('.png', '_scatter_data.csv')
            scatter_df.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"  Saved scatter data: {csv_path}")

            # Depth class data
            depth_df = pd.DataFrame({
                'depth_class': labels,
                'n_samples': n_samples,
                'mae_m': maes,
                'misclassification_rate_pct': mis_rates
            })
            csv_path = save_path.replace('.png', '_by_depth_class.csv')
            depth_df.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"  Saved depth class data: {csv_path}")

            # Calibration data
            csv_path = save_path.replace('.png', '_calibration_by_depth.csv')
            calib_df.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"  Saved calibration data: {csv_path}")
        # ---------------------------------------------------

        # =========================================================================
        # PLOT 1: Observed vs Predicted (Row 0, Col 0)
        # =========================================================================
        ax_pipe = fig.add_subplot(gs[0, 0])
        
        # Plot correct predictions
        ax_pipe.scatter(y_soil_true[~is_misclassified], 
                       y_soil_pred[~is_misclassified], 
                       c='green', alpha=0.3, s=5, 
                       label=f'Correct Binary (n={(~is_misclassified).sum():,})',
                       edgecolors='none')
        
        # Plot misclassified (predicted as outcrop when it's soil)
        if is_misclassified.sum() > 0:
            ax_pipe.scatter(y_soil_true[is_misclassified], 
                           y_soil_pred[is_misclassified], 
                           c='red', marker='x', s=30, linewidths=1,
                           label=f'Binary Error (n={is_misclassified.sum():,})',
                           alpha=0.8)
        
        # 1:1 line
        max_val = max(y_soil_true.max(), y_soil_pred.max())
        ax_pipe.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=2, label='1:1 Line')
        
        # Annotations (Kept small annotations on the plot for context)
        metrics_text = (
            f"R² = {pipeline_eval['R2']:.2f}\n"
            f"CCC = {pipeline_eval['CCC']:.2f}\n"
            f"RMSE = {pipeline_eval['RMSE']:.2f}m\n"
            f"MAE = {pipeline_eval['MAE']:.2f}m\n"
            f"n = {pipeline_eval['n_samples']:,}"
        )
        ax_pipe.text(0.95, 0.4, metrics_text, 
                    transform=ax_pipe.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', 
                             edgecolor='beige', alpha=0.9, linewidth=1))
        
        mae_increase = pipeline_eval['MAE'] - reg_eval['MAE']
        error_text = (
            f"Binary Error Impact:\n"
            f"ΔMAE = {mae_increase:.2f}m\n"
            f"({100*mae_increase/pipeline_eval['MAE']:.1f}% of total)"
        )
        ax_pipe.text(0.95, 0.05, error_text,
                    transform=ax_pipe.transAxes,
                    fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white',
                             edgecolor='beige', alpha=0.9, linewidth=1))
        
        ax_pipe.set_xlabel('Observed Depth (m)', fontsize=10)
        ax_pipe.set_ylabel('Predicted Depth (m)', fontsize=10)
        ax_pipe.set_title(f'{title_prefix}Observed vs Predicted',
                         fontsize=12)
        ax_pipe.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_pipe.grid(alpha=0.3)
        
        # =========================================================================
        # PLOT 2: Error Analysis by Depth Class (Row 0, Col 1)
        # =========================================================================
        ax_err = fig.add_subplot(gs[0, 1])
        
        # Plot bars
        x_pos = np.arange(len(labels))
        bars = ax_err.bar(x_pos, maes, color='steelblue', alpha=0.7, 
                          label='MAE')
        
        # Add value annotations on bars
        for i, (bar, mae, n) in enumerate(zip(bars, maes, n_samples)):
            if n > 0:
                height = bar.get_height()
                ax_err.text(bar.get_x() + bar.get_width()/2., height,
                           f'{mae:.1f}m',
                           ha='center', va='bottom', fontsize=8)
                ax_err.text(bar.get_x() + bar.get_width()/2., 1,
                           f'n={n:,}',
                           ha='center', va='top', fontsize=7 )
        
        # Plot misclassification rate on secondary axis
        ax_err2 = ax_err.twinx()
        line = ax_err2.plot(x_pos, mis_rates, 'o-', color='darksalmon', 
                           linewidth=1.5, markersize=5,markeredgecolor='black',
                           label='Misclass. Rate (%)', alpha=0.9)
        
        for i, (x, rate) in enumerate(zip(x_pos, mis_rates)):
            if rate > 0:
                ax_err2.text(x + 0.3, rate + 0.1, f'{rate:.1f}%',
                            ha='center', va='bottom', fontsize=8,)
        
        # Formatting
        ax_err.set_xlabel('Depth Class', fontsize=10)
        ax_err.set_ylabel('MAE (m)', fontsize=10, color='steelblue')
        ax_err2.set_ylabel('Misclassification Rate (%)', fontsize=10, 
                     color='sienna')
        ax_err.set_title(f'{title_prefix}Error Analysis by Depth Class',
                        fontsize=12)
        ax_err.set_xticks(x_pos)
        ax_err.set_xticklabels(labels, rotation=30, ha='right')
        ax_err.tick_params(axis='y')
        ax_err2.tick_params(axis='y')
        ax_err.grid(alpha=0.3, axis='y')
        ax_err.set_ylim(bottom=0)
        ax_err2.set_ylim(bottom=0)
        lines1, labels1 = ax_err.get_legend_handles_labels()
        lines2, labels2 = ax_err2.get_legend_handles_labels()
        ax_err.legend(lines1 + lines2, labels1 + labels2, 
                    loc='upper left', 
                    bbox_to_anchor=(0.25, 1.0),
                    fontsize=9, framealpha=0.9)


        # =========================================================================
        # PLOT 3: PICP 90% by Depth Class (Bar) (Row 0, Col 2)
        # =========================================================================
        ax_pdep = fig.add_subplot(gs[0, 2])
        if 'PICP_90' in depth_calib.columns:
            # Colors based on deviation from 0.90
            cols = ['green' if abs(x-0.9)<0.05 else 'orange' if abs(x-0.9)<0.1 else 'darksalmon' 
                    for x in depth_calib['PICP_90']]
            ax_pdep.bar(depth_calib['class'], depth_calib['PICP_90']*100, color=cols, alpha=0.7, edgecolor='k')
            ax_pdep.axhline(90, color='r', ls='--', label='Target 90%')
            ax_pdep.set_title('90% Interval Coverage by Depth', fontsize=12)
            ax_pdep.set_xlabel('Depth Class', fontsize=10)
            ax_pdep.set_ylabel('Observed PICP (%)', fontsize=10)
            ax_pdep.tick_params(axis='x', rotation=30)
            ax_pdep.grid(alpha=0.3, axis='y')
            ax_pdep.legend(fontsize=9)
            
            # Add value annotations on bars
            for i, (bar, picp) in enumerate(zip(ax_pdep.patches, depth_calib['PICP_90'])):
                ax_pdep.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                             f'{picp*100:.1f}',
                             ha='center', va='bottom', fontsize=8)

        # =========================================================================
        # PLOT 4: PICP Reliability Plot (Scatter) (Row 1, Col 0)
        # =========================================================================
        ax_picp = fig.add_subplot(gs[1, 0])
        nom_cov, obs_picp = [], []
        
        # Generate all 10% to 90% PIs
        for nom in range(10, 100, 10):
            if f'PICP_{nom}' in overall_calib:
                nom_cov.append(nom)
                obs_picp.append(overall_calib[f'PICP_{nom}'] * 100)
        
        ax_picp.plot(nom_cov, obs_picp, 'o-', color='blue', label='Observed')
        ax_picp.plot([0,100], [0,100], 'k--', alpha=0.5, label='Ideal')
        ax_picp.set_title(f'{title_prefix}PICP Reliability', fontsize=12)
        ax_picp.set_xlabel('Nominal Coverage (%)', fontsize=10)
        ax_picp.set_ylabel('Observed PICP (%)', fontsize=10)
        ax_picp.set_xlim(0, 100)
        ax_picp.set_ylim(0, 100)
        
        ax_picp.grid(True, alpha=0.3)
 
        # Annotate 50% and 90% PIs
        for i, (nom, picp_val) in enumerate(zip(nom_cov, obs_picp)):
            if nom in [50, 90]:
                 ax_picp.annotate(f'{picp_val:.1f}', 
                    xy=(nom, picp_val), 
                    xytext=(-10, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='blue',
                    alpha=0.8)

        # Highlight 90% PI
        if 90 in nom_cov:
            idx_90 = nom_cov.index(90)
            ax_picp.plot(nom_cov[idx_90], obs_picp[idx_90], 'ro', markersize=10, 
                    markeredgewidth=2, markerfacecolor='none', label=f'90% PI (PICP={obs_picp[idx_90]:.1f}%)')
        ax_picp.legend(fontsize=9)
        
        # =========================================================================
        # PLOT 5: QCP Reliability (Scatter) (Row 1, Col 1)
        # =========================================================================
        ax_qcp = fig.add_subplot(gs[1, 1])
        q_levs, q_obs = [], []
        for q in quantiles:
            if f'QCP_{q:.3f}' in overall_calib:
                q_levs.append(q)
                q_obs.append(overall_calib[f'QCP_{q:.3f}'])
        
        ax_qcp.plot(q_levs, q_obs, 'o-', color='purple', label='Observed')
        ax_qcp.plot([0,1], [0,1], 'k--', alpha=0.5, label='Ideal')
        
        for q, qcp in zip(q_levs, q_obs):
            if q in [0.05, 0.50, 0.95]: 
                ax_qcp.annotate(f'{qcp*100:.1f}%', 
                        xy=(q, qcp), 
                        xytext=(-13, 8),
                        textcoords='offset points',
                        fontsize=8,
                        color='blue')
                        
        ax_qcp.set_title(f'{title_prefix}QCP Reliability (All True Soil)', fontsize=12)
        ax_qcp.set_xlabel('Quantile Level', fontsize=10)
        ax_qcp.set_ylabel('Observed Frequency', fontsize=10)
        ax_qcp.grid(True, alpha=0.3)
        ax_qcp.legend(fontsize=9)
        
        # =========================================================================
        # PLOT 6: QCP by Depth (Lines for select quantiles) (Row 1, Col 2)
        # =========================================================================
        ax_qdep = fig.add_subplot(gs[1, 2])
        target_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        key_qs = [q for q in target_quantiles if q in quantiles]
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(key_qs)))

        for q, color in zip(key_qs, colors):
            col_name = f'QCP_{q:.3f}'
            if col_name in depth_calib.columns:
                ax_qdep.plot(
                    depth_calib['class'], 
                    depth_calib[col_name], 
                    marker='o',
                    linestyle='-',
                    color=color,
                    markeredgecolor='black',
                    alpha=0.8,
                    label=f'{q:.2f}'
                )

        ax_qdep.set_yticks([0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
        ax_qdep.set_ylim(-0.05, 1.05)
        minor_ticks = np.arange(0, 1.1, 0.1)
        ax_qdep.set_yticks(minor_ticks, minor=True)
        ax_qdep.grid(which='minor', color='lightgrey', linestyle=':', alpha=0.3)
        ax_qdep.grid(which='major', color='grey', linestyle='--', alpha=0.5)

        for q in key_qs:
            ax_qdep.axhline(y=q, color='gray', linestyle=':', alpha=0.3, zorder=0)

        ax_qdep.set_title('QCP Stability by Depth', fontsize=12)
        ax_qdep.set_xlabel('Depth Class', fontsize=10)
        ax_qdep.set_ylabel('Observed Frequency', fontsize=10)
        ax_qdep.legend(title="Quantile", fontsize='small', edgecolor='white')
        ax_qdep.tick_params(axis='x', rotation=30)
    
    
    # -------------------------------------------------------------------------
    # NEW CODE: Save Summary Metrics to TXT and CSV
    # -------------------------------------------------------------------------

    # Create detailed summary text string (as done in the old plot)
    mae_increase = pipeline_eval['MAE'] - reg_eval['MAE']
    summary_text = f"""
        ========================================
        FULL PIPELINE EVALUATION
        ========================================

        OVERALL PERFORMANCE (All True Soil):
        Samples:          {pipeline_eval['n_samples']:,}
        MAE:              {pipeline_eval['MAE']:.3f} m
        RMSE:             {pipeline_eval['RMSE']:.3f} m
        R²:               {pipeline_eval['R2']:.3f}
        Bias:             {pipeline_eval['Bias']:.3f} m
        PICP (90%):       {pipeline_eval['PICP_90']:.1%}
        PI Width (90%):   {pipeline_eval['PI_Width_90']:.2f} m

        BINARY MODEL IMPACT:
        Misclassified:    {pipeline_eval['n_misclassified_as_outcrop']:,}
        Error Rate:       {pipeline_eval['pct_misclassified']:.2f}%

        COMPARISON (Pipeline vs Clean):
        Clean MAE:        {reg_eval['MAE']:.3f} m
        Pipeline MAE:     {pipeline_eval['MAE']:.3f} m
        ΔMAE:             {mae_increase:+.3f} m
        
        Clean R²:         {reg_eval['R2']:.3f}
        Pipeline R²:      {pipeline_eval['R2']:.3f}
        ΔR²:              {pipeline_eval['R2'] - reg_eval['R2']:+.3f}

        ERROR ATTRIBUTION:
        Binary contributes {100*mae_increase/pipeline_eval['MAE']:.1f}% of total MAE
        QRF contributes    {100*reg_eval['MAE']/pipeline_eval['MAE']:.1f}% of total MAE

        ========================================
        """

    # Save summary metrics to CSV
    if save_path:
        # 1. Save to TXT file
        txt_path = save_path.replace('.png', '_summary_metrics.txt')
        with open(txt_path, 'w') as f:
            f.write(summary_text.strip()) # Remove leading/trailing whitespace
        print(f"  Saved summary metrics: {txt_path}")

        # 2. Save to CSV file (structured data)
        summary_df = pd.DataFrame({
            'metric': ['n_samples', 'MAE', 'RMSE', 'R2', 'Bias', 'PICP_90', 
                      'PI_Width_90', 'n_misclassified', 'pct_misclassified',
                      'clean_MAE', 'clean_R2', 'delta_MAE', 'delta_R2',
                      'binary_contribution_pct', 'qrf_contribution_pct'],
            'value': [
                pipeline_eval['n_samples'],
                pipeline_eval['MAE'],
                pipeline_eval['RMSE'],
                pipeline_eval['R2'],
                pipeline_eval['Bias'],
                pipeline_eval['PICP_90'],
                pipeline_eval['PI_Width_90'],
                pipeline_eval['n_misclassified_as_outcrop'],
                pipeline_eval['pct_misclassified'],
                reg_eval['MAE'],
                reg_eval['R2'],
                mae_increase,
                pipeline_eval['R2'] - reg_eval['R2'],
                100*mae_increase/pipeline_eval['MAE'],
                100*reg_eval['MAE']/pipeline_eval['MAE']
            ]
        })
        csv_path = save_path.replace('.png', '_summary_metrics.csv')
        summary_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"  Saved summary metrics: {csv_path}")
    
    # Final save and show logic
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def plot_spatial_results(y_train, y_test, residuals, X_train, X_test, 
                         save_path=None, fold_num=None):
    """
    Create spatial distribution plots (1x3 grid).
    
    Args:
        y_train: Training target values
        y_test: Test target values
        residuals: Test residuals (y_test - y_pred_50)
        X_train: Training features (must include 'N' and 'E')
        X_test: Test features (must include 'N' and 'E')
        save_path: Path to save the plot
        fold_num: Optional fold number for title
    """
    if 'N' not in X_test.columns or 'E' not in X_test.columns:
        print("  ⚠ Skipping spatial plot: missing 'N' or 'E' coordinates")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15)
    title_prefix = f'Fold {fold_num} - ' if fold_num is not None else ''

    # Color mappings
    depth_cmap = plt.cm.plasma
    depth_norm = plt.cm.colors.Normalize(vmin=0, vmax=150)

    # Plot A: Training Data
    ax = axes[0]
    scatter_train = ax.scatter(X_train['E'], X_train['N'], 
                              c=y_train, cmap=depth_cmap, norm=depth_norm,
                              s=0.5, alpha=0.6, edgecolors='none')
    ax.set_title(title_prefix + f'Training Data (n={len(y_train):,})', 
                fontsize=10)
    ax.set_ylabel('Northing (N)', fontsize=10)
    ax.set_xlabel('Easting (E)', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    cbar1 = fig.colorbar(scatter_train, ax=ax, orientation='vertical', pad=0.02)
    cbar1.set_label('Depth to Bedrock (m)', fontsize=9)

    # Plot B: Test Data
    ax = axes[1]
    scatter_test = ax.scatter(X_test['E'], X_test['N'], 
                              c=y_test, cmap=depth_cmap, norm=depth_norm,
                              s=0.5, alpha=0.6, edgecolors='none')
    ax.set_title(title_prefix + f'Test Data (n={len(y_test):,})', 
                fontsize=10)
    ax.set_xlabel('Easting (E)', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    cbar2 = fig.colorbar(scatter_test, ax=ax, orientation='vertical', pad=0.02)
    cbar2.set_label('Depth to Bedrock (m)', fontsize=9)
    
    # Plot C: Spatial Residuals
    ax = axes[2]
    max_abs_resid = max(abs(residuals.min()), abs(residuals.max()))
    norm = SymLogNorm(linthresh=0.1, vmin=-max_abs_resid, vmax=max_abs_resid) 

    scatter_resid = ax.scatter(
        X_test['E'], X_test['N'],
        c=residuals, cmap='RdBu_r', norm=norm,
        s=0.5, alpha=0.8, edgecolors='none'
    )
    ax.set_title(title_prefix + 'Spatial Residuals', fontsize=10)
    ax.set_xlabel('Easting (E)', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    cbar3 = fig.colorbar(scatter_resid, ax=ax, orientation='vertical', pad=0.02)
    cbar3.set_label('Residual (m)', fontsize=9)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved spatial plot: {save_path}")
        plt.close(fig)
    
    return fig


def plot_optimization_history(study, save_path=None, title='Optimization History'):
    """
    Plot Optuna optimization history.
    
    Args:
        study: Optuna study object
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Extract completed trials
    completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    
    if len(completed_trials) == 0:
        print(f"  ⚠ No completed trials to plot for {title}")
        return
    
    trial_numbers = [t.number for t in completed_trials]
    objective_values = [t.value for t in completed_trials]
    
    # Plot 1: Objective Value History
    axes[0].plot(trial_numbers, objective_values, 'o-', alpha=0.7, linewidth=1, markersize=4)
    axes[0].axhline(y=min(objective_values), color='r', linestyle='--', 
                    label=f'Best: {min(objective_values):.4f}', alpha=0.7)
    axes[0].set_xlabel('Trial Number', fontsize=10)
    axes[0].set_ylabel('Objective Value', fontsize=10)
    axes[0].set_title(f'{title} - Objective Value', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Cumulative Best Value
    cumulative_best = []
    current_best = float('inf')
    for val in objective_values:
        current_best = min(current_best, val)
        cumulative_best.append(current_best)
    
    axes[1].plot(trial_numbers, cumulative_best, 'o-', color='green', 
                alpha=0.7, linewidth=2, markersize=4, label='Cumulative Best')
    axes[1].set_xlabel('Trial Number', fontsize=10)
    axes[1].set_ylabel('Best Objective Value', fontsize=10)
    axes[1].set_title(f'{title} - Convergence', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved optimization plot: {save_path}")
        plt.close(fig)
    
    return fig