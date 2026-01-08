"""
Main workflow script for two-part soil depth prediction model.

Usage:
    # Default run
    python main.py
    
    # Custom parameters (Data type is now hardcoded in config)
    python main.py --subsample 1.0 --binary_trials 20 --qrf_trials 30
    
    # Quick test with less data
    python main.py --subsample 0.001 --binary_trials 1 --qrf_trials 1
"""
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import shap
from tqdm import tqdm
from config import DATA_PATHS
import os
# Import from modules
from config import (
    DATA_PATHS, QUANTILES_FULL, DEPTH_BINS, DEPTH_LABELS,
    FEATURE_COLUMNS, PREDICTION_CONFIG
)
from data_utils import (
    subset_columns, stratified_subsample, 
    create_output_dir, save_results_to_csv
)
from metrics import (
    evaluate_fused_model, 
    print_evaluation_summary, 
)
from models.binary_model import train_binary_model
from models.qrf_model import train_qrf_model
from models.fusion import fuse_predictions
from visualization.plots import (
    plot_twopart_evaluation,
    plot_spatial_results, 
    plot_optimization_history,
    plot_pipeline_evaluation_enhanced
)

from evaluation import comprehensive_fused_evaluation

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def load_and_prepare_data(data_type, subsample_frac):
    """Load data, subset columns, and optionally subsample."""
    print(f"\n{'='*70}")
    print("STEP 1: LOADING DATA")
    print(f"{'='*70}")
    
    train_path = DATA_PATHS['train']
    test_path = DATA_PATHS['test']
    fold_name = os.path.basename(train_path).split('_')[0]
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")
    
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    
    data_train = subset_columns(data_train, FEATURE_COLUMNS)
    data_test = subset_columns(data_test, FEATURE_COLUMNS)
    
    X_train = data_train.drop(columns=['DJUP'])
    y_train = data_train['DJUP']
    X_test = data_test.drop(columns=['DJUP'])
    y_test = data_test['DJUP']
    
    print(f"\nData loaded:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Train depth range: {y_train.min():.2f}m - {y_train.max():.2f}m")
    print(f"  Train outcrops: {(y_train == 0).sum():,} ({100*(y_train == 0).mean():.1f}%)")
    
    if subsample_frac < 1.0:
        print(f"\nSubsampling to {subsample_frac:.1%}...")
        X_train, y_train = stratified_subsample(
            X_train, y_train, subsample_frac, DEPTH_BINS, DEPTH_LABELS
        )
        print(f"  Training samples after subsampling: {len(X_train):,}")
    
    return X_train, y_train, X_test, y_test, fold_name


def train_models(X_train, y_train, output_dir, n_binary_trials, n_qrf_trials):
    """Train both binary and QRF models."""
    print(f"\n{'='*70}")
    print("STEP 2: TRAINING BINARY MODEL (OUTCROP CLASSIFIER)")
    print(f"{'='*70}")
    
    binary_model, binary_study = train_binary_model(
        X_train, y_train, output_dir, n_trials=n_binary_trials
    )
    
    print(f"\n{'='*70}")
    print("STEP 3: TRAINING QRF MODEL (DEPTH REGRESSOR)")
    print(f"{'='*70}")
    
    qrf_model, quantiles, qrf_study = train_qrf_model(
        X_train, y_train, output_dir, 
        quantiles=QUANTILES_FULL, 
        n_trials=n_qrf_trials
    )
    
    return binary_model, binary_study, qrf_model, quantiles, qrf_study


def make_predictions(binary_model, qrf_model, X_test, y_test, quantiles, binary_threshold):
    """Generate predictions from both models and fuse them."""
    print(f"\n{'='*70}")
    print("STEP 4: PREDICTION ON TEST SET")
    print(f"{'='*70}")
    
    print("  Making binary predictions...")
    binary_proba = binary_model.predict_proba(X_test)[:, 1]
    
    print("  Making QRF predictions...")
    qrf_predictions = qrf_model.predict(
        X_test,
        quantiles=quantiles,
        **PREDICTION_CONFIG
    )
    
    print("  Fusing predictions...")
    fused_results = fuse_predictions(
        binary_proba=binary_proba,
        qrf_predictions=qrf_predictions,
        binary_threshold=binary_threshold
    )
    
    print(f"\nPrediction summary:")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Predicted outcrops: {fused_results['is_outcrop'].sum():,} "
          f"({100*fused_results['is_outcrop'].mean():.1f}%)")
    print(f"  Predicted depth: {(~fused_results['is_outcrop']).sum():,} "
          f"({100*(~fused_results['is_outcrop']).mean():.1f}%)")
    print(f"  True outcrops: {(y_test == 0).sum():,} "
          f"({100*(y_test == 0).mean():.1f}%)")
    
    return binary_proba, qrf_predictions, fused_results


def calculate_qrf_shap(qrf_model, X_test, background_size=1000, batch_size=500):
    """Calculates SHAP values for the QRF model with a progress bar."""
    print(f"\n  {'='*30}")
    print("  CALCULATING SHAP EXPLANATIONS")
    print(f"  {'='*30}")
    
    model_dict = {
        "objective": qrf_model.criterion,
        "tree_output": "raw_value",
        "trees": [estimator.tree_ for estimator in qrf_model.estimators_]
    }
    
    if len(X_test) > background_size:
        background_idx = np.random.choice(len(X_test), size=background_size, replace=False)
        background = X_test.iloc[background_idx]
        print(f"  Using {len(background)} samples as background for SHAP")
    else:
        background = X_test
        print(f"  Using all {len(background)} samples as background for SHAP")
        
    explainer = shap.TreeExplainer(model_dict, data=background)
    
    all_shap_values = []
    print(f"  Processing {len(X_test)} samples in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(X_test), batch_size), desc="  SHAP Progress"):
        batch = X_test.iloc[i:i+batch_size]
        shap_batch = explainer.shap_values(batch, check_additivity=False)
        
        if isinstance(shap_batch, list):
            shap_batch = shap_batch[0]
            
        all_shap_values.append(shap_batch)
    
    shap_values_array = np.vstack(all_shap_values)
    
    if len(shap_values_array.shape) > 2:
        shap_values_array = shap_values_array[:, :, 0]

    print("  ✅ SHAP values successfully calculated.")
    return pd.DataFrame(shap_values_array, columns=X_test.columns)


def create_visualizations(y_test, fused_results, quantiles, comprehensive_results,
                         X_train, X_test, y_train, 
                         binary_study, qrf_study, output_dir):
    """Generate all visualization plots."""
    print(f"\n{'='*70}")
    print("STEP 6: CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"  Plots directory: {plots_dir}")
    
    # Get evaluation from comprehensive results
    evaluation = comprehensive_results['fused']
    
    # Main evaluation plot
    print("  Creating two-part evaluation plot...")
    from evaluation import evaluate_calibration_by_depth
    plot_twopart_evaluation(
        y_test, fused_results, quantiles, 
        evaluate_calibration_by_depth, evaluation,
        save_path=f'{plots_dir}/twopart_model_evaluation.png'
    )
    
    # Spatial plot if coordinates available
    if 'N' in X_test.columns and 'E' in X_test.columns:
        print("  Creating spatial results plot...")
        q_map = {q: i for i, q in enumerate(quantiles)}
        y_pred_median = fused_results['fused_predictions'][:, q_map[0.5]]
        residuals = y_test - y_pred_median
        
        plot_spatial_results(
            y_train, y_test, residuals, X_train, X_test,
            save_path=f'{plots_dir}/spatial_results.png'
        )
    
    # Optimization history plots
    print("  Creating optimization history plots...")
    plot_optimization_history(
        binary_study, 
        save_path=f'{plots_dir}/binary_optimization.png', 
        title='Binary Model Optimization'
    )
    plot_optimization_history(
        qrf_study, 
        save_path=f'{plots_dir}/qrf_optimization.png',
        title='QRF Model Optimization'
    )
    
    # Enhanced pipeline plot
    print("  Creating enhanced pipeline plot...")
    plot_pipeline_evaluation_enhanced(
        y_test, fused_results, quantiles, evaluation,
        save_path=f'{plots_dir}/pipeline_enhanced.png',
        title_prefix=""
    )
    
    print("  ✓ All visualizations created")


def save_all_results(output_dir, comprehensive_results, #shap_df,
                     y_test, fused_results, quantiles,
                     data_type, subsample_frac, n_binary_trials, n_qrf_trials,
                     binary_threshold, timestamp, X_train, X_test):
    """Save all results to CSV and text files."""
    print(f"\n{'='*70}")
    print("STEP 7: SAVING RESULTS")
    print(f"{'='*70}")
    
    # Extract from comprehensive results
    evaluation = comprehensive_results['fused']
    
    # Save predictions
    q_map = {q: i for i, q in enumerate(quantiles)}
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_median': fused_results['fused_predictions'][:, q_map[0.5]],
        'outcrop_proba': fused_results['outcrop_proba'],
        'is_outcrop_pred': fused_results['is_outcrop'],
        'is_outcrop_true': (y_test == 0)
    })
    predictions_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)
    print("  ✓ Saved test_predictions.csv")
    
    # if not shap_df.empty:
    #     shap_df.to_csv(f'{output_dir}/shap_values.csv', index=False)
    #     print("  ✓ Saved shap_values.csv")
    
    # Save summary text
    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write(f"Two-Part Model Results\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Data Type: {data_type}\n")
        f.write(f"  Subsample Fraction: {subsample_frac}\n")
        f.write(f"  Binary Trials: {n_binary_trials}\n")
        f.write(f"  QRF Trials: {n_qrf_trials}\n")
        f.write(f"  Binary Threshold: {binary_threshold}\n")
        f.write(f"  Timestamp: {timestamp}\n\n")
        
        f.write(f"Data:\n")
        f.write(f"  Training Samples: {len(X_train):,}\n")
        f.write(f"  Test Samples: {len(X_test):,}\n")
        f.write(f"  Features: {len(X_train.columns)}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Binary AUC: {evaluation['binary']['AUC']:.4f}\n")
        f.write(f"  Binary Accuracy: {evaluation['binary']['Accuracy']:.4f}\n")
        f.write(f"  Binary F1-Score: {evaluation['binary']['F1_Score']:.4f}\n\n")
        
        f.write(f"  Regression (Clean) MAE: {evaluation['regression_clean']['MAE']:.2f}m\n")
        f.write(f"  Regression (Clean) RMSE: {evaluation['regression_clean']['RMSE']:.2f}m\n")
        f.write(f"  Regression (Clean) R²: {evaluation['regression_clean']['R2']:.4f}\n")
        f.write(f"  Regression (Clean) CCC: {evaluation['regression_clean']['CCC']:.4f}\n\n")
        
        f.write(f"  Full Pipeline MAE: {evaluation['full_pipeline']['MAE']:.2f}m\n")
        f.write(f"  Full Pipeline RMSE: {evaluation['full_pipeline']['RMSE']:.2f}m\n")
        f.write(f"  Full Pipeline R²: {evaluation['full_pipeline']['R2']:.4f}\n")
        f.write(f"  Misclassified: {evaluation['full_pipeline']['n_misclassified_as_outcrop']} "
                f"({evaluation['full_pipeline']['pct_misclassified']:.1f}%)\n")
    
    print("  ✓ Saved summary.txt")
    print("  ✓ All results saved")


def run_two_part_workflow(data_type='censored', 
                          subsample_frac=1, 
                          n_binary_trials=20, 
                          n_qrf_trials=20,
                          binary_threshold=0.5):
    """
    Run complete two-part workflow.
    
    FIXED VERSION: Uses comprehensive_fused_evaluation() ONCE to get all metrics.
    No more redundant evaluation calls!
    """
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    # Execute workflow steps
    # 1. Capture the fold_name from the function
    X_train, y_train, X_test, y_test, fold_name = load_and_prepare_data(data_type, subsample_frac)
    
    output_dir = create_output_dir(data_type, subsample_frac, timestamp, fold_name)
    
    print(f"\n{'='*70}")
    print(f"TWO-PART MODEL WORKFLOW: {data_type.upper()}")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Binary threshold: {binary_threshold}")
    print(f"Timestamp: {timestamp}")


    # 2. Pass that fold_name into your directory creation function
    binary_model, binary_study, qrf_model, quantiles, qrf_study = train_models(
        X_train, y_train, output_dir, n_binary_trials, n_qrf_trials
    )
    
    binary_proba, qrf_predictions, fused_results = make_predictions(
        binary_model, qrf_model, X_test, y_test, quantiles, binary_threshold
    )
    

    # ===========================================================================
    # SINGLE COMPREHENSIVE EVALUATION 
    # ===========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: COMPREHENSIVE EVALUATION")
    print(f"{'='*70}")
    
    comprehensive_results = comprehensive_fused_evaluation(
        y_test=y_test,
        outcrop_proba=binary_proba,
        y_pred_all=qrf_predictions,
        quantiles=quantiles,
        output_dir=f'{output_dir}/comprehensive_evaluation',
        threshold=binary_threshold
    )
    
    # This function already:
    # - Evaluates binary component
    # - Evaluates regression component (QRF only on depth > 0)
    # - Evaluates fused model (three perspectives)
    # - Runs threshold sensitivity analysis
    # - Evaluates by depth class
    # - Saves all CSVs
    # - Creates plots
    
    print("  ✓ Comprehensive evaluation complete")
    
    # ===========================================================================
    # Visualizations and Final Save
    # ===========================================================================
    create_visualizations(
        y_test, fused_results, quantiles, comprehensive_results,
        X_train, X_test, y_train, binary_study, qrf_study, output_dir
    )
        # Calculate SHAP
    # shap_df = pd.DataFrame()
    # try:
    #     shap_df = calculate_qrf_shap(qrf_model, X_test)
    # except Exception as e:
    #     print(f"\n  ⚠️ SHAP calculation failed: {e}")
    #     print("  Continuing workflow without SHAP values...")

    save_all_results(
        output_dir, comprehensive_results, #shap_df,
        y_test, fused_results, quantiles,
        data_type, subsample_frac, n_binary_trials, n_qrf_trials,
        binary_threshold, timestamp, X_train, X_test
    )
    
    print(f"\n{'='*70}")
    print("✓ WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'binary_model': binary_model,
        'qrf_model': qrf_model,
        'binary_study': binary_study,
        'qrf_study': qrf_study,
        'fused_results': fused_results,
        'comprehensive_results': comprehensive_results,
        'output_dir': output_dir,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'quantiles': quantiles,
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Two-Part Soil Depth Prediction Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--subsample', 
        type=float, 
        default=1.0,
        help='Fraction of training data to use (0.0-1.0)'
    )
    parser.add_argument(
        '--binary_trials', 
        type=int, 
        default=20,
        help='Number of Optuna trials for binary model'
    )
    parser.add_argument(
        '--qrf_trials', 
        type=int, 
        default=20,
        help='Number of Optuna trials for QRF model'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Binary classification threshold (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.subsample <= 1.0:
        parser.error("--subsample must be between 0 and 1")
    if not 0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.binary_trials < 1:
        parser.error("--binary_trials must be at least 1")
    if args.qrf_trials < 1:
        parser.error("--qrf_trials must be at least 1")
    
    # Run workflow
    print("\n" + "="*70)
    print("STARTING TWO-PART MODEL WORKFLOW")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Subsample: {args.subsample:.1%}")
    print(f"  Binary Trials: {args.binary_trials}")
    print(f"  QRF Trials: {args.qrf_trials}")
    print(f"  Threshold: {args.threshold}")
    
    results = run_two_part_workflow(
        subsample_frac=args.subsample,
        n_binary_trials=args.binary_trials,
        n_qrf_trials=args.qrf_trials,
        binary_threshold=args.threshold
    )
    
    print("\n" + "="*70)
    print("✓ SUCCESS - Workflow completed successfully!")
    print("="*70)
    
    return results


if __name__ == '__main__':
    results = main()
