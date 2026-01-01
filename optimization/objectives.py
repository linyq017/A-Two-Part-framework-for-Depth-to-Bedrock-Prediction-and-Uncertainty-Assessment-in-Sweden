"""Optuna objective functions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error
from quantile_forest import RandomForestQuantileRegressor
import optuna


def objective_binary(trial, X, y_binary):
    """
    Binary model objective function with proper error handling.
    
    Args:
        trial: Optuna trial object
        X: Feature matrix
        y_binary: Binary labels (0 or 1)
    
    Returns:
        Combined score (lower is better)
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = RandomForestClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    logloss_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_binary)):
        X_train = X.iloc[train_idx]
        y_train = y_binary.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y_binary.iloc[val_idx]
        
        # Check if validation fold has both classes
        if len(np.unique(y_val)) < 2:
            print(f"  Warning: Fold {fold_idx} has only one class, skipping...")
            continue
        
        try:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            
            # Handle case where model only predicts one class
            if y_pred_proba.shape[1] == 1:
                print(f"  Warning: Fold {fold_idx} - model only predicts one class, skipping...")
                continue
            
            # Extract probability for positive class (outcrop = 1)
            y_pred_proba_positive = y_pred_proba[:, 1]
            
            # Calculate metrics only if both classes present
            if len(np.unique(y_val)) == 2:
                auc_scores.append(roc_auc_score(y_val, y_pred_proba_positive))
                logloss_scores.append(log_loss(y_val, y_pred_proba_positive))
            
        except Exception as e:
            print(f"  Warning: Fold {fold_idx} failed with error: {e}")
            continue
    
    # Need at least 2 folds to compute meaningful average
    if len(auc_scores) < 2:
        # Return a penalty score
        return 10.0  # High penalty for failed trials
    
    combined_score = -np.mean(auc_scores) + 0.5 * np.mean(logloss_scores)
    
    trial.set_user_attr("mean_auc", np.mean(auc_scores))
    trial.set_user_attr("mean_logloss", np.mean(logloss_scores))
    trial.set_user_attr("n_valid_folds", len(auc_scores))
    
    return combined_score


def objective_qrf(trial, X, y):
    """
    QRF model objective function optimizing ONLY for MAE (Median accuracy).
    
    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Continuous target values
    
    Returns:
        Mean Validation MAE (lower is better)
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 600),
        'max_depth': trial.suggest_int('max_depth', 30, 60),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
        'max_samples': trial.suggest_float('max_samples', 0.5, 0.8),
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Ensure this is imported or defined in your scope
    from config import DEPTH_BINS 
    
    model = RandomForestQuantileRegressor(**params)
    
    # Binning for StratifiedKFold
    y_binned = pd.cut(y, bins=DEPTH_BINS, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    val_mae_scores = []
    
    for step, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        X_fold = X.iloc[train_idx]
        y_fold = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Check for sufficient samples
        if len(y_fold) < 10 or len(y_val) < 5:
            print(f"  Warning: Fold {step} has too few samples, skipping...")
            continue
        
        try:
            model.fit(X_fold, y_fold)
            
            # OPTIMIZATION: Only predict the median (0.5) to save time
            # We don't need lower/upper quantiles for MAE optimization
            y_pred_median = model.predict(X_val, quantiles=[0.5])
            
            # Flatten to ensure 1D array
            y_pred_median = y_pred_median.ravel()
            
            # Calculate MAE only
            val_mae = mean_absolute_error(y_val, y_pred_median)
            
            # Check for NaN or inf
            if np.isnan(val_mae) or np.isinf(val_mae):
                print(f"  Warning: Fold {step} produced invalid metrics, skipping...")
                continue
            
            val_mae_scores.append(val_mae)
            
            # Report intermediate value for pruning
            trial.report(np.mean(val_mae_scores), step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        except optuna.TrialPruned:
            raise  # Re-raise pruning exceptions
        except Exception as e:
            print(f"  Warning: Fold {step} failed with error: {e}")
            continue
    
    # Need at least 2 folds to compute meaningful average
    if len(val_mae_scores) < 2:
        return 1000.0  # High penalty for failed trials
    
    mean_val_mae = np.mean(val_mae_scores)
    
    # Log the metric for analysis later
    trial.set_user_attr("mean_val_mae", mean_val_mae)
    trial.set_user_attr("n_valid_folds", len(val_mae_scores))
    
    return mean_val_mae

# def objective_qrf(trial, X, y):
#     """
#     QRF model objective function with proper error handling.
    
#     Args:
#         trial: Optuna trial object
#         X: Feature matrix
#         y: Continuous target values
    
#     Returns:
#         Combined score (lower is better)
#     """
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 300, 600),
#         'max_depth': trial.suggest_int('max_depth', 30, 60),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
#         'max_samples': trial.suggest_float('max_samples', 0.5, 0.8),
#         'n_jobs': -1,
#         'random_state': 42
#     }
    
#     from config import DEPTH_BINS
    
#     model = RandomForestQuantileRegressor(**params)
#     y_binned = pd.cut(y, bins=DEPTH_BINS, labels=False)
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
#     val_mae_scores = []
#     val_interval_scores = []
    
#     alpha = 0.1  # 90% prediction interval
    
#     for step, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
#         X_fold = X.iloc[train_idx]
#         y_fold = y.iloc[train_idx]
#         X_val = X.iloc[val_idx]
#         y_val = y.iloc[val_idx]
        
#         # Check for sufficient samples
#         if len(y_fold) < 10 or len(y_val) < 5:
#             print(f"  Warning: Fold {step} has too few samples, skipping...")
#             continue
        
#         try:
#             model.fit(X_fold, y_fold)
            
#             # Make predictions for different quantiles
#             y_pred_median = model.predict(X_val, quantiles=[0.5])
#             y_pred_lower = model.predict(X_val, quantiles=[alpha/2])
#             y_pred_upper = model.predict(X_val, quantiles=[1 - alpha/2])
            
#             # Flatten if needed
#             y_pred_median = y_pred_median.ravel()
#             y_pred_lower = y_pred_lower.ravel()
#             y_pred_upper = y_pred_upper.ravel()
            
#             # Calculate MAE
#             val_mae = mean_absolute_error(y_val, y_pred_median)
            
#             # Calculate interval score
#             width = y_pred_upper - y_pred_lower
#             penalty_below = (2/alpha) * (y_pred_lower - y_val.values) * (y_val.values < y_pred_lower)
#             penalty_above = (2/alpha) * (y_val.values - y_pred_upper) * (y_val.values > y_pred_upper)
#             val_interval = np.mean(width + penalty_below + penalty_above)
            
#             # Check for NaN or inf
#             if np.isnan(val_mae) or np.isinf(val_mae) or np.isnan(val_interval) or np.isinf(val_interval):
#                 print(f"  Warning: Fold {step} produced invalid metrics, skipping...")
#                 continue
            
#             val_mae_scores.append(val_mae)
#             val_interval_scores.append(val_interval)
            
#             # Report intermediate value for pruning
#             trial.report(np.mean(val_mae_scores), step)
            
#             # Check if trial should be pruned
#             if trial.should_prune():
#                 raise optuna.TrialPruned()
                
#         except optuna.TrialPruned:
#             raise  # Re-raise pruning exceptions
#         except Exception as e:
#             print(f"  Warning: Fold {step} failed with error: {e}")
#             continue
    
#     # Need at least 2 folds to compute meaningful average
#     if len(val_mae_scores) < 2:
#         # Return a penalty score
#         return 1000.0  # High penalty for failed trials
    
#     mean_val_mae = np.mean(val_mae_scores)
#     mean_val_interval = np.mean(val_interval_scores)
    
#     trial.set_user_attr("mean_val_mae", mean_val_mae)
#     trial.set_user_attr("mean_val_interval", mean_val_interval)
#     trial.set_user_attr("n_valid_folds", len(val_mae_scores))
    
#     # Combined score: 70% MAE + 30% interval score
#     combined_score = 0.7 * mean_val_mae + 0.3 * mean_val_interval
    
#     return combined_score