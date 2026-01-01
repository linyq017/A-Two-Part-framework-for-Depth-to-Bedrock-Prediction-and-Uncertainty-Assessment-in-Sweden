"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def subset_columns(data, feature_cols):
    """Subset relevant columns."""
    required_cols = ['DJUP'] + feature_cols
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return data[required_cols]


def stratified_subsample(X_full, y_full, frac, depth_bins, depth_labels, random_state=42):
    """Subsample with stratification."""
    if frac >= 1.0:
        return X_full.copy(), y_full.copy()
    
    y_binned = pd.cut(y_full, bins=depth_bins, labels=depth_labels)
    indices = np.arange(len(X_full))
    train_idx, _ = train_test_split(
        indices, train_size=frac, stratify=y_binned, random_state=random_state
    )
    return X_full.iloc[train_idx].copy(), y_full.iloc[train_idx].copy()

def create_output_dir(data_type, subsample_frac, timestamp, fold_name):
    """
    Create output directory with dynamic fold naming.
    fold_name could be 'fold1', 'fold4', etc.
    """
    output_dir = (
        f'/workspace/data/soildepth/Hypertune/TwoStageFusion/'
        f'{data_type}_{subsample_frac:.4f}_{timestamp}_{fold_name}'
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_results_to_csv(output_dir, results_dict):
    """Save dataframes to CSV."""
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    for name, df in results_dict.items():
        path = f'{csv_dir}/{name}.csv'
        df.to_csv(path, index=False, float_format='%.4f')
        print(f"  Saved {name} to {path}")
