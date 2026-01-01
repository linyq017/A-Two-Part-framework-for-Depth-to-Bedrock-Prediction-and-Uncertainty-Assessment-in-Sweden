
# Two-Part (Hurdle) Machine Learning Framework for Depth to Bedrock Mapping

## Overview

This repository contains the implementation of a Two-Part (Hurdle) Machine Learning Framework designed to map Depth to Bedrock (DTB). Standard regression models often struggle with "structural zeros" (bedrock outcrops) common in post-glacial terrains like Sweden, leading to biased predictions and unreliable uncertainty estimates.

Our solution explicitly decouples the modeling process into two stages:

- **Classification**: Identifying bedrock outcrops vs. sediments using a Binary Random Forest
- **Regression**: Predicting depth to bedrock (DTB) using a Quantile Regression Forest (QRF)

By integrating these components, this framework outperformed a national IDW interpolation-based model and a global DTB map.

## Key Features

- **Hurdle Model Architecture**: Handles zero-inflated data by separating the probability of occurrence (bedrock) from the continuous value (depth)
- **Uncertainty Quantification**: Leverages Quantile Regression Forests (QRF) to generate prediction intervals (PIs) and uncertainty maps
- **Depth-Stratified Evaluation**: Includes scripts for evaluating model performance across distinct depth zones (0–2m, 2–10m, >30m), revealing critical insights often masked by global metrics

## Project Structure

```
├── models/                  # Core model definitions
│   ├── binary_model.py      # Random Forest Classifier (Bedrock vs. Soil)
│   ├── qrf_model.py         # Quantile Regression Forest (Depth prediction)
│   └── fusion.py            # Logic to combine classification and regression results
├── optimization/            # Hyperparameter tuning
│   └── objectives.py        # Objective functions for model optimization
├── visualization/           # Plotting and Analysis
│   └── plots.py             # Scripts for generating maps and SHAP plots
├── config.py                # Global configuration (paths, hyperparameters)
├── data_utils.py            # Data loading and preprocessing functions
├── evaluation.py            # Metrics calculation (RMSE, PICP, etc.)
├── main.py                  # Main entry point for training and prediction
├── metrics.py               # Custom metric definitions
├── prediction.py            # Inference script for generating final maps
├── README.md                # This file - project documentation
└── Dockerfile               # Docker configuration 
```

## Usage

### Basic Usage

The entire pipeline runs through the `main.py` script. To run the training and evaluation workflow using the default configuration:

```bash
python main.py
```

To run with custom parameters or a quick test:
```bash
 # Custom parameters
 python main.py --subsample 1.0 --binary_trials 20 --qrf_trials 30
 
 # Quick test with a subsample of data
 python main.py --subsample 0.001 --binary_trials 1 --qrf_trials 1
```

This will:
1. Loads training/testing splits from config.py, subsets features, and optionally applies stratified subsampling for rapid prototyping
2. Trains a Random Forest to identify bedrock outcrops (Depth = 0), optimized via Optuna.
3. Trains a Quantile Regression Forest (QRF) on the soil subset to predict depth and uncertainty intervals.
4. Combines outputs using a Hurdle mechanism—predicting 0 m if the outcrop probability exceeds the threshold, otherwise utilizing the QRF prediction.
5. Calculates depth-stratified metrics (RMSE, Bias, PICP), generates SHAP explanations, and produces validation plots.
6. Saves trained models (.joblib), prediction logs (.csv), and a summary report (.txt) to a timestamped results directory.

   
## Example Workflow

1. **Setup**: Ensure data is in `data/` and paths are set in `config.py`

2. **Training**: Run `python main.py`
   - The script trains the Binary Model first
   - It then trains the QRF model on the "Soil" subset
   - Metrics are calculated and printed

3. **Inference**: Run `python prediction.py`
   - Loads the saved `.joblib` models
   - Predicts depth and uncertainty for target raster tiles


### Configuration

All pipeline settings are managed in `config.py`.  

Edit `config.py` to set your paths:

```python
# config.py

# Data Paths (Point to your specific training/testing CSVs)
DATA_PATHS = {
    'train': '/path/to/your/fold_train.csv',
    'test': '/path/to/your/fold_test.csv'
}

# Quantiles for Uncertainty Estimation
QUANTILES_FULL = [0.005, 0.05, 0.5, 0.95, 0.995]

# Feature Selection
FEATURE_COLUMNS = ['N', 'E', 'Slope20', 'DEM', 'Aspect20', ...]

```

## Module Descriptions

### models/

Contains the core logic for the Two-Part Framework:

- **binary_model.py**: Implementation of the Random Forest Classifier to distinguish between bedrock outcrops (0 m) and non-outcrops (> 0 m)
- **qrf_model.py**: Implementation of the Quantile Regression Forest (QRF) using `quantile-forest` to predict depth and uncertainty intervals
- **fusion.py**: Logic for combining the binary and regression outputs (i.e., applying the regression prediction only where the binary model predicts "Soil")

### optimization/

- **objectives.py**: Defines objective functions for hyperparameter tuning.

### visualization/

- **plots.py**: Functions to generate comprehensive model performance figures, including spatial residual maps, uncertainty calibration plots, and hyperparameter optimization histories.


### Core Scripts

- **main.py**: The entry point. Initializes the session, loads data, trains models, and triggers evaluation

- **data_utils.py**: Contains helper functions for data management, including feature subsetting, creating timestamped output directories, and performing stratified subsampling to maintain depth distributions during rapid testing.
- **metrics.py**: Defines custom performance metrics (e.g., CCC, PICP) and implements the logic to evaluate the binary, regression, and fused model components independently.
- **evaluation.py**: Calculates depth-stratified metrics (RMSE, Bias, PICP), performing threshold sensitivity analysis, and generating CSV reports for the binary, regression, and fused model components
- **prediction.py**: implements the inference (prediction) phase of the two-part DTB mapping framework. It takes trained models and applies them to new geospatial data tiles to generate final map products.
 

## Output Files

The framework generates outputs in two categories: Training/Validation outputs (from `main.py`) and Final Map Products (from `prediction.py`).

### 1. Training & Validation Outputs

These files are saved in a timestamped directory (e.g., `results/1.0000_20251223_fold4/`).

#### Root Files

| File | Description |
|------|-------------|
| `summary.txt` | High-level text report summarizing optimal hyperparameters, RMSE, MAE, and misclassification rates |
| `test_predictions.csv` | Raw predictions for every test sample (columns: y_true, y_pred_median, outcrop_proba, is_outcrop_pred) |
| `binary_model.pkl` | Trained Random Forest classifier (serialized object) |
| `qrf_model.pkl` | Trained Quantile Regression Forest (serialized object) |
| `session_output.log` | Console logs capturing the training progress (optional) |

#### comprehensive_evaluation/

Generated by `evaluation.py` to analyze model performance in depth.

| File | Description |
|------|-------------|
| `fused_binary_metrics.csv` | Metrics for the classifier only (AUC, Precision, Recall, F1) |
| `fused_regression_clean_metrics.csv` | Metrics for the QRF on correctly classified soil (ideal regression performance) |
| `fused_full_pipeline_metrics.csv` | Metrics for the final fused map (includes penalties for misclassifying soil as bedrock) |
| `performance_by_depth_class.csv` | Detailed table showing RMSE/Bias/PICP broken down by depth bins (0-2m, 2-5m, etc.) |
| `threshold_sensitivity.csv` | Data showing how Accuracy vs. Recall changes as you adjust the probability threshold (0.1–0.9) |
| `threshold_sensitivity.png` | Plot visualizing the trade-off between finding bedrock and minimizing depth error |
| `calibration_by_depth_class.png` | Bar chart showing if the 90% uncertainty intervals actually cover 90% of the data across depths |
| `fused_model_summary.png` | Large summary dashboard combining confusion matrices, residual plots, and metrics |

#### plots/

Generated by `visualization/plots.py` for publication-quality figures.

| File | Description |
|------|-------------|
| `pipeline_enhanced.png` | Master evaluation figure (Observed vs. Predicted, Error by Depth, Uncertainty Reliability) |
| `pipeline_enhanced_scatter_data.csv` | Underlying data for the scatter plot, useful for regenerating plots in other software |
| `pipeline_enhanced_by_depth_class.csv` | Underlying data for the error-by-depth bar charts |
| `pipeline_enhanced_summary_metrics.txt` | Text dump of the exact metrics displayed inside the plot |
| `binary_optimization.png` | Trace plot of the Optuna hyperparameter search for the classifier |
| `qrf_optimization.png` | Trace plot of the Optuna hyperparameter search for the regressor |
| `spatial_results.png` | Maps showing residuals (errors) plotted by X/Y coordinates to spot spatial bias |

### 2. Final Map Products

Generated by `prediction.py` when applying the model to new data tiles. All outputs are GeoTIFF format (int16).

| Directory/File Type | Description |
|---------------------|-------------|
| `RF_Probability/*.tif` | Probability (0–100%) that a pixel is Bedrock/Outcrop |
| `QRF_50_Median/*.tif` | Median predicted sediment depth (primary map) |
| `QRF_05/*.tif` | Conservative lower bound (5th percentile depth) |
| `QRF_95/*.tif` | Conservative upper bound (95th percentile depth) |
| `Fused_2m/*.tif` | **Final Product** - Combines RF and QRF. Pixels with high rock probability are set to 0 |
| `Fused_10m/*.tif` | Resampled version of Fused_2m at 10m resolution for regional analysis |



## License


## Citation


## Contact


