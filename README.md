

## Overview
 
This repository contains the implementation of a Two-Part (Hurdle) Machine Learning Framework designed to map Depth to Bedrock (DTB). Standard regression models often struggle with "structural zeros" (bedrock outcrops) common in post-glacial terrains like Sweden, leading to biased predictions and unreliable uncertainty estimates.

Our solution explicitly decouples the modeling process into two stages:

Classification: Identifying bedrock outcrops vs. sediments using a Binary Random Forest.

Regression: Predicting depth to bedrock (DTB) using a Quantile Regression Forest (QRF).

By integrating these components, this framework outperformed a national IDW interpolation-based model and a global DTB map.


Key Features

Hurdle Model Architecture: Handles zero-inflated data by separating the probability of occurrence (bedrock) from the continuous value (depth).

Uncertainty Quantification: Leverages Quantile Regression Forests (QRF) to generate prediction intervals (PIs) and uncertainty maps.

Depth-Stratified Evaluation: Includes scripts for evaluating model performance across distinct depth zones (0–2m, 2–10m, >30m), revealing critical insights often masked by global metrics.

QuadMap Visualization: Includes the QuadMap tool (v1.0.0), a variable-resolution visualization method that aggregates uncertain regions to prevent false precision.

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
└── prediction.py            # Inference script for generating final maps
```
 

[Include your license information here]
