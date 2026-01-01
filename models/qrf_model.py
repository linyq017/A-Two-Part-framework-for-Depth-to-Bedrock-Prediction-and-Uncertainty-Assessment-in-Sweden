"""QRF regressor training."""
import joblib
import json
from pathlib import Path
from quantile_forest import RandomForestQuantileRegressor
import optuna


def train_qrf_model(X, y_train, output_dir, quantiles=None, n_trials=20):
    """Train QRF with Optuna."""
    from optimization.objectives import objective_qrf
    
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    mask_nonzero = y_train > 0
    X_nonzero = X[mask_nonzero]
    y_nonzero = y_train[mask_nonzero]
    
    print(f"Non-zero depths: {len(y_nonzero):,}")
    print(f"Range: {y_nonzero.min():.2f}m - {y_nonzero.max():.2f}m")
    
    # Optuna
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    study.optimize(lambda t: objective_qrf(t, X_nonzero, y_nonzero), n_trials=n_trials, show_progress_bar=True)
    
    # Extract best params
    best_params = study.best_params.copy()
    best_params.update({'random_state': 42, 'n_jobs': -1})
    
    print(f"\nBest params: {best_params}")
    print(f"Best score: {study.best_value:.6f}")
    
    # Train final model
    model = RandomForestQuantileRegressor(**best_params)
    model.fit(X_nonzero, y_nonzero)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path / 'qrf_model.pkl')
    joblib.dump(quantiles, output_path / 'quantiles.pkl')
    with open(output_path / 'qrf_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    study.trials_dataframe().to_csv(output_path / 'qrf_optuna_study.csv', index=False)
    
    print("✓ QRF model saved")
    return model, quantiles, study