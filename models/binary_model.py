"""Binary outcrop classifier training."""
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import optuna


def train_binary_model(X, y_train, output_dir, n_trials=10):
    """Train binary model with Optuna."""
    from optimization.objectives import objective_binary
    
    y_binary = (y_train == 0).astype(int)
    
    print(f"Training samples: {len(y_train):,}")
    print(f"  Outcrops: {y_binary.sum():,} ({100*y_binary.mean():.1f}%)")
    
    # Optuna
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(lambda t: objective_binary(t, X, y_binary), n_trials=n_trials, show_progress_bar=True)
    
    # Extract best params
    best_params = study.best_params.copy()
    best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
    
    print(f"\nBest params: {best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    # Train final model
    model = RandomForestClassifier(**best_params)
    model.fit(X, y_binary)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path / 'binary_model.pkl')
    with open(output_path / 'binary_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    study.trials_dataframe().to_csv(output_path / 'binary_optuna_study.csv', index=False)
    
    print("✓ Binary model saved")
    return model, study