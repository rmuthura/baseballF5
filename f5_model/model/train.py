"""
Model training for F5 runs prediction.

Trains an XGBoost model with Poisson objective for predicting
runs allowed by starting pitchers through 5 innings.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from f5_model.utils.statcast_pull import get_processed_data_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_models_dir() -> Path:
    """Get the models directory path."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def load_training_data() -> pd.DataFrame:
    """Load the training dataset."""
    path = get_processed_data_dir() / "training_data.parquet"
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepare features and target for training.

    Args:
        df: Training DataFrame

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature column names
    """
    # Target
    y = df['f5_runs_allowed']

    # Features - exclude identifiers and target
    exclude_cols = ['game_pk', 'game_date', 'starter', 'f5_runs_allowed']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    # Handle any remaining object columns
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes

    return X, y, feature_cols


def time_series_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically for train/test.

    Args:
        df: DataFrame with 'game_date' column
        test_size: Fraction of data to use for testing

    Returns:
        train_df, test_df
    """
    df = df.sort_values('game_date')

    split_idx = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_idx]['game_date']

    train_df = df[df['game_date'] < split_date].copy()
    test_df = df[df['game_date'] >= split_date].copy()

    logger.info(f"Train/test split at {split_date}")
    logger.info(f"  Train: {len(train_df):,} rows ({df['game_date'].min()} to {split_date})")
    logger.info(f"  Test: {len(test_df):,} rows ({split_date} to {df['game_date'].max()})")

    return train_df, test_df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None
) -> xgb.XGBRegressor:
    """
    Train XGBoost model with Poisson objective.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional, for early stopping)
        y_val: Validation target

    Returns:
        Trained XGBoost model
    """
    params = {
        'objective': 'count:poisson',
        'eval_metric': 'mae',
        'max_depth': 4,
        'learning_rate': 0.01,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'n_estimators': 1500,
        'random_state': 42,
        'n_jobs': -1,
    }

    model = xgb.XGBRegressor(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
    else:
        model.fit(X_train, y_train, verbose=100)

    return model


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> dict:
    """
    Perform time series cross-validation.

    Args:
        X: Features
        y: Target
        n_splits: Number of CV splits

    Returns:
        Dictionary with CV results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv = X.iloc[train_idx]
        y_train_cv = y.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]
        y_val_cv = y.iloc[val_idx]

        # Train model for this fold
        model = train_model(X_train_cv, y_train_cv)

        # Predict and evaluate
        y_pred = model.predict(X_val_cv)
        mae = np.mean(np.abs(y_val_cv - y_pred))

        cv_scores.append(mae)
        logger.info(f"  Fold {fold + 1}: MAE = {mae:.3f}")

    return {
        'cv_mae_mean': np.mean(cv_scores),
        'cv_mae_std': np.std(cv_scores),
        'cv_scores': cv_scores
    }


def run_training() -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Main training function.

    Returns:
        model: Trained XGBoost model
        train_df: Training data
        test_df: Test data
    """
    logger.info("=" * 60)
    logger.info("F5 Runs Prediction Model Training")
    logger.info("=" * 60)

    # Load data
    logger.info("\n1. Loading training data...")
    df = load_training_data()
    logger.info(f"   Loaded {len(df):,} rows")

    # Split train/test
    logger.info("\n2. Splitting train/test (80/20 chronological)...")
    train_df, test_df = time_series_split(df, test_size=0.2)

    # Prepare features
    logger.info("\n3. Preparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    logger.info(f"   Features: {len(feature_names)}")
    logger.info(f"   Train samples: {len(X_train):,}")
    logger.info(f"   Test samples: {len(X_test):,}")

    # Handle missing values
    logger.info("\n4. Handling missing values...")
    # XGBoost handles NaN natively, but let's check
    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()
    logger.info(f"   Train missing values: {train_missing:,}")
    logger.info(f"   Test missing values: {test_missing:,}")

    # Cross-validation
    logger.info("\n5. Running cross-validation...")
    cv_results = cross_validate(X_train, y_train, n_splits=5)
    logger.info(f"   CV MAE: {cv_results['cv_mae_mean']:.3f} (+/- {cv_results['cv_mae_std']:.3f})")

    # Train final model
    logger.info("\n6. Training final model...")
    model = train_model(X_train, y_train, X_test, y_test)

    # Save model
    logger.info("\n7. Saving model...")
    model_path = get_models_dir() / "f5_runs_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"   Saved to {model_path}")

    # Save feature names
    feature_path = get_models_dir() / "feature_names.txt"
    with open(feature_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    logger.info(f"   Feature names saved to {feature_path}")

    return model, train_df, test_df


if __name__ == "__main__":
    model, train_df, test_df = run_training()
