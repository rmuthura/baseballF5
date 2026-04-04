"""
Model evaluation for F5 runs prediction.

Computes comprehensive metrics including:
- MAE, RMSE, R²
- Calibration by bucket
- Over/under accuracy for betting lines
- Poisson probability distributions
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from f5_model.utils.statcast_pull import get_processed_data_dir
from f5_model.model.train import (
    load_training_data, prepare_features, time_series_split, get_models_dir
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute standard regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute calibration by prediction buckets.

    For each predicted range, compute the actual mean.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        DataFrame with calibration results
    """
    df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})

    # Define buckets
    bins = [0, 1, 2, 3, 4, 5, 100]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']

    df['bucket'] = pd.cut(df['predicted'], bins=bins, labels=labels, right=False)

    calibration = df.groupby('bucket', observed=True).agg({
        'actual': ['mean', 'std', 'count'],
        'predicted': 'mean'
    }).round(3)

    calibration.columns = ['actual_mean', 'actual_std', 'count', 'predicted_mean']

    return calibration


def compute_over_under_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lines: list = [0.5, 1.5, 2.5, 3.5, 4.5]
) -> Dict:
    """
    Compute over/under accuracy for common betting lines.

    Args:
        y_true: Actual values
        y_pred: Predicted values (lambda for Poisson)
        lines: Betting lines to evaluate

    Returns:
        Dictionary with accuracy for each line
    """
    results = {}

    for line in lines:
        # Predicted probability of OVER the line
        # P(X > line) = 1 - P(X <= line) = 1 - CDF(floor(line))
        p_over = 1 - poisson.cdf(int(line), y_pred)

        # Model predicts over if p_over > 0.5
        pred_over = p_over > 0.5

        # Actual over
        actual_over = y_true > line

        # Accuracy
        correct = pred_over == actual_over
        accuracy = correct.mean()

        # Also compute by confidence
        high_conf = (p_over > 0.6) | (p_over < 0.4)
        if high_conf.sum() > 0:
            high_conf_accuracy = correct[high_conf].mean()
        else:
            high_conf_accuracy = np.nan

        results[f'over_{line}'] = {
            'accuracy': accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'n_samples': len(y_true),
            'n_high_conf': high_conf.sum(),
            'actual_over_rate': actual_over.mean()
        }

    return results


def compute_poisson_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Poisson log-likelihood of predictions.

    Args:
        y_true: Actual counts
        y_pred: Predicted lambda values

    Returns:
        Mean log-likelihood
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 0.01, None)

    log_lik = poisson.logpmf(y_true.astype(int), y_pred)

    return log_lik.mean()


def predict_distribution(lambda_val: float, max_runs: int = 10) -> Dict:
    """
    Get the full Poisson probability distribution for a prediction.

    Args:
        lambda_val: Predicted lambda (mean runs)
        max_runs: Maximum runs to include

    Returns:
        Dictionary with probabilities and over/under odds
    """
    probs = {k: poisson.pmf(k, lambda_val) for k in range(max_runs + 1)}

    over_under = {
        f'over_{line}': 1 - poisson.cdf(int(line), lambda_val)
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]
    }

    return {
        'lambda': lambda_val,
        'probabilities': probs,
        'over_under': over_under
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from the model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances sorted by importance
    """
    importance = model.feature_importances_

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return df


def run_evaluation() -> Dict:
    """
    Run full model evaluation and print results.

    Returns:
        Dictionary with all evaluation results
    """
    logger.info("=" * 60)
    logger.info("F5 Runs Prediction Model Evaluation")
    logger.info("=" * 60)

    # Load model and data
    logger.info("\n1. Loading model and data...")
    model_path = get_models_dir() / "f5_runs_model.pkl"
    model = joblib.load(model_path)

    df = load_training_data()
    train_df, test_df = time_series_split(df, test_size=0.2)

    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Make predictions
    logger.info("\n2. Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {}

    # Regression metrics
    logger.info("\n3. Computing regression metrics...")
    train_metrics = compute_regression_metrics(y_train.values, y_pred_train)
    test_metrics = compute_regression_metrics(y_test.values, y_pred_test)

    results['train_metrics'] = train_metrics
    results['test_metrics'] = test_metrics

    print("\n" + "=" * 60)
    print("REGRESSION METRICS")
    print("=" * 60)
    print(f"\n{'Metric':<10} {'Train':>10} {'Test':>10}")
    print("-" * 32)
    for metric in ['MAE', 'RMSE', 'R2']:
        print(f"{metric:<10} {train_metrics[metric]:>10.3f} {test_metrics[metric]:>10.3f}")

    # Calibration
    logger.info("\n4. Computing calibration...")
    calibration = compute_calibration(y_test.values, y_pred_test)
    results['calibration'] = calibration

    print("\n" + "=" * 60)
    print("CALIBRATION (Test Set)")
    print("=" * 60)
    print("\nPredicted Range | Actual Mean | Predicted Mean | Count")
    print("-" * 55)
    for bucket, row in calibration.iterrows():
        print(f"{bucket:<15} | {row['actual_mean']:>11.2f} | {row['predicted_mean']:>14.2f} | {int(row['count']):>5}")

    # Over/under accuracy
    logger.info("\n5. Computing over/under accuracy...")
    ou_results = compute_over_under_accuracy(y_test.values, y_pred_test)
    results['over_under'] = ou_results

    print("\n" + "=" * 60)
    print("OVER/UNDER ACCURACY (Test Set)")
    print("=" * 60)
    print(f"\n{'Line':<10} {'Accuracy':>10} {'High Conf':>12} {'Actual O%':>10}")
    print("-" * 45)
    for line, data in ou_results.items():
        hc = f"{data['high_conf_accuracy']:.1%}" if not np.isnan(data['high_conf_accuracy']) else "N/A"
        print(f"{line:<10} {data['accuracy']:>10.1%} {hc:>12} {data['actual_over_rate']:>10.1%}")

    # Poisson log-likelihood
    logger.info("\n6. Computing Poisson log-likelihood...")
    train_ll = compute_poisson_log_likelihood(y_train.values, y_pred_train)
    test_ll = compute_poisson_log_likelihood(y_test.values, y_pred_test)
    results['poisson_log_likelihood'] = {'train': train_ll, 'test': test_ll}

    print("\n" + "=" * 60)
    print("POISSON LOG-LIKELIHOOD")
    print("=" * 60)
    print(f"\nTrain: {train_ll:.4f}")
    print(f"Test:  {test_ll:.4f}")

    # Feature importance
    logger.info("\n7. Computing feature importance...")
    importance_df = get_feature_importance(model, feature_names)
    results['feature_importance'] = importance_df

    print("\n" + "=" * 60)
    print("TOP 15 FEATURES BY IMPORTANCE")
    print("=" * 60)
    print(f"\n{'Rank':<5} {'Feature':<35} {'Importance':>12}")
    print("-" * 55)
    for i, row in importance_df.head(15).iterrows():
        print(f"{importance_df.index.get_loc(i)+1:<5} {row['feature']:<35} {row['importance']:>12.4f}")

    # Distribution comparison
    print("\n" + "=" * 60)
    print("PREDICTED VS ACTUAL DISTRIBUTION (Test Set)")
    print("=" * 60)
    print(f"\n{'Runs':<6} {'Actual':>10} {'Predicted':>12}")
    print("-" * 30)

    actual_dist = y_test.value_counts().sort_index()
    pred_probs = np.array([poisson.pmf(k, y_pred_test).mean() for k in range(8)])

    for runs in range(8):
        actual_pct = actual_dist.get(runs, 0) / len(y_test) * 100
        pred_pct = pred_probs[runs] * 100
        print(f"{runs:<6} {actual_pct:>9.1f}% {pred_pct:>11.1f}%")

    # Save results
    results_path = get_models_dir() / "evaluation_results.pkl"
    joblib.dump(results, results_path)
    logger.info(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_evaluation()
