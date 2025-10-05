"""
Model Training and Evaluation Script

This script:
1. Loads data using DataLoader
2. Preprocesses features (handles missing values)
3. Trains multiple models with class imbalance handling
4. Evaluates against baseline model
5. Generates visualizations and metrics
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from data_etl import SQLiteDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create plots directory
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def preprocess_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data: handle missing values and create additional features.
    
    Args:
        df: Raw dataframe
        target: Name of target column
        
    Returns:
        Tuple of (X features, y target)
    """
    logger.info("Preprocessing data...")
    
    # Separate target and features
    y = df[target]
    X = df.drop(columns=[target, 'model_pred'])  # Drop target and existing model prediction
    
    # Drop identifier columns (not useful for prediction)
    id_cols = ['retailer_id', 'transaction_id', 'item_id']
    X = X.drop(columns=[col for col in id_cols if col in X.columns])
    
    # Create binary flag for first-time returners (where history features are NULL)
    X['is_first_return'] = X['days_since_last_return'].isnull().astype(int)
    
    # Log missing values before imputation
    missing_before = X.isnull().sum()
    logger.info(f"Missing values before imputation:\n{missing_before[missing_before > 0]}")
    
    # Impute missing values with median (for customers with no history)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    logger.info(f"Features shape: {X_imputed.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X_imputed, y


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, 
                   model_name: str) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary: 0 or 1)
        y_prob: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    
    logger.info(f"\n{model_name} Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = PLOTS_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix: {filename}")


def plot_roc_curves(results: Dict[str, Dict], y_test: np.ndarray):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        auc = result['metrics']['roc_auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = PLOTS_DIR / 'roc_curves_comparison.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves: {filename}")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Adjust top_n if we have fewer features
        top_n = min(top_n, len(importances))
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filename = PLOTS_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Saved feature importance: {filename}")


def run_training(conn: sqlite3.Connection):
    """
    Main training pipeline.
    
    Args:
        conn: Database connection
    """
    logger.info("="*80)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Define features to use
    features = [
        'return_rate', 'num_pur', 'num_ret', 'amt_ret', 'amount',
        'num_true_ret', 'return_to_purchase_ratio', 'days_since_last_return',
        'item_repeat_return_ratio', 'return_acceleration', 'avg_return_amount',
        'high_value_return_flag'
    ]
    
    # Load data using DataLoader
    loader = SQLiteDataLoader(conn)
    df = loader.load(target='return_fraud', features=features + ['model_pred'])
    
    # Preprocess data
    X, y = preprocess_data(df, target='return_fraud')
    
    # Train-test split (stratified to maintain fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Training fraud rate: {y_train.mean():.2%}")
    logger.info(f"Test fraud rate: {y_test.mean():.2%}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate baseline model (existing model_pred)
    logger.info("\n" + "="*80)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("="*80)
    
    # Get baseline probabilities and apply threshold
    baseline_prob = df.loc[X_test.index, 'model_pred'].values
    baseline_pred = (baseline_prob >= 0.5).astype(int)  # Apply 0.5 threshold
    
    baseline_metrics = evaluate_model(y_test, baseline_pred, baseline_prob, "Baseline Model")
    plot_confusion_matrix(y_test, baseline_pred, "Baseline Model")
    
    # Store results
    results = {
        'Baseline Model': {
            'y_pred': baseline_pred,
            'y_prob': baseline_prob,
            'metrics': baseline_metrics
        }
    }
    
    # Train new models
    logger.info("\n" + "="*80)
    logger.info("TRAINING NEW MODELS")
    logger.info("="*80)
    
    # 1. Logistic Regression
    logger.info("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_pred, lr_prob, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'y_pred': lr_pred,
        'y_prob': lr_prob,
        'metrics': lr_metrics
    }
    
    # 2. Random Forest
    logger.info("\n--- Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_pred, rf_prob, "Random Forest")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")
    plot_feature_importance(rf_model, X.columns.tolist(), "Random Forest")
    
    results['Random Forest'] = {
        'model': rf_model,
        'y_pred': rf_pred,
        'y_prob': rf_prob,
        'metrics': rf_metrics
    }
    
    # 3. XGBoost
    logger.info("\n--- XGBoost ---")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_prob, "XGBoost")
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
    plot_feature_importance(xgb_model, X.columns.tolist(), "XGBoost")
    
    results['XGBoost'] = {
        'model': xgb_model,
        'y_pred': xgb_pred,
        'y_prob': xgb_prob,
        'metrics': xgb_metrics
    }
    
    # K-Fold Cross-Validation
    logger.info("\n" + "="*80)
    logger.info("K-FOLD CROSS-VALIDATION (5 folds)")
    logger.info("="*80)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for model_name, result in results.items():
        if 'model' in result:
            cv_scores = cross_val_score(
                result['model'], X, y, cv=skf, scoring='roc_auc', n_jobs=-1
            )
            logger.info(f"{model_name} CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Plot ROC curves
    plot_roc_curves(results, y_test)
    
    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*80)
    
    metrics_df = pd.DataFrame([r['metrics'] for r in results.values()])
    logger.info("\n" + metrics_df.to_string(index=False))
    
    # Determine best model
    best_model_name = metrics_df.loc[metrics_df['f1'].idxmax(), 'model']
    logger.info(f"\n🏆 Best Model (by F1 Score): {best_model_name}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    # For standalone execution
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "sample.db"
    
    conn = sqlite3.connect(db_path)
    run_training(conn)
    conn.close()