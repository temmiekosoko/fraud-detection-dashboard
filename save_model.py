"""
Save the trained Random Forest model for use in Streamlit app.
Run this after creating the database with create_dataset.py

Usage: python save_model.py
"""

import pickle
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from data_etl import SQLiteDataLoader

# Configuration
import sys
DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample.db"  # Uses sample.db by default
MODEL_PATH = "models/random_forest_model.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"
RANDOM_STATE = 42

# Create models directory
Path("models").mkdir(exist_ok=True)

def preprocess_data(df, target):
    """Preprocess exactly as done in train_model.py"""
    # Separate target and features
    y = df[target]
    X = df.drop(columns=[target, 'model_pred'])
    
    # Drop identifier columns
    id_cols = ['retailer_id', 'transaction_id', 'item_id']
    X = X.drop(columns=[col for col in id_cols if col in X.columns])
    
    # Create binary flag for first-time returners
    X['is_first_return'] = X['days_since_last_return'].isnull().astype(int)
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_imputed, y, imputer

def train_and_save_model():
    """Train Random Forest model and save it."""
    
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    # Define features (same as train_model.py)
    features = [
        'return_rate', 'num_pur', 'num_ret', 'amt_ret', 'amount',
        'num_true_ret', 'return_to_purchase_ratio', 'days_since_last_return',
        'item_repeat_return_ratio', 'return_acceleration', 'avg_return_amount',
        'high_value_return_flag'
    ]
    
    # Load data
    print("Loading data...")
    loader = SQLiteDataLoader(conn)
    df = loader.load(target='return_fraud', features=features + ['model_pred'])
    
    # Preprocess data (adds is_first_return feature)
    print("Preprocessing data...")
    X, y, imputer = preprocess_data(df, target='return_fraud')
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training fraud rate: {y_train.mean():.2%}")
    
    # Train Random Forest (same hyperparameters as train_model.py)
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate on test set
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, rf_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, rf_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, rf_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, rf_prob):.4f}")
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save feature info and imputer
    feature_info = {
        'features': features,  # Original features from SQL
        'feature_names': list(X.columns),  # After preprocessing (includes is_first_return)
        'imputer': imputer
    }
    
    print(f"Saving feature info to {FEATURE_INFO_PATH}...")
    with open(FEATURE_INFO_PATH, 'wb') as f:
        pickle.dump(feature_info, f)
    
    conn.close()
    print("\n✅ Model training and saving complete!")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Feature info: {FEATURE_INFO_PATH}")

if __name__ == "__main__":
    train_and_save_model()