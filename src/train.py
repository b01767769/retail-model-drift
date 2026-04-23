import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss

logger = logging.getLogger(__name__)

def save_feature_importance_artifact(
    rf_model: RandomForestClassifier, 
    feature_names: List[str], 
    output_path: str = "artifacts/feature_importance.png"
) -> None:
    """
    Extracts feature importance from the Random Forest model and saves a visualization.
    Prevents memory leaks by explicitly closing Matplotlib figures.
    """
    try:
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 4))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='#1f77b4')
        plt.title("Random Forest Feature Importance (Gini)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        
    except Exception as e:
        logger.warning(f"Failed to generate feature importance plot: {e}")
    finally:
        plt.close('all') 


def train_model(
    training_data: pd.DataFrame, 
    config: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'target',
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Trains the Random Forest model strictly adhering to injected hyperparameters.
    Implements stratified splitting. Expects data to be pre-scaled by RFMFeatureEngineer.
    Generates all MLflow-required metrics.
    """
    if feature_cols is None:
        feature_cols = ['recency_scaled', 'frequency_scaled', 'monetary_log_scaled']
        
    if training_data.empty:
        raise ValueError("Cannot train model: Input DataFrame is empty.")
        
    missing_cols = [col for col in feature_cols + [target_col] if col not in training_data.columns]
    if missing_cols:
        raise KeyError(f"Training data is missing required columns: {missing_cols}")
        
    feature_matrix = training_data[feature_cols]
    target_vector = training_data[target_col]
    
    random_seed = config.get('random_seed', 42)
    
    X_train, X_val, y_train, y_val = train_test_split(
        feature_matrix, 
        target_vector, 
        test_size=0.2, 
        stratify=target_vector, 
        random_state=random_seed
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 200),
        max_depth=config.get('max_depth', None),
        min_samples_leaf=config.get('min_samples_leaf', 5),
        random_state=random_seed,
        n_jobs=-1
    )
    
    logger.info(f"Fitting Random Forest model with shape {X_train.shape}...")
    rf_model.fit(X_train, y_train)
    
    val_probabilities = rf_model.predict_proba(X_val)[:, 1]
    val_predictions = rf_model.predict(X_val)
    
    validation_metrics = {
        'auc': float(roc_auc_score(y_val, val_probabilities)),
        'accuracy': float(accuracy_score(y_val, val_predictions)),
        'precision': float(precision_score(y_val, val_predictions, zero_division=0)),
        'recall': float(recall_score(y_val, val_predictions, zero_division=0)),
        'f1': float(f1_score(y_val, val_predictions, zero_division=0)),
        'val_loss': float(log_loss(y_val, val_probabilities))
    }
    
    # Generate Artifacts
    save_feature_importance_artifact(rf_model, feature_cols)
    
    # Save Model Artifact to disk for MLflow pickup
    model_path = Path("artifacts/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, model_path)
    
    return rf_model, validation_metrics


def train_baseline_rf(
    training_data: pd.DataFrame, 
    config: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Specific wrapper for Baseline Training."""
    logger.info("Starting Baseline Training.")
    return train_model(training_data, config, feature_cols)


def train_challenger(
    cumulative_data: pd.DataFrame, 
    config: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Specific wrapper for Challenger Training."""
    logger.info(f"Starting Challenger Training on {len(cumulative_data)} records.")
    return train_model(cumulative_data, config, feature_cols)