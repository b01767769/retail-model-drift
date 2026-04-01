import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

def save_feature_importance_artifact(
    model_pipeline: Pipeline, 
    feature_names: List[str], 
    output_path: str = "artifacts/feature_importance.png"
) -> None:
    """
    Extracts feature importance from the Random Forest model and saves a visualization.
    Prevents memory leaks by explicitly closing Matplotlib figures.
    """
    try:
        rf_model = model_pipeline.named_steps.get('rf')
        if rf_model is None:
            raise ValueError("Pipeline does not contain a step named 'rf'.")
            
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
        # Force garbage collection of the figure to prevent RAM bloat in loops
        plt.close('all') 


def train_baseline_rf(
    training_data: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'target',
    random_seed: int = 42
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Trains the baseline Random Forest model strictly adhering to Chapter 3.6 hyperparameters.
    Implements stratified splitting and secure scaling pipelines.
    """
    if feature_cols is None:
        feature_cols = ['recency', 'frequency', 'monetary']
        
    if training_data.empty:
        raise ValueError("Cannot train model: Input DataFrame is empty.")
        
    missing_cols = [col for col in feature_cols + [target_col] if col not in training_data.columns]
    if missing_cols:
        raise KeyError(f"Training data is missing required columns: {missing_cols}")
        
    # Isolate features and target
    feature_matrix = training_data[feature_cols]
    target_vector = training_data[target_col]
    
    # Stratified Data Split
    # Ensures the rare positive class (high spenders) is evenly distributed
    X_train, X_val, y_train, y_val = train_test_split(
        feature_matrix, 
        target_vector, 
        test_size=0.2, 
        stratify=target_vector, 
        random_state=random_seed
    )
    
    # Pipeline Construction
    # Scaling is embedded to prevent data leakage during cross-validation/evaluation
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_seed,
            n_jobs=-1
        ))
    ])
    
   
    logger.info("Fitting Random Forest pipeline...")
    rf_pipeline.fit(X_train, y_train)
    
    # Extract validation probabilities for the positive class
    val_probabilities = rf_pipeline.predict_proba(X_val)[:, 1]
    
    validation_metrics = {
        'val_auc': float(roc_auc_score(y_val, val_probabilities))
    }
    
   
    save_feature_importance_artifact(rf_pipeline, feature_cols)
    
    return rf_pipeline, validation_metrics


def train_challenger(
    cumulative_data: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'target',
    random_seed: int = 42
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Wraps baseline training logic specifically for challenger models.
    Designed to accept cumulative data arrays concatenated by the orchestrator.
    """
    logger.info(f"Training Challenger model on cumulative dataset of size {len(cumulative_data)}")
    return train_baseline_rf(
        training_data=cumulative_data,
        feature_cols=feature_cols,
        target_col=target_col,
        random_seed=random_seed
    )
