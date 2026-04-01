import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

logger = logging.getLogger(__name__)

def compute_residual_stability(true_labels: np.ndarray, predicted_probs: np.ndarray) -> float:
    """
    Calculates the standard deviation of prediction errors across deciles.
    Acts as a model-centric diagnostic to verify calibration stability.
    """
    if len(true_labels) != len(predicted_probs):
        raise ValueError("Labels and probabilities must have the same length.")
        
    # Construct minimal DataFrame for vectorized binning and aggregation
    eval_df = pd.DataFrame({
        'actual': true_labels,
        'predicted': predicted_probs
    })
    eval_df['residual_error'] = eval_df['actual'] - eval_df['predicted']
    
    try:
        # Group predictions into 10 decile bins (Quantile Binning)
        eval_df['prob_decile'] = pd.qcut(eval_df['predicted'], q=10, duplicates='drop')
        decile_mean_errors = eval_df.groupby('prob_decile', observed=False)['residual_error'].mean()
        return float(decile_mean_errors.std())
    except ValueError:
        # Failsafe: Triggered if the model predicts the exact same probability for all samples
        logger.warning("Residual stability calculation failed (likely identical predictions). Returning 0.0")
        return 0.0

def evaluate_model_bootstrap(
    model_pipeline, 
    eval_dataframe: pd.DataFrame, 
    feature_cols: List[str] = None,
    target_col: str = 'target',
    n_bootstraps: int = 1000, 
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Evaluates model performance including all secondary metrics 
    and 95% Bootstrap Confidence Intervals for AUC.
    """
    if feature_cols is None:
        feature_cols = ['recency', 'frequency', 'monetary']

    if eval_dataframe.empty:
        raise ValueError("Evaluation DataFrame is empty.")
    
    missing_cols = [col for col in feature_cols + [target_col] if col not in eval_dataframe.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in evaluation data: {missing_cols}")

    feature_matrix = eval_dataframe[feature_cols]
    target_vector = eval_dataframe[target_col].reset_index(drop=True)
    
    # 2. Extract both probabilities (for AUC/Brier) and hard classes (for Acc/Prec/Recall)
    pred_probabilities = model_pipeline.predict_proba(feature_matrix)[:, 1]
    pred_classes = model_pipeline.predict(feature_matrix)
    
    # 3. base secondary metrics
    performance_metrics = {
        'accuracy': float(accuracy_score(target_vector, pred_classes)),
        'precision': float(precision_score(target_vector, pred_classes, zero_division=0)),
        'recall': float(recall_score(target_vector, pred_classes, zero_division=0)),
        'f1': float(f1_score(target_vector, pred_classes, zero_division=0)),
        'brier_score': float(brier_score_loss(target_vector, pred_probabilities)),
        'residual_stability': compute_residual_stability(target_vector.to_numpy(), pred_probabilities)
    }
    
    # 4. Bootstrap 95% Confidence Intervals for AUC
    target_array = target_vector.to_numpy()
    num_samples = len(pred_probabilities)
    rng = np.random.RandomState(random_seed)
    bootstrapped_auc_scores = []
    
    logger.info(f"Running {n_bootstraps} bootstrap resamples for AUC confidence intervals...")
    
    for _ in range(n_bootstraps):
        sample_indices = rng.randint(0, num_samples, num_samples)
        sampled_targets = target_array[sample_indices]
        
        # Failsafe: Ensure both classes are present in the bootstrap sample
        if len(np.unique(sampled_targets)) < 2:
            continue
            
        sampled_probs = pred_probabilities[sample_indices]
        iter_score = roc_auc_score(sampled_targets, sampled_probs)
        bootstrapped_auc_scores.append(iter_score)
        
    if not bootstrapped_auc_scores:
        raise ValueError("All bootstrap iterations failed due to single-class samples.")
        
    sorted_auc_array = np.sort(bootstrapped_auc_scores)
    
    # 5. Append statistical bounds
    performance_metrics['auc'] = float(np.mean(sorted_auc_array))
    performance_metrics['auc_lower'] = float(sorted_auc_array[int(0.025 * len(sorted_auc_array))])
    performance_metrics['auc_upper'] = float(sorted_auc_array[int(0.975 * len(sorted_auc_array))])
    
    return performance_metrics

def compare_models(
    champion_metrics: Dict[str, float], 
    challenger_metrics: Dict[str, float], 
    min_auc_improvement: float = 0.02,
    max_brier_degradation: float = 0.01
) -> Tuple[bool, Dict[str, float]]:
    """
    Evaluates Champion vs Challenger models for promotion readiness. 
    Promotion requires either a practical improvement OR strict statistical separation,
    alongside stable calibration.
    """
    # Did it improve by at least the minimum delta?
    auc_delta = challenger_metrics['auc'] - champion_metrics['auc']
    meets_practical_bar = auc_delta >= min_auc_improvement
    
    # 2. Statistical Threshold Check: Does Challenger's worst-case (lower CI) beat Champion's best-case (upper CI)?
    meets_statistical_bar = challenger_metrics['auc_lower'] > champion_metrics['auc_upper']
    
    # 3. Calibration Safeguard: Ensure Brier Score didn't collapse (Lower is better)
    brier_delta = challenger_metrics['brier_score'] - champion_metrics['brier_score']
    is_calibration_stable = brier_delta < max_brier_degradation
    
    is_promoted = (meets_practical_bar or meets_statistical_bar) and is_calibration_stable
    
    promotion_report = {
        "delta_auc": float(auc_delta),
        "delta_brier": float(brier_delta),
        "practical_pass": bool(meets_practical_bar),
        "statistical_pass": bool(meets_statistical_bar),
        "brier_stable": bool(is_calibration_stable),
        "promoted": bool(is_promoted)
    }
    
    return is_promoted, promotion_report
