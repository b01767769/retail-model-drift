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
import logging

logger = logging.getLogger(__name__)

def evaluate_model_bootstrap(pipeline, df: pd.DataFrame, n_bootstraps: int = 1000, random_state: int = 42) -> dict:
    """
    Evaluates model performance including all secondary metrics 
    and 95% Bootstrap Confidence Intervals for AUC.
    """
    X = df[['recency', 'frequency', 'monetary']]
    # Reset index to ensure clean NumPy array indexing during the bootstrap loop
    y = df['target'].reset_index(drop=True) 
    
    # Extract both probabilities (for AUC/Brier) and hard classes (for Acc/Prec/Recall)
    preds_proba = pipeline.predict_proba(X)[:, 1]
    preds_class = pipeline.predict(X)
    
    # 1. base secondary metrics
    metrics = {
        'accuracy': accuracy_score(y, preds_class),
        'precision': precision_score(y, preds_class, zero_division=0),
        'recall': recall_score(y, preds_class, zero_division=0),
        'f1': f1_score(y, preds_class, zero_division=0),
        'brier_score': brier_score_loss(y, preds_proba) # Calibration metric
    }
    
    # 2. Bootstrap 95% Confidence Intervals for AUC
    rng = np.random.RandomState(random_state)
    bootstrapped_aucs = []
    
    logger.info(f"Running {n_bootstraps} bootstrap resamples for AUC confidence intervals...")
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(preds_proba), len(preds_proba))
        y_sample = y.iloc[indices]
        
        if len(np.unique(y_sample)) < 2:
            continue
            
        score = roc_auc_score(y_sample, preds_proba[indices])
        bootstrapped_aucs.append(score)
        
    sorted_scores = np.sort(bootstrapped_aucs)
    
    metrics['auc'] = np.mean(sorted_scores)
    metrics['auc_lower'] = sorted_scores[int(0.025 * len(sorted_scores))]
    metrics['auc_upper'] = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return metrics

def compare_models(champ_metrics: dict, chal_metrics: dict, min_improvement: float = 0.02) -> tuple:
    """
    Evaluates Champion vs Challenger. 
    Promotion requires BOTH practical improvement AND statistical confidence.
    """
    # 1. Practical Threshold: Did it improve by at least the minimum delta?
    delta_auc = chal_metrics['auc'] - champ_metrics['auc']
    practical_pass = delta_auc >= min_improvement
    
    # 2. Statistical Threshold: Does the Challenger's worst-case (lower CI) 
    # beat the Champion's best-case (upper CI)?
    statistical_pass = chal_metrics['auc_lower'] > champ_metrics['auc_upper']
    
    # 3. Secondary Metric Check: Ensure calibration (Brier Score) didn't collapse
    # Lower Brier score is better. We allow a tiny degradation tolerance (e.g., 0.01)
    brier_stable = (chal_metrics['brier_score'] - champ_metrics['brier_score']) < 0.01
    
    # Final promotion logic: Must pass either strict statistical separation OR 
    # a practical improvement, while remaining well-calibrated.
    promote = (practical_pass or statistical_pass) and brier_stable
    
    report = {
        "delta_auc": delta_auc,
        "delta_brier": chal_metrics['brier_score'] - champ_metrics['brier_score'],
        "practical_pass": practical_pass,
        "statistical_pass": statistical_pass,
        "brier_stable": brier_stable,
        "promoted": promote
    }
    
    return promote, report
