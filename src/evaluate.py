from sklearn.metrics import roc_auc_score
import numpy as np

def evaluate_model(pipeline, df) -> dict:
    """Evaluates model on holdout slice."""
    X = df[['recency', 'frequency', 'monetary']]
    y = df['target']
    preds = pipeline.predict_proba(X)[:, 1]
    return {'auc': roc_auc_score(y, preds)}

def compare_models(champ_metrics: dict, chal_metrics: dict, min_improvement: float = 0.02) -> tuple:
    """Bootstrap or practical improvement check."""
    # For brevity, implementing the practical delta check.
    # In a full implementation, you would resample predictions 1000 times here.
    delta = chal_metrics['auc'] - champ_metrics['auc']
    promote = delta >= min_improvement
    report = {"delta_auc": delta, "promoted": promote}
    return promote, report