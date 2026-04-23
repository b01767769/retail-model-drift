import logging
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def compute_residual_stability(true_labels: np.ndarray, predicted_probs: np.ndarray) -> float:
    """
    Calculates the standard deviation of prediction errors across deciles.
    Acts as a model-centric diagnostic to verify calibration stability.
    """
    if len(true_labels) != len(predicted_probs):
        raise ValueError("Labels and probabilities must have the same length.")
        
    eval_df = pd.DataFrame({
        'actual': true_labels,
        'predicted': predicted_probs
    })
    eval_df['residual_error'] = eval_df['actual'] - eval_df['predicted']
    
    try:
        # Group predictions into 10 decile bins
        eval_df['prob_decile'] = pd.qcut(eval_df['predicted'], q=10, duplicates='drop')
        decile_mean_errors = eval_df.groupby('prob_decile', observed=False)['residual_error'].mean()
        return float(decile_mean_errors.std())
    except ValueError:
        logger.warning("Residual stability calculation failed. Returning 0.0")
        return 0.0

def generate_confusion_matrix_artifact(true_labels: np.ndarray, pred_classes: np.ndarray, output_path: str = "artifacts/confusion_matrix.png"):
    """Saves a visual confusion matrix to disk."""
    try:
        cm = confusion_matrix(true_labels, pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Spend", "High Spend"])
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        plt.title("Evaluation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
    except Exception as e:
        logger.warning(f"Failed to generate confusion matrix plot: {e}")
    finally:
        plt.close('all')


def evaluate_model_bootstrap(
    rf_model: RandomForestClassifier, 
    eval_dataframe: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'target',
    n_bootstraps: int = 1000, 
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Evaluates model performance including all secondary metrics, 
    confusion matrix artifact, and 95% Bootstrap CIs for AUC.
    """
    if feature_cols is None:
        feature_cols = ['recency_scaled', 'frequency_scaled', 'monetary_log_scaled']

    if eval_dataframe.empty:
        raise ValueError("Evaluation DataFrame is empty.")
    
    missing_cols = [col for col in feature_cols + [target_col] if col not in eval_dataframe.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in evaluation data: {missing_cols}")

    feature_matrix = eval_dataframe[feature_cols]
    target_vector = eval_dataframe[target_col].reset_index(drop=True)
    
    pred_probabilities = rf_model.predict_proba(feature_matrix)[:, 1]
    pred_classes = rf_model.predict(feature_matrix)
    
    # Generate requested artifact
    generate_confusion_matrix_artifact(target_vector.to_numpy(), pred_classes)
    
    performance_metrics = {
        'accuracy': float(accuracy_score(target_vector, pred_classes)),
        'precision': float(precision_score(target_vector, pred_classes, zero_division=0)),
        'recall': float(recall_score(target_vector, pred_classes, zero_division=0)),
        'f1': float(f1_score(target_vector, pred_classes, zero_division=0)),
        'brier_score': float(brier_score_loss(target_vector, pred_probabilities)),
        'residual_stability': compute_residual_stability(target_vector.to_numpy(), pred_probabilities)
    }
    
    # Bootstrap AUC
    target_array = target_vector.to_numpy()
    num_samples = len(pred_probabilities)
    rng = np.random.RandomState(random_seed)
    bootstrapped_auc_scores = []
    
    logger.info(f"Running {n_bootstraps} bootstrap resamples for AUC...")
    
    for _ in range(n_bootstraps):
        sample_indices = rng.randint(0, num_samples, num_samples)
        sampled_targets = target_array[sample_indices]
        
        if len(np.unique(sampled_targets)) < 2:
            continue
            
        sampled_probs = pred_probabilities[sample_indices]
        bootstrapped_auc_scores.append(roc_auc_score(sampled_targets, sampled_probs))
        
    if not bootstrapped_auc_scores:
        raise ValueError("Bootstrap failed due to single-class samples.")
        
    sorted_auc_array = np.sort(bootstrapped_auc_scores)
    
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
    """Evaluates Champion vs Challenger models for promotion readiness."""
    auc_delta = challenger_metrics['auc'] - champion_metrics['auc']
    meets_practical_bar = auc_delta >= min_auc_improvement
    
    meets_statistical_bar = challenger_metrics['auc_lower'] > champion_metrics['auc_upper']
    
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


def compile_evaluation_report(
    slice_number: int,
    metrics: Dict[str, float],
    run_id: str,
    psi_summary: str,
    output_path: str = "artifacts/evaluation_report.html"
) -> str:
    """
    Compiles metrics, MLflow links, and artifact references into a standard HTML report.
    """
    logger.info("Compiling standard evaluation report...")
    
    # We use relative paths in the HTML so images load if opened from the artifacts folder
    html_content = f"""
    <html>
    <head>
        <title>Slice {slice_number} Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            table {{ border-collapse: collapse; width: 50%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .container {{ display: flex; gap: 20px; }}
            img {{ max-width: 400px; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <h2>Model Evaluation Report: Slice {slice_number}</h2>
        <p><strong>MLflow Run ID:</strong> <a href="#">{run_id}</a></p>
        <p><strong>PSI Context:</strong> {psi_summary}</p>
        
        <h3>Performance Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>AUC (Mean)</td><td>{metrics.get('auc', 0):.4f}</td></tr>
            <tr><td>AUC 95% CI</td><td>[{metrics.get('auc_lower', 0):.4f}, {metrics.get('auc_upper', 0):.4f}]</td></tr>
            <tr><td>Brier Score (Calibration)</td><td>{metrics.get('brier_score', 0):.4f}</td></tr>
            <tr><td>Accuracy</td><td>{metrics.get('accuracy', 0):.4f}</td></tr>
            <tr><td>Precision / Recall</td><td>{metrics.get('precision', 0):.4f} / {metrics.get('recall', 0):.4f}</td></tr>
            <tr><td>Residual Stability</td><td>{metrics.get('residual_stability', 0):.4f}</td></tr>
        </table>

        <h3>Diagnostic Visuals</h3>
        <div class="container">
            <div>
                <h4>Feature Importance</h4>
                <img src="feature_importance.png" alt="Feature Importance Plot missing">
            </div>
            <div>
                <h4>Confusion Matrix</h4>
                <img src="confusion_matrix.png" alt="Confusion Matrix missing">
            </div>
        </div>
    </body>
    </html>
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)
        
    logger.info(f"Report saved to {output_path}")
    return output_path