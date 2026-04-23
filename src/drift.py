import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)

def extract_quantile_bins(
    baseline_data: pd.DataFrame, 
    feature_cols: List[str], 
    num_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Calculates fixed baseline bins for PSI computation based on empirical quantiles.
    Safeguards the outer bounds to capture future outliers.
    """
    if baseline_data.empty:
        raise ValueError("Cannot extract bins from an empty baseline DataFrame.")
        
    bin_dictionary = {}
    
    for feature in feature_cols:
        if feature not in baseline_data.columns:
            raise KeyError(f"Feature '{feature}' not found in baseline data.")
            
        try:
            _, bin_edges = pd.qcut(baseline_data[feature], q=num_bins, retbins=True, duplicates='drop')
            
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            bin_dictionary[feature] = bin_edges
            
        except Exception as e:
            logger.error(f"Failed to generate quantile bins for {feature}: {e}")
            raise
            
    return bin_dictionary


def compute_psi_report(
    baseline_bins: Dict[str, np.ndarray], 
    eval_data: pd.DataFrame, 
    min_customers: int = 50, 
    min_bin_count: int = 5,
    epsilon: float = 1e-6
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Computes PSI, generates histogram counts, and returns a formatted DataFrame 
    for artifact reporting.
    """
    if eval_data.empty:
        raise ValueError("Cannot compute PSI on an empty evaluation DataFrame.")
        
    report = {
        "is_reliable": True, 
        "features": {},
        "diagnostics": {} # Added to store counts and quantiles
    }
    
    n_eval_samples = len(eval_data)
    
    if n_eval_samples < min_customers:
        report["is_reliable"] = False
        logger.warning(f"Slice size ({n_eval_samples}) below minimum threshold ({min_customers}). PSI flagged as unreliable.")
        
    report_rows = []
        
    for feature, edges in baseline_bins.items():
        if feature not in eval_data.columns:
            raise KeyError(f"Feature '{feature}' missing from evaluation data.")
            
        n_bins_actual = len(edges) - 1
        expected_proportions = np.ones(n_bins_actual) / n_bins_actual
        
        observed_counts, _ = np.histogram(eval_data[feature], bins=edges)
        
        report["diagnostics"][feature] = {
            "decile_boundaries": [float(x) for x in edges[1:-1]], # Exclude infs
            "histogram_counts": [int(x) for x in observed_counts]
        }
        
        if np.any(observed_counts < min_bin_count):
            report["is_reliable"] = False
            logger.debug(f"Feature '{feature}' has bin counts below minimum threshold ({min_bin_count}).")
            
        observed_proportions = observed_counts / n_eval_samples
        
        expected_safe = np.where(expected_proportions == 0, epsilon, expected_proportions)
        observed_safe = np.where(observed_proportions == 0, epsilon, observed_proportions)
        
        psi_array = (observed_safe - expected_safe) * np.log(observed_safe / expected_safe)
        total_psi = float(np.sum(psi_array))
        
        report["features"][feature] = total_psi
        
        for i in range(n_bins_actual):
            report_rows.append({
                "Feature": feature,
                "Bin_Index": i,
                "Expected_Prop": expected_proportions[i],
                "Observed_Count": observed_counts[i],
                "Observed_Prop": observed_proportions[i],
                "PSI_Contribution": psi_array[i]
            })
            
    psi_df = pd.DataFrame(report_rows)
    return report, psi_df


def generate_drift_artifacts(
    psi_report: Dict[str, Any], 
    psi_df: pd.DataFrame, 
    output_dir: str = "artifacts/"
) -> Dict[str, str]:
    """Saves the PSI computations to disk for MLflow to pick up."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    csv_path = f"{output_dir}psi_report.csv"
    json_path = f"{output_dir}drift_diagnostics.json"
    
    psi_df.to_csv(csv_path, index=False)
    
    with open(json_path, 'w') as f:
        json.dump(psi_report["diagnostics"], f, indent=4)
        
    return {"psi_report_csv": csv_path, "drift_diagnostics_json": json_path}


def check_drift_trigger(
    psi_report: Dict[str, Any], 
    auc_drop: float, 
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Evaluates the dual-gate drift trigger using config parameters.
    Returns the boolean trigger and a text summary for MLflow logging/reporting.
    """
    psi_threshold = config.get("psi_threshold", 0.25)
    auc_tolerance = config.get("auc_tolerance", 0.03)
    
    if not psi_report.get("is_reliable", False):
        summary = "Drift evaluation skipped. Sample size or bin counts too low for statistical reliability."
        return False, summary
        
    feature_psis = psi_report.get("features", {})
    if not feature_psis:
        return False, "No PSI features evaluated."
        
    max_feature = max(feature_psis, key=feature_psis.get)
    max_psi_observed = feature_psis[max_feature]
    
    # Dual-gate Corroboration
    psi_triggered = max_psi_observed >= psi_threshold
    auc_triggered = auc_drop >= auc_tolerance
    
    is_retrain_triggered = psi_triggered and auc_triggered
    
    summary = (
        f"Retrain Triggered: {is_retrain_triggered}. "
        f"Max PSI: {max_psi_observed:.3f} on '{max_feature}' (Threshold: {psi_threshold}). "
        f"AUC Drop: {auc_drop:.3f} (Tolerance: {auc_tolerance})."
    )
    
    return is_retrain_triggered, summary


def compute_wasserstein_distance(
    baseline_data: pd.DataFrame, 
    eval_data: pd.DataFrame, 
    feature_cols: List[str]
) -> Dict[str, float]:
    """Secondary drift detector immune to binning logic."""
    report = {"features": {}}
    
    for feature in feature_cols:
        if feature not in baseline_data.columns or feature not in eval_data.columns:
            raise KeyError(f"Feature '{feature}' missing from data for Wasserstein computation.")
            
        w_dist = wasserstein_distance(baseline_data[feature], eval_data[feature])
        report["features"][feature] = float(w_dist)
        
    return report