import logging
from typing import Dict, List, Any

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
            
            # Secure the outer edges to catch all future out-of-bounds data
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
) -> Dict[str, Any]:
    """
    Computes Population Stability Index (PSI) with strict reliability constraints and 
    mathematical epsilon fallbacks.
    """
    if eval_data.empty:
        raise ValueError("Cannot compute PSI on an empty evaluation DataFrame.")
        
    report = {
        "is_reliable": True, 
        "features": {}
    }
    
    n_eval_samples = len(eval_data)
    
    if n_eval_samples < min_customers:
        report["is_reliable"] = False
        logger.warning(f"Slice size ({n_eval_samples}) below minimum threshold ({min_customers}). PSI flagged as unreliable.")
        
    for feature, edges in baseline_bins.items():
        if feature not in eval_data.columns:
            raise KeyError(f"Feature '{feature}' missing from evaluation data.")
            
        # The expected distribution was originally split uniformly by quantiles.
        # If duplicates were dropped during creation, n_bins_actual might be < num_bins.
        n_bins_actual = len(edges) - 1
        expected_proportions = np.ones(n_bins_actual) / n_bins_actual
        
        observed_counts, _ = np.histogram(eval_data[feature], bins=edges)
        
        if np.any(observed_counts < min_bin_count):
            report["is_reliable"] = False
            logger.debug(f"Feature '{feature}' has bin counts below minimum threshold ({min_bin_count}).")
            
        observed_proportions = observed_counts / n_eval_samples
        
        expected_proportions = np.where(expected_proportions == 0, epsilon, expected_proportions)
        observed_proportions = np.where(observed_proportions == 0, epsilon, observed_proportions)
        
        # PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))
        psi_array = (observed_proportions - expected_proportions) * np.log(observed_proportions / expected_proportions)
        
        report["features"][feature] = float(np.sum(psi_array))
        
    return report


def check_drift_trigger(
    psi_report: Dict[str, Any], 
    auc_drop: float, 
    psi_threshold: float = 0.25, 
    auc_tolerance: float = 0.03
) -> bool:
    """
    Evaluates the dual-gate drift trigger. Requires BOTH severe distributional shift (PSI)
    AND corroborated performance degradation (AUC drop).
    """
    if not psi_report.get("is_reliable", False):
        logger.info("Drift trigger suppressed: PSI report flagged as statistically unreliable.")
        return False
        
    feature_psis = psi_report.get("features", {})
    if not feature_psis:
        return False
        
    max_psi_observed = max(feature_psis.values())
    
    # Corroborated logic: AND gate
    if max_psi_observed >= psi_threshold and auc_drop >= auc_tolerance:
        return True
        
    return False


def compute_wasserstein_distance(
    baseline_data: pd.DataFrame, 
    eval_data: pd.DataFrame, 
    feature_cols: List[str]
) -> Dict[str, float]:
    """
    Computes Earth Mover's Distance (Wasserstein) as a secondary drift detector.
    Used specifically for robustness checks as it is immune to bin-size sensitivity.
    """
    report = {"features": {}}
    
    for feature in feature_cols:
        if feature not in baseline_data.columns or feature not in eval_data.columns:
            raise KeyError(f"Feature '{feature}' missing from data for Wasserstein computation.")
            
        w_dist = wasserstein_distance(baseline_data[feature], eval_data[feature])
        report["features"][feature] = float(w_dist)
        
    return report
