import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_quantile_bins(df: pd.DataFrame, features: list, bins: int = 10) -> dict:
    """Calculates fixed baseline bins for PSI computation."""
    bin_dict = {}
    for feature in features:
        # Generate unique bin edges using quantiles
        _, edges = pd.qcut(df[feature], q=bins, retbins=True, duplicates='drop')
        # Ensure outer edges catch all future outliers
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_dict[feature] = edges
    return bin_dict

def compute_psi_report(baseline_bins: dict, current_data: pd.DataFrame, min_customers: int = 50, min_bin_count: int = 5) -> dict:
    """Computes operational PSI with reliability constraints."""
    report = {"is_reliable": True, "features": {}}
    
    if len(current_data) < min_customers:
        report["is_reliable"] = False
        logger.warning(f"Slice size ({len(current_data)}) below minimum threshold.")
    
    epsilon = 1e-6 # To avoid division by zero
    
    for feature, edges in baseline_bins.items():
        # Using numpy histogram for fast binning
        expected_counts, _ = np.histogram(current_data[feature], bins=edges) # Approximation for baseline distribution
        observed_counts, _ = np.histogram(current_data[feature], bins=edges)
        
        # In a real system, expected_counts comes directly from the baseline slice data.
        # For this snippet, we assume equal distribution (deciles) if baseline_bins were perfect deciles:
        expected_prop = np.ones(len(edges)-1) / (len(edges)-1) 
        
        observed_prop = observed_counts / len(current_data)
        
        # Reliability check
        if np.any(observed_counts < min_bin_count):
            report["is_reliable"] = False
            
        # Add epsilon
        expected_prop = np.where(expected_prop == 0, epsilon, expected_prop)
        observed_prop = np.where(observed_prop == 0, epsilon, observed_prop)
        
        # PSI Formula
        psi_val = np.sum((observed_prop - expected_prop) * np.log(observed_prop / expected_prop))
        report["features"][feature] = float(psi_val)
        
    return report

def check_drift_trigger(psi_report: dict, auc_drop: float, psi_threshold: float = 0.25, auc_tolerance: float = 0.03) -> bool:
    """Requires corroboration between PSI and performance before triggering."""
    if not psi_report["is_reliable"]:
        return False
        
    max_psi = max(psi_report["features"].values())
    if max_psi >= psi_threshold and auc_drop >= auc_tolerance:
        return True
    return False