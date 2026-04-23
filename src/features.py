import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RFMFeatureEngineer:
    """
    Stateful RFM Feature Engineering pipeline.
    Must be fitted on training data to lock in Winsorization caps and Scaler parameters,
    then applied to subsequent data slices.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the engineer with injected configurations to avoid hardcoding.
        
        Expected config keys:
        - winsorize_percentile (float): e.g., 99.5
        - log_transform_monetary (bool): e.g., True
        - apply_scaling (bool): e.g., True
        - target_percentile (float): e.g., 75.0
        """
        self.config = config
        self.winsorize_pct = self.config.get('winsorize_percentile', 99.5)
        self.is_fitted = False
        
        self.caps = {'frequency': None, 'monetary': None}
        self.scaler = StandardScaler() if self.config.get('apply_scaling', True) else None

    def _compute_base_rfm(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Computes raw Recency, Frequency, and Monetary values securely."""
        if df.empty:
            raise ValueError("Cannot compute RFM on an empty DataFrame.")
            
        # Avoid SettingWithCopyWarning
        df = df.copy()
        df['spend'] = df['quantity'] * df['price']
        
        if reference_date is None:
            reference_date = df['invoicedate'].max()
            
        rfm = (
            df.groupby('customerid')
              .agg(
                  recency=('invoicedate', lambda x: (reference_date - x.max()).days),
                  frequency=('invoice', 'nunique'),
                  monetary=('spend', 'sum')
              )
              .reset_index()
        )
        return rfm

    def fit(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None):
        """Learns and stores parameters (Caps, Scaler) from the baseline/training slice."""
        logger.info("Fitting RFM Feature Engineer on training slice...")
        rfm = self._compute_base_rfm(df, reference_date)
        
        self.caps['frequency'] = np.percentile(rfm['frequency'], self.winsorize_pct)
        self.caps['monetary'] = np.percentile(rfm['monetary'], self.winsorize_pct)
        
        features_to_scale = ['recency', 'frequency', 'monetary']
        
        rfm['frequency'] = rfm['frequency'].clip(upper=self.caps['frequency'])
        rfm['monetary'] = rfm['monetary'].clip(upper=self.caps['monetary'])
        
        if self.config.get('log_transform_monetary', True):
            # Safe log transform: clip lower bound to 0 to avoid log(negative) NaNs
            rfm['monetary_log'] = np.log1p(rfm['monetary'].clip(lower=0))
            features_to_scale.append('monetary_log')
            
        # 3. Fit Scaler
        if self.scaler:
            self.scaler.fit(rfm[features_to_scale])
            
        self.is_fitted = True
        logger.info(f"Fitted caps -> Freq: {self.caps['frequency']}, Mon: {self.caps['monetary']}")
        return self

    def transform(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applies learned transformations to any slice and extracts diagnostics."""
        if not self.is_fitted:
            raise RuntimeError("You must call fit() on training data before calling transform().")
            
        rfm = self._compute_base_rfm(df, reference_date)
        
        diagnostics = self._extract_diagnostics(rfm)
        
        rfm['frequency'] = rfm['frequency'].clip(upper=self.caps['frequency'])
        rfm['monetary_raw'] = rfm['monetary'] 
        rfm['monetary'] = rfm['monetary'].clip(upper=self.caps['monetary'])
        
        features_to_scale = ['recency', 'frequency', 'monetary']
        if self.config.get('log_transform_monetary', True):
            rfm['monetary_log'] = np.log1p(rfm['monetary'].clip(lower=0))
            features_to_scale.append('monetary_log')
            
        if self.scaler:
            scaled_features = self.scaler.transform(rfm[features_to_scale])
            
            for i, col in enumerate(features_to_scale):
                rfm[f'{col}_scaled'] = scaled_features[:, i]
                
        return rfm, diagnostics

    def fit_transform(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Convenience method for the training slice."""
        return self.fit(df, reference_date).transform(df, reference_date)

    def _extract_diagnostics(self, rfm: pd.DataFrame) -> Dict[str, Any]:
        """Calculates skewness, kurtosis, deciles, and negative counts safely."""
        diag = {}
        
        diag['missing_counts'] = rfm.isnull().sum().to_dict()
        
        total_customers = len(rfm)
        neg_monetary = (rfm['monetary'] < 0).sum()
        diag['negative_monetary_prop'] = (neg_monetary / total_customers) if total_customers > 0 else 0.0

        for col in ['recency', 'frequency', 'monetary']:
            col_data = rfm[col].dropna()
            if len(col_data) > 0:
                diag[f'{col}_skew'] = float(skew(col_data))
                diag[f'{col}_kurtosis'] = float(kurtosis(col_data))
                
                diag[f'{col}_deciles'] = np.percentile(col_data, np.arange(0, 101, 10)).tolist()
                counts, bin_edges = np.histogram(col_data, bins='auto')
                diag[f'{col}_hist_counts'] = counts.tolist()
                diag[f'{col}_hist_edges'] = bin_edges.tolist()
                
        return diag


def assign_targets(current_rfm: pd.DataFrame, next_rfm: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assigns binary labels based on the subsequent slice's monetary value.
    Extracts the threshold target percentile from the central config.
    """
    percentile = config.get('target_percentile', 75.0)
    
    threshold = np.percentile(next_rfm['monetary_raw'].clip(lower=0), percentile)
    
    # Map next slice monetary to current customers (Left Join to assume 0 spend if missing)
    next_spend = next_rfm[['customerid', 'monetary_raw']].rename(columns={'monetary_raw': 'next_monetary'})
    
    merged = current_rfm.merge(next_spend, on='customerid', how='left')
    merged['next_monetary'] = merged['next_monetary'].fillna(0)
    
    merged['target'] = (merged['next_monetary'] > threshold).astype(int)
    
    return merged.drop(columns=['next_monetary'])