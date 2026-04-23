import pandas as pd
import numpy as np

def compute_rfm_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Compute Recency, Frequency and Monetary features from transactional data."""
    
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

def assign_targets(current_rfm: pd.DataFrame, next_rfm: pd.DataFrame, percentile: float = 75.0) -> pd.DataFrame:
    """Assigns binary labels based on the subsequent slice's monetary value."""
    # Calculate the threshold on the NEXT slice
    threshold = np.percentile(next_rfm['monetary'].clip(lower=0), percentile)
    
    # Map next slice monetary to current customers (Left Join to assume 0 spend if missing)
    next_spend = next_rfm[['customerid', 'monetary']].rename(columns={'monetary': 'next_monetary'})
    merged = current_rfm.merge(next_spend, on='customerid', how='left')
    merged['next_monetary'] = merged['next_monetary'].fillna(0)
    
    # Create target
    merged['target'] = (merged['next_monetary'] > threshold).astype(int)
    return merged.drop(columns=['next_monetary'])
