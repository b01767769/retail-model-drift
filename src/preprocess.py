import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_and_slice_data(df: pd.DataFrame, num_slices: int = 6) -> dict[int, pd.DataFrame]:
    """Cleans data and partitions it chronologically."""
    df.columns = df.columns.str.replace(' ', '').str.lower()
    
    df = df.dropna(subset=['customerid'])
    
    # 3. Handle Returns and Zero Prices
    # Exclude UnitPrice <= 0 unless it's a valid return (Quantity < 0)
    valid_purchases = (df['price'] > 0)
    valid_returns = (df['price'] >= 0) & (df['quantity'] < 0)
    df = df[valid_purchases | valid_returns].copy()
    
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    df = df.sort_values('invoicedate').reset_index(drop=True)
    
    # 5. Split into slices using numpy array splitting (efficient)
    slice_arrays = np.array_split(df, num_slices)
    
    slices_dict = {}
    for i, slice_df in enumerate(slice_arrays, start=1):
        slices_dict[i] = slice_df.copy()
        logger.info(f"Slice {i} created: {len(slice_df)} records.")
        
    return slices_dict

def inject_drift(df):
    df = df.copy()

    # 1. Mean shift
    df["income"] = df["income"] * 1.25

    # 2. Variance inflation
    df["age"] = df["age"] + np.random.normal(0, 10, len(df))

    # 3. Categorical skew
    df["region"] = df["region"].replace({
        "north": "south",
        "east": "south"
    })

    return df

