import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_and_slice_data(df: pd.DataFrame, num_slices: int = 6) -> dict[int, pd.DataFrame]:
    """Cleans data and partitions it into equal chronological time blocks."""
    # 1. Standardize columns
    df.columns = df.columns.str.replace(' ', '').str.lower()
    
    # 2. Drop missing customers
    df = df.dropna(subset=['customerid'])
    
    # 3. Handle Returns and Zero Prices
    # Both valid purchases and valid returns should likely have a price > 0
    valid_purchases = (df['price'] > 0) & (df['quantity'] > 0)
    valid_returns = (df['price'] > 0) & (df['quantity'] < 0)
    df = df[valid_purchases | valid_returns].copy()
    
    # 4. Chronological Sorting
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    df = df.sort_values('invoicedate').reset_index(drop=True)
    
    # 5. Split by Time Windows (Prevents fracturing invoices/days)
    start_date = df['invoicedate'].min()
    end_date = df['invoicedate'].max()
    
    date_bins = pd.date_range(start=start_date, end=end_date, periods=num_slices + 1)
    
    slices_dict = {}
    for i in range(num_slices):
        # Define the window (inclusive on left, exclusive on right, except the last bin)
        mask = (df['invoicedate'] >= date_bins[i])
        if i < num_slices - 1:
            mask &= (df['invoicedate'] < date_bins[i+1])
        else:
            mask &= (df['invoicedate'] <= date_bins[i+1])
            
        slice_df = df[mask].copy()
        slices_dict[i + 1] = slice_df
        
        logger.info(f"Slice {i + 1} ({date_bins[i].date()} to {date_bins[i+1].date()}): {len(slice_df)} records.")
        
    return slices_dict

def inject_drift(df):
    df = df.copy()

    # 1. Mean shift
    df["price"] = df ["price"] * 1.25

    # 2. Variance inflation
    df["age"] = df["age"] + np.random.normal(0, 10, len(df))

    # 3. Categorical skew
    df["region"] = df["region"].replace({
        "north": "south",
        "east": "south"
    })

    return df


