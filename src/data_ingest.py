import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str | Path) -> pd.DataFrame:
    """Securely loads raw transaction data."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    logger.info(f"Loading data from {path}...")
    cols = ['Invoice', 'StockCode', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID']
    df = pd.read_csv(path, usecols=cols)
    return df