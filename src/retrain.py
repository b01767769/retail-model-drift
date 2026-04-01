import logging
import pandas as pd
from typing import Dict, Tuple, List, Literal, Optional
from sklearn.pipeline import Pipeline

from src.train import train_baseline_rf

logger = logging.getLogger(__name__)

def assemble_training_window(
    historical_slice_dict: Dict[int, pd.DataFrame],
    target_slice_idx: int,
    build_strategy: Literal["cumulative", "sliding"] = "cumulative",
    sliding_window_size: int = 1
) -> pd.DataFrame:
    """
    Securely compiles historical data for challenger training.
    Strictly enforces temporal boundaries to prevent look-ahead bias.
    """
    if target_slice_idx <= 1:
        raise ValueError(f"Cannot train a challenger for Slice {target_slice_idx}. Insufficient history.")
        
    if build_strategy == "cumulative":
        eligible_keys = [k for k in historical_slice_dict.keys() if k < target_slice_idx]
    
    elif build_strategy == "sliding":
        min_allowed_idx = max(1, target_slice_idx - sliding_window_size)
        eligible_keys = [k for k in historical_slice_dict.keys() if min_allowed_idx <= k < target_slice_idx]
    
    else:
        raise ValueError(f"Invalid build_strategy '{build_strategy}'. Must be 'cumulative' or 'sliding'.")

    if not eligible_keys:
        raise ValueError(f"Data assembly failed: No valid slices found for strategy '{build_strategy}' before Slice {target_slice_idx}.")

    eligible_keys.sort()
    logger.info(f"Assembling {build_strategy} challenger data using Slices: {eligible_keys}")
    
    data_blocks = [historical_slice_dict[k] for k in eligible_keys]
    compiled_dataframe = pd.concat(data_blocks, axis=0, ignore_index=True)
    
    return compiled_dataframe

def execute_challenger_retraining(
    slice_repository: Dict[int, pd.DataFrame],
    current_slice_id: int,
    feature_columns: Optional[List[str]] = None,
    target_column: str = 'target',
    training_strategy: Literal["cumulative", "sliding"] = "cumulative",
    window_length: int = 1,
    random_seed: int = 42
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Orchestrates the assembly of historical data and the training of a new Challenger model.
    """
    logger.info(f"--- Initiating Challenger Retraining for Slice {current_slice_id} ---")
    
    compiled_training_data = assemble_training_window(
        historical_slice_dict=slice_repository,
        target_slice_idx=current_slice_id,
        build_strategy=training_strategy,
        sliding_window_size=window_length
    )
    
    dataset_size = len(compiled_training_data)
    logger.info(f"Challenger dataset compiled. Total historical records: {dataset_size}")
    
    challenger_pipeline, challenger_train_metrics = train_baseline_rf(
        training_data=compiled_training_data,
        feature_cols=feature_columns,
        target_col=target_column,
        random_seed=random_seed
    )
    
    challenger_train_metrics['training_strategy'] = training_strategy
    challenger_train_metrics['training_records'] = dataset_size
    
    return challenger_pipeline, challenger_train_metrics
