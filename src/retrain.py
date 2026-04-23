import logging
from typing import Dict, Tuple, List, Literal, Optional, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.train import train_challenger

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
    
    # Concatenate historical slices
    data_blocks = [historical_slice_dict[k] for k in eligible_keys]
    compiled_dataframe = pd.concat(data_blocks, axis=0, ignore_index=True)
    
    return compiled_dataframe

def execute_challenger_retraining(
    slice_repository: Dict[int, pd.DataFrame],
    current_slice_id: int,
    config: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Orchestrates the assembly of historical data and the training of a new Challenger model.
    Accepts the master configuration dictionary to ensure hyperparameters and strategies 
    remain synchronized with the orchestrator.
    """
    logger.info(f"--- Initiating Challenger Retraining for Slice {current_slice_id} ---")
    
    training_strategy = config.get("retrain_strategy", "cumulative")
    window_length = config.get("retrain_sliding_window_size", 1)
    
    compiled_training_data = assemble_training_window(
        historical_slice_dict=slice_repository,
        target_slice_idx=current_slice_id,
        build_strategy=training_strategy,
        sliding_window_size=window_length
    )
    
    dataset_size = len(compiled_training_data)
    logger.info(f"Challenger dataset compiled. Total historical records: {dataset_size}")
    
    challenger_model, challenger_train_metrics = train_challenger(
        cumulative_data=compiled_training_data,
        config=config,
        feature_cols=feature_columns,
    )
    
    # 3. Append Data Lineage Metrics for MLflow
    # Note: Explicitly casting to float/str to ensure MLflow schema compatibility
    challenger_train_metrics['training_strategy'] = str(training_strategy)
    challenger_train_metrics['training_records'] = float(dataset_size)
    
    return challenger_model, challenger_train_metrics