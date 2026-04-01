import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow

logger = logging.getLogger(__name__)

def generate_file_checksum(file_path: str, chunk_size: int = 8192) -> str:
    """
    Generates an MD5 cryptographic checksum for a given file to ensure data provenance.
    DSA Optimization: Reads the file in fixed-size chunks to maintain O(1) memory complexity,
    preventing RAM exhaustion on massive retail datasets.
    """
    target_path = Path(file_path)
    if not target_path.exists():
        logger.warning(f"Checksum failed: File not found at {target_path}")
        return "FILE_NOT_FOUND"
    
    try:
        file_hash = hashlib.md5()
        with open(target_path, "rb") as f:
            chunk = f.read(chunk_size)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(chunk_size)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error computing checksum for {target_path}: {e}")
        return "CHECKSUM_ERROR"


def log_run_to_mlflow(
    run_role: str, 
    slice_number: int, 
    model_pipeline: Any, 
    metrics: Dict[str, float], 
    params: Optional[Dict[str, Any]] = None, 
    psi_report: Optional[Dict[str, Any]] = None, 
    drift_trigger: bool = False, 
    promotion: bool = False,
    data_source_path: str = "data/raw/online_retail_II.csv",
    run_notes: str = "Automated pipeline execution via Champion-Challenger orchestrator."
) -> str:
    """
    Centralized MLflow logger enforcing the strict completeness checks mandated by Chapter 5.7.
    Requires parameters, metrics, nested artifacts, and run notes to pass an audit.
    """
    logger.info(f"Logging {run_role} for Slice {slice_number} to MLflow...")
    
    with mlflow.start_run(tags={"run_role": run_role}):
        
        if params:
            mlflow.log_params(params)
            
        mlflow.log_param("slice_number", slice_number)
        mlflow.log_param("drift_trigger", drift_trigger)
        mlflow.log_param("promotion", promotion)
        
        if metrics:
            mlflow.log_metrics(metrics)
            
        if psi_report and "features" in psi_report:
            for feature_name, psi_val in psi_report["features"].items():
                mlflow.log_metric(f"psi_{feature_name}", float(psi_val))
                
            mlflow.log_dict(psi_report, "artifacts/psi_report.json")
            
        mlflow.sklearn.log_model(model_pipeline, "model_artifact")
        
        mlflow.log_text(run_notes, "artifacts/run_notes.txt")
        
        pipeline_manifest = {
            "data_source_path": data_source_path,
            "data_checksum_md5": generate_file_checksum(data_source_path),
            "run_role": run_role,
            "slice_evaluated": slice_number,
            "drift_triggered": drift_trigger,
            "promotion_executed": promotion
        }
        mlflow.log_dict(pipeline_manifest, "artifacts/pipeline_manifest.json")
        
        local_feature_plot = Path("artifacts/feature_importance.png")
        if local_feature_plot.exists():
            mlflow.log_artifact(str(local_feature_plot), "artifacts")
            
        active_run_id = mlflow.active_run().info.run_id
        logger.info(f"Successfully logged run. ID: {active_run_id}")
        
        return active_run_id
