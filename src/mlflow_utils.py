import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# --- Enforced Governance Schemas ---
REQUIRED_PARAMS = {
    'model_type', 'n_estimators', 'max_depth', 'min_samples_leaf', 
    'train_slices', 'target_percentile', 'psi_bins', 'psi_min_count', 'random_seed'
}
REQUIRED_METRICS = {
    'auc', 'accuracy', 'precision', 'recall', 'f1', 
    'psi_recency', 'psi_frequency', 'psi_monetary', 'val_loss'
}
REQUIRED_TAGS = {
    'run_role', 'slice_number', 'drift_trigger', 'promotion'
}
REQUIRED_ARTIFACT_KEYS = {
    'model', 'scaler', 'feature_importance_plot', 'psi_report_csv', 
    'confusion_matrix', 'run_notes'
}

def generate_file_checksum(file_path: str, chunk_size: int = 8192) -> str:
    """
    Generates an MD5 cryptographic checksum for a given file to ensure data provenance.
    DSA Optimization: Reads the file in fixed-size chunks to maintain O(1) memory complexity.
    """
    target_path = Path(file_path)
    if not target_path.exists():
        logger.warning(f"Checksum failed: File not found at {target_path}")
        return "FILE_NOT_FOUND"
    
    try:
        file_hash = hashlib.md5()
        with open(target_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error computing checksum for {target_path}: {e}")
        return "CHECKSUM_ERROR"


def generate_pipeline_manifest(
    data_source_path: str,
    slice_boundaries: Dict[str, str],
    run_role: str,
    slice_number: int,
    output_path: str = "artifacts/pipeline_manifest.json"
) -> str:
    """
    Constructs and saves the pipeline manifest capturing data versions and slice boundaries.
    """
    manifest = {
        "data_source_path": data_source_path,
        "data_checksum_md5": generate_file_checksum(data_source_path),
        "run_role": run_role,
        "slice_evaluated": slice_number,
        "slice_boundaries": slice_boundaries,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    return output_path


def log_run_standard(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tags: Dict[str, Any],
    artifact_paths: Dict[str, str],
    experiment_name: str = "retail_model_drift"
) -> str:
    """
    Centralized MLflow logger enforcing the strict completeness checks.
    Raises ValueError if the injected dictionaries do not match the required schema.
    """
    missing_params = REQUIRED_PARAMS - params.keys()
    missing_metrics = REQUIRED_METRICS - metrics.keys()
    missing_tags = REQUIRED_TAGS - tags.keys()
    missing_artifacts = REQUIRED_ARTIFACT_KEYS - artifact_paths.keys()

    if any([missing_params, missing_metrics, missing_tags, missing_artifacts]):
        error_msg = (
            f"Run failed schema validation.\n"
            f"Missing Params: {missing_params}\n"
            f"Missing Metrics: {missing_metrics}\n"
            f"Missing Tags: {missing_tags}\n"
            f"Missing Artifacts: {missing_artifacts}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    mlflow.set_experiment(experiment_name)

    logger.info(f"Starting MLflow run for slice {tags['slice_number']}...")
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        
        mlflow.sklearn.log_model(artifact_paths['model'], "model_artifact", serialization_format="cloudpickle")
        mlflow.sklearn.log_model(artifact_paths['scaler'], "scaler_artifact", serialization_format="cloudpickle")
        
        for key, path_str in artifact_paths.items():
            if key not in ['model', 'scaler']:  # Already handled above
                if Path(path_str).exists():
                    mlflow.log_artifact(path_str, artifact_path="evaluation_artifacts")
                else:
                    logger.warning(f"Artifact missing from disk, skipping: {path_str}")
        
        active_run_id = run.info.run_id
        logger.info(f"Successfully logged strict schema run. ID: {active_run_id}")
        
        return active_run_id


def register_promoted_champion(
    run_id: str,
    model_registry_name: str,
    changelog_entry: str,
    psi_trigger_summary: str,
    validation_results_summary: str
) -> str:
    """
    Interfaces with the MLflow Model Registry to promote a model.
    Attaches changelogs and governance triggers to the version metadata.
    """
    logger.info(f"Registering run {run_id} to registry '{model_registry_name}'...")
    
    model_uri = f"runs:/{run_id}/model_artifact"
    mv = mlflow.register_model(model_uri, model_registry_name)
    
    client = MlflowClient()
    
    full_description = (
        f"**Changelog:** {changelog_entry}\n"
        f"**Validation Summary:** {validation_results_summary}\n"
        f"**PSI Trigger Context:** {psi_trigger_summary}"
    )
    client.update_model_version(
        name=model_registry_name,
        version=mv.version,
        description=full_description
    )
    
    client.set_model_version_tag(model_registry_name, mv.version, "psi_triggered", "true" if "retrain" in psi_trigger_summary.lower() else "false")
    client.set_model_version_tag(model_registry_name, mv.version, "validation_passed", "true")
    
    logger.info(f"Successfully registered Version {mv.version} of {model_registry_name}.")
    return mv.version