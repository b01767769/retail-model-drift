import mlflow
from sklearn.pipeline import Pipeline

def log_run_to_mlflow(run_role: str, slice_number: int, model: Pipeline, metrics: dict, params: dict = None, psi_report: dict = None, drift_trigger: bool = False, promotion: bool = False):
    """Centralized logging to ensure auditability."""
    with mlflow.start_run(tags={"run_role": run_role}):
        if params:
            mlflow.log_params(params)
            
        mlflow.log_metrics(metrics)
        mlflow.log_param("slice_number", slice_number)
        mlflow.log_param("drift_trigger", drift_trigger)
        mlflow.log_param("promotion", promotion)
        
        if psi_report:
            mlflow.log_dict(psi_report, "artifacts/psi_report.json")
            
        mlflow.sklearn.log_model(model, "model_artifact")
        return mlflow.active_run().info.run_id
