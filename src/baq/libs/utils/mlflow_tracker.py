import mlflow
import os
from typing import Dict, Any, Optional


class MLflowTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlops_config = config["mlops"]["experiment_tracking"]
        
        # Set up MLflow tracking URI
        if "tracking_uri" in self.mlops_config:
            mlflow.set_tracking_uri(self.mlops_config["tracking_uri"])
        
        # Experiment name
        self.experiment_name = self.mlops_config.get("experiment_name", "iris_classification")
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run"""
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Generate a version name if versioning is enabled
        versioning = self.config["artifacts"].get("versioning", {})
        if versioning.get("enable", False) and not run_name:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_type = self.config["model"]["type"]
            pattern = versioning.get("naming_pattern", "{model_type}_{timestamp}")
            run_name = pattern.format(model_type=model_type, timestamp=timestamp)
        
        # Start the run
        mlflow.start_run(run_name=run_name)
    
    def log_parameters(self, params: Dict[str, Any] = None) -> None:
        """Log model parameters to MLflow"""
        if params is None:
            # Log parameters from config
            model_type = self.config["model"]["type"]
            params = self.config["model"]["hyperparameters"][model_type]
        
        # Log all parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log model type
        mlflow.log_param("model_type", self.config["model"]["type"])
        
        # Log dataset info
        mlflow.log_param("dataset", self.config["data"]["dataset"])
        
        # Log preprocessing info
        mlflow.log_param("scaling_method", self.config["processing"]["scaling"]["method"])
        
        # Log feature engineering info
        fe_config = self.config["processing"].get("feature_engineering", {})
        
        if fe_config.get("polynomial_features", {}).get("enable", False):
            mlflow.log_param("poly_degree", fe_config["polynomial_features"]["degree"])
        
        if fe_config.get("pca", {}).get("enable", False):
            mlflow.log_param("pca_components", fe_config["pca"]["n_components"])
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def log_artifacts(self) -> None:
        """Log artifacts to MLflow"""
        base_path = self.config["artifacts"]["base_path"]
        
        # Log model
        model_path = os.path.join(
            base_path, 
            self.config["artifacts"]["model"]["path"],
            self.config["artifacts"]["model"]["filename"]
        )
        
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model")
        
        # Log plots
        plots_path = os.path.join(base_path, self.config["artifacts"]["plots"]["path"])
        if os.path.exists(plots_path):
            for plot_name in ["confusion_matrix", "roc_curve", "feature_importance"]:
                plot_file = self.config["artifacts"]["plots"][plot_name]
                plot_path = os.path.join(plots_path, plot_file)
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, "plots")
        
        # Log reports
        reports_path = os.path.join(base_path, self.config["artifacts"]["reports"]["path"])
        if os.path.exists(reports_path):
            for report_name in ["classification_report", "model_card"]:
                report_file = self.config["artifacts"]["reports"][report_name]
                report_path = os.path.join(reports_path, report_file)
                if os.path.exists(report_path):
                    mlflow.log_artifact(report_path, "reports")
    
    def log_model(self, model) -> None:
        """Log scikit-learn model to MLflow"""
        mlflow.sklearn.log_model(model, "model")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        mlflow.end_run()
    
    def setup_model_registry(self, model, model_name: Optional[str] = None) -> None:
        """Register model in MLflow Model Registry if enabled"""
        if not self.config["mlops"]["model_registry"].get("enable", False):
            return
        
        if model_name is None:
            model_name = f"{self.config['data']['dataset']}_{self.config['model']['type']}"
        
        try:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=model_name
            )
        except Exception as e:
            print(f"Failed to register model: {e}")
    
    def get_best_model(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """Get the URI of the best model from MLflow Model Registry"""
        if not self.config["mlops"]["model_registry"].get("enable", False):
            return None
        
        client = mlflow.tracking.MlflowClient()
        
        try:
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            if latest_version:
                return latest_version[0].source
            return None
        except Exception as e:
            print(f"Failed to get best model: {e}")
            return None 