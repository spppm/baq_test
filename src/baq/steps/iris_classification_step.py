from typing import Dict, Any

from ..libs.processors.data_processor import DataProcessor
from ..libs.models.model_trainer import ModelTrainer
from ..libs.evaluators.model_evaluator import ModelEvaluator
from ..libs.utils.mlflow_tracker import MLflowTracker


class IrisClassificationStep:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.model_trainer = ModelTrainer(config)
        self.model_evaluator = ModelEvaluator(config)
        self.mlflow_tracker = MLflowTracker(config)
    
    def execute(self) -> Dict[str, Any]:
        """Execute the complete Iris classification pipeline"""
        # Start MLflow run
        self.mlflow_tracker.start_run()
        
        try:
            # Log parameters
            self.mlflow_tracker.log_parameters()
            
            # Process data
            X_train, X_test, y_train, y_test, feature_names = self.data_processor.process()
            
            # Cross-validation if enabled
            cv_results = self.model_trainer.cross_validate(X_train, y_train)
            
            # Train model
            model = self.model_trainer.train(X_train, y_train)
            
            # Save model
            self.model_trainer.save_model()
            
            # Evaluate model
            metrics = self.model_evaluator.run_evaluation(model, X_test, y_test, feature_names)
            
            # Log CV metrics if available
            if cv_results:
                self.mlflow_tracker.log_metrics(cv_results)
            
            # Log evaluation metrics
            self.mlflow_tracker.log_metrics(metrics)
            
            # Log model to MLflow
            self.mlflow_tracker.log_model(model)
            
            # Log artifacts
            self.mlflow_tracker.log_artifacts()
            
            # Register model if enabled
            self.mlflow_tracker.setup_model_registry(model)
            
            # Prepare results
            results = {
                "metrics": metrics,
                "cv_results": cv_results,
                "model": model,
                "artifacts_base_path": self.config["artifacts"]["base_path"]
            }
            
            # Get feature importance if available
            feature_importance = self.model_trainer.get_feature_importance(feature_names)
            if feature_importance is not None:
                results["feature_importance"] = feature_importance
            
            return results
            
        finally:
            # End MLflow run
            self.mlflow_tracker.end_run() 