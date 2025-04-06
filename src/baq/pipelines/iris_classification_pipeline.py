from typing import Dict, Any
import os
import logging

from ..libs.utils.config_loader import ConfigLoader
from ..steps.iris_classification_step import IrisClassificationStep


class IrisClassificationPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('IrisClassificationPipeline')
    
    def run(self) -> Dict[str, Any]:
        """Run the full Iris classification pipeline"""
        self.logger.info(f"Loading configuration from {self.config_path}")
        config = ConfigLoader.load_config(self.config_path)
        
        # Create necessary directories
        self._create_artifact_directories(config)
        
        # Execute the classification step
        self.logger.info("Starting Iris classification pipeline")
        step = IrisClassificationStep(config)
        results = step.execute()
        
        # Log results summary
        self._log_results_summary(results)
        
        self.logger.info("Iris classification pipeline completed successfully")
        return results
    
    def _create_artifact_directories(self, config: Dict[str, Any]) -> None:
        """Create directories for storing artifacts"""
        base_path = config["artifacts"]["base_path"]
        
        # Create base artifacts directory
        os.makedirs(base_path, exist_ok=True)
        
        # Create subdirectories for different artifact types
        for artifact_type in ["model", "scaler", "metrics", "plots", "reports"]:
            if artifact_type in config["artifacts"]:
                artifact_path = config["artifacts"][artifact_type]["path"]
                os.makedirs(os.path.join(base_path, artifact_path), exist_ok=True)
    
    def _log_results_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of the pipeline results"""
        self.logger.info("Pipeline Results Summary:")
        
        # Log metrics
        if "metrics" in results:
            self.logger.info("Evaluation Metrics:")
            for metric, value in results["metrics"].items():
                self.logger.info(f"  {metric}: {value:.4f}")
        
        # Log cross-validation results if available
        if "cv_results" in results and results["cv_results"]:
            self.logger.info("Cross-Validation Results:")
            for metric, value in results["cv_results"].items():
                self.logger.info(f"  {metric}: {value:.4f}")
        
        # Log top features if available
        if "feature_importance" in results and not results["feature_importance"].empty:
            self.logger.info("Top 5 Important Features:")
            for _, row in results["feature_importance"].head(5).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}") 