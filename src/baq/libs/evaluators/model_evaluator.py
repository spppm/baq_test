import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from typing import Dict, Any, List, Optional

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


class ModelEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config["training"]
        self.artifacts_config = config["artifacts"]
        self.label_encoder = None
        
    def _load_label_encoder(self) -> Optional[LabelEncoder]:
        """Load label encoder if it exists"""
        base_path = self.artifacts_config["base_path"]
        model_path = self.artifacts_config["model"]["path"]
        encoder_path = os.path.join(base_path, model_path, "label_encoder.pkl")
        
        if os.path.exists(encoder_path):
            return joblib.load(encoder_path)
        return None
        
    def evaluate(self, model: BaseEstimator, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model and calculate metrics"""
        metrics = self.training_config["metrics"]
        results = {}
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate specified metrics
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_test, y_pred)
            
        if "precision" in metrics:
            # Explicitly use weighted average for multiclass
            results["precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            
        if "recall" in metrics:
            # Explicitly use weighted average for multiclass
            results["recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
        if "f1" in metrics:
            # Explicitly use weighted average for multiclass
            results["f1"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return results
    
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save metrics to a JSON file"""
        base_path = self.artifacts_config["base_path"]
        metrics_path = self.artifacts_config["metrics"]["path"]
        metrics_filename = self.artifacts_config["metrics"]["filename"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, metrics_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(full_path, metrics_filename), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def generate_confusion_matrix(self, model: BaseEstimator, X_test: np.ndarray, y_test: pd.Series) -> None:
        """Generate and save confusion matrix plot"""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Try to load label encoder to get original class names
        self.label_encoder = self._load_label_encoder()
        
        # Use original class names if available, otherwise use model classes
        if self.label_encoder is not None:
            class_labels = self.label_encoder.classes_
        else:
            class_labels = model.classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                  xticklabels=class_labels, 
                  yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the plot
        base_path = self.artifacts_config["base_path"]
        plots_path = self.artifacts_config["plots"]["path"]
        cm_filename = self.artifacts_config["plots"]["confusion_matrix"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, plots_path)
        os.makedirs(full_path, exist_ok=True)
        
        plt.savefig(os.path.join(full_path, cm_filename))
        plt.close()
    
    def generate_roc_curve(self, model: BaseEstimator, X_test: np.ndarray, y_test: pd.Series) -> None:
        """Generate and save ROC curve plot for multiclass classification"""
        # For ROC curve, we need class probabilities
        if not hasattr(model, "predict_proba"):
            return
        
        y_score = model.predict_proba(X_test)
        
        y_test_bin = pd.get_dummies(y_test).values
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        plt.figure(figsize=(10, 8))
        
        # Try to load label encoder to get original class names
        if self.label_encoder is None:
            self.label_encoder = self._load_label_encoder()
        
        # Use original class names if available, otherwise use model classes
        if self.label_encoder is not None:
            class_labels = self.label_encoder.classes_
        else:
            class_labels = model.classes_
        
        for i, class_name in enumerate(model.classes_):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Get original class name if available
            display_name = class_labels[i] if i < len(class_labels) else class_name
            
            plt.plot(
                fpr[i], 
                tpr[i], 
                lw=2,
                label=f'ROC curve for {display_name} (area = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot
        base_path = self.artifacts_config["base_path"]
        plots_path = self.artifacts_config["plots"]["path"]
        roc_filename = self.artifacts_config["plots"]["roc_curve"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, plots_path)
        os.makedirs(full_path, exist_ok=True)
        
        plt.savefig(os.path.join(full_path, roc_filename))
        plt.close()
    
    def save_classification_report(self, model: BaseEstimator, X_test: np.ndarray, y_test: pd.Series) -> None:
        """Generate and save classification report"""
        y_pred = model.predict(X_test)
        
        # If label encoder is available, create a report with original class names
        if self.label_encoder is None:
            self.label_encoder = self._load_label_encoder()
        
        # Generate the standard classification report with zero_division=0 to handle potential division by zero
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # If we have a label encoder, modify the report to include original class names
        if self.label_encoder is not None:
            # Create a new report with original class names
            enhanced_report = {}
            for key, value in report.items():
                if key.isdigit() or (isinstance(key, (int, float)) and key in model.classes_):
                    # This is a class key, convert to original name
                    class_idx = int(key)
                    if 0 <= class_idx < len(self.label_encoder.classes_):
                        class_name = self.label_encoder.classes_[class_idx]
                        enhanced_report[class_name] = value
                else:
                    # This is a general metric key
                    enhanced_report[key] = value
            
            report = enhanced_report
        
        base_path = self.artifacts_config["base_path"]
        reports_path = self.artifacts_config["reports"]["path"]
        report_filename = self.artifacts_config["reports"]["classification_report"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, reports_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Save report
        with open(os.path.join(full_path, report_filename), 'w') as f:
            json.dump(report, f, indent=4)
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame) -> None:
        """Plot and save feature importance"""
        if feature_importance is None or feature_importance.empty:
            return
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        base_path = self.artifacts_config["base_path"]
        plots_path = self.artifacts_config["plots"]["path"]
        importance_filename = self.artifacts_config["plots"]["feature_importance"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, plots_path)
        os.makedirs(full_path, exist_ok=True)
        
        plt.savefig(os.path.join(full_path, importance_filename))
        plt.close()
    
    def create_model_card(self, metrics: Dict[str, float], feature_importance: pd.DataFrame = None) -> None:
        """Generate a model card with key information"""
        base_path = self.artifacts_config["base_path"]
        reports_path = self.artifacts_config["reports"]["path"]
        model_card_filename = self.artifacts_config["reports"]["model_card"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, reports_path)
        os.makedirs(full_path, exist_ok=True)
        
        model_type = self.config["model"]["type"]
        hyperparams = self.config["model"]["hyperparameters"][model_type]
        
        with open(os.path.join(full_path, model_card_filename), 'w') as f:
            f.write("# Model Card\n\n")
            
            f.write("## Model Details\n\n")
            f.write(f"- **Model Type:** {model_type}\n")
            f.write(f"- **Dataset:** {self.config['data']['dataset']}\n")
            f.write("- **Hyperparameters:**\n")
            for param, value in hyperparams.items():
                f.write(f"  - {param}: {value}\n")
            
            # Add class information if available
            if self.label_encoder is not None:
                f.write("\n## Class Information\n\n")
                f.write("| Class Name | Encoded Value |\n")
                f.write("|------------|---------------|\n")
                for i, class_name in enumerate(self.label_encoder.classes_):
                    f.write(f"| {class_name} | {i} |\n")
            
            f.write("\n## Performance Metrics\n\n")
            for metric, value in metrics.items():
                f.write(f"- **{metric}:** {value:.4f}\n")
            
            if feature_importance is not None and not feature_importance.empty:
                f.write("\n## Top Features\n\n")
                for _, row in feature_importance.head(10).iterrows():
                    f.write(f"- **{row['feature']}:** {row['importance']:.4f}\n")
    
    def generate_precision_recall_curve(self, model: BaseEstimator, X_test: np.ndarray, y_test: pd.Series) -> None:
        """Generate and save precision-recall curve plot for multiclass classification"""
        if not hasattr(model, "predict_proba"):
            return
        
        y_score = model.predict_proba(X_test)
        
        y_test_bin = pd.get_dummies(y_test).values
        
        # Try to load label encoder to get original class names
        if self.label_encoder is None:
            self.label_encoder = self._load_label_encoder()
        
        # Use original class names if available, otherwise use model classes
        if self.label_encoder is not None:
            class_labels = self.label_encoder.classes_
        else:
            class_labels = model.classes_
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(model.classes_):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
            
            # Get original class name if available
            display_name = class_labels[i] if i < len(class_labels) else class_name
            
            plt.plot(
                recall, 
                precision, 
                lw=2,
                label=f'{display_name} (AP = {avg_precision:.2f})'
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(True)
        
        # Save the plot
        base_path = self.artifacts_config["base_path"]
        plots_path = self.artifacts_config["plots"]["path"]
        pr_filename = self.artifacts_config["plots"]["precision_recall_curve"]
        
        full_path = os.path.join(base_path, plots_path)
        os.makedirs(full_path, exist_ok=True)
        
        plt.savefig(os.path.join(full_path, pr_filename))
        plt.close()
    
    def run_evaluation(self, model: BaseEstimator, X_test: np.ndarray, y_test: pd.Series, feature_names: List[str]) -> Dict[str, float]:
        """Run all evaluations and save artifacts"""
        # Try to load the label encoder first
        self.label_encoder = self._load_label_encoder()
        
        # Calculate metrics
        metrics = self.evaluate(model, X_test, y_test)
        
        # Save metrics to file
        self.save_metrics(metrics)
        
        # Generate and save confusion matrix
        self.generate_confusion_matrix(model, X_test, y_test)
        
        # Generate and save ROC curve
        self.generate_roc_curve(model, X_test, y_test)
        
        # Generate and save precision-recall curve
        self.generate_precision_recall_curve(model, X_test, y_test)
        
        # Save classification report
        self.save_classification_report(model, X_test, y_test)
        
        # Get and plot feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
            # Plot feature importance
            self.plot_feature_importance(feature_importance)
        
        # Create model card
        self.create_model_card(metrics, feature_importance)
        
        return metrics 