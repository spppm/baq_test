import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import joblib
import os
from typing import Dict, Tuple, Any, List


class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config["data"]
        self.processing_config = config["processing"]
        self.artifacts_config = config["artifacts"]
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Load dataset from specified path in config"""
        data_path = self.data_config["data_path"]
        return pd.read_csv(data_path)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target from the dataframe"""
        features = self.data_config["features"]
        target = self.data_config["target"]
        
        X = df[features]
        y = df[target]
        
        return X, y
    
    def encode_target(self, y: pd.Series) -> pd.Series:
        """Encode categorical target variable to numeric labels"""
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Save label encoder for later use
        self._save_label_encoder()
        
        # Create mapping dictionary for reference
        self._create_label_mapping()
        
        return pd.Series(y_encoded, index=y.index)
    
    def _save_label_encoder(self) -> None:
        """Save the fitted label encoder"""
        base_path = self.artifacts_config["base_path"]
        model_path = self.artifacts_config["model"]["path"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, model_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Save the label encoder
        joblib.dump(self.label_encoder, os.path.join(full_path, "label_encoder.pkl"))
    
    def _create_label_mapping(self) -> None:
        """Create and save a mapping of original class names to encoded values"""
        base_path = self.artifacts_config["base_path"]
        reports_path = self.artifacts_config["reports"]["path"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, reports_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Create mapping
        mapping = {
            'class_name': self.label_encoder.classes_,
            'encoded_value': np.arange(len(self.label_encoder.classes_))
        }
        mapping_df = pd.DataFrame(mapping)
        
        # Save as CSV
        mapping_df.to_csv(os.path.join(full_path, "class_mapping.csv"), index=False)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        split_config = self.data_config["split"]
        
        stratify = y if split_config.get("stratify", False) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test
    
    def _get_scaler(self, method: str) -> Any:
        """Return scaler based on method specified in config"""
        scaling_config = self.processing_config["scaling"]
        
        if method == "standard":
            return StandardScaler(
                with_mean=scaling_config.get("with_mean", True),
                with_std=scaling_config.get("with_std", True)
            )
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline based on config"""
        scaling_config = self.processing_config["scaling"]
        method = scaling_config["method"]
        scaler = self._get_scaler(method)
        
        missing_values_strategy = self.processing_config["missing_values"]
        imputer = SimpleImputer(strategy=missing_values_strategy)
        
        # Create basic preprocessing pipeline
        preprocessing_steps = [
            ("imputer", imputer),
            ("scaler", scaler)
        ]
        
        # Add feature engineering steps if enabled
        fe_config = self.processing_config.get("feature_engineering", {})
        
        if fe_config.get("polynomial_features", {}).get("enable", False):
            poly_degree = fe_config["polynomial_features"]["degree"]
            preprocessing_steps.append(
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
            )
            
        if fe_config.get("pca", {}).get("enable", False):
            n_components = fe_config["pca"]["n_components"]
            preprocessing_steps.append(
                ("pca", PCA(n_components=n_components))
            )
        
        return Pipeline(steps=preprocessing_steps)
    
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessing pipeline on training data and transform both train and test data"""
        preprocessing_pipeline = self.create_preprocessing_pipeline(X_train)
        
        X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
        X_test_transformed = preprocessing_pipeline.transform(X_test)
        
        # Save preprocessor
        self._save_preprocessor(preprocessing_pipeline)
        
        return X_train_transformed, X_test_transformed
    
    def _save_preprocessor(self, preprocessor: Pipeline) -> None:
        """Save the fitted preprocessor"""
        base_path = self.artifacts_config["base_path"]
        scaler_path = self.artifacts_config["scaler"]["path"]
        scaler_filename = self.artifacts_config["scaler"]["filename"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, scaler_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Save the preprocessor
        joblib.dump(preprocessor, os.path.join(full_path, scaler_filename))
    
    def process(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str]]:
        """Execute the full data processing pipeline"""
        # Load data
        df = self.load_data()
        
        # Prepare features and target
        X, y = self.prepare_data(df)
        
        # Encode target if it's string type
        if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
            y = self.encode_target(y)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Transform data
        X_train_transformed, X_test_transformed = self.fit_transform(X_train, X_test)
        
        return X_train_transformed, X_test_transformed, y_train, y_test, X.columns.tolist() 