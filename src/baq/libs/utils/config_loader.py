import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        ConfigLoader._validate_config(config)
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate the configuration has all required sections"""
        required_sections = ['data', 'processing', 'model', 'training', 'mlops', 'artifacts']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data config
        if 'features' not in config['data']:
            raise ValueError("Missing 'features' in data configuration")
        
        if 'target' not in config['data']:
            raise ValueError("Missing 'target' in data configuration")
        
        if 'data_path' not in config['data']:
            raise ValueError("Missing 'data_path' in data configuration")
        
        # Validate model config
        if 'type' not in config['model']:
            raise ValueError("Missing 'type' in model configuration")
            
        if 'hyperparameters' not in config['model']:
            raise ValueError("Missing 'hyperparameters' in model configuration")
        
        model_type = config['model']['type']
        if model_type not in config['model']['hyperparameters']:
            raise ValueError(f"Hyperparameters for model type '{model_type}' not found")
        
        # Validate artifacts config
        if 'base_path' not in config['artifacts']:
            raise ValueError("Missing 'base_path' in artifacts configuration")
        
        required_artifacts = ['model', 'scaler', 'metrics', 'plots', 'reports']
        for artifact in required_artifacts:
            if artifact not in config['artifacts']:
                raise ValueError(f"Missing '{artifact}' in artifacts configuration") 