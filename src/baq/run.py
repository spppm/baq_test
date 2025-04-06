import argparse
import os
import sys
import pandas as pd

from baq.pipelines.iris_classification_pipeline import IrisClassificationPipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Iris Classification Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to the configuration file"
    )
    return parser.parse_args()


def check_encoder_mapping(artifacts_base_path):
    """Print class mapping information if available"""
    mapping_path = os.path.join(artifacts_base_path, "reports", "class_mapping.csv")
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        print("\nClass Encoding Mapping:")
        print("-" * 40)
        for _, row in mapping_df.iterrows():
            print(f"  {row['class_name']} -> {row['encoded_value']}")
        print("-" * 40)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print(f"Starting Iris Classification Pipeline with config: {args.config}")
    
    # Run the pipeline
    pipeline = IrisClassificationPipeline(args.config)
    results = pipeline.run()
    
    print("\nPipeline execution completed successfully.")
    
    # Print model performance summary
    if "metrics" in results:
        print("\nModel Performance Metrics:")
        for metric, value in results["metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    # If feature importance is available, print top features
    if "feature_importance" in results and not results["feature_importance"].empty:
        print("\nTop 5 Important Features:")
        for _, row in results["feature_importance"].head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Print class mapping if available
    check_encoder_mapping(results.get("artifacts_base_path", "artifacts"))
