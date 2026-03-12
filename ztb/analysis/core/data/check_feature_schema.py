import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.io.data_loader import DataLoader
from ztb.io.json_io import read_json
from ztb.utils.config import TypedConfig

def check_feature_schema(
    dataset_path: str = "ml-dataset-enhanced.csv",
) -> dict[str, Any]:
    """Check feature schema compatibility.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Dictionary with schema comparison results
    """
    # Check model schema - use config-based path
    config = TypedConfig()
    schema_path = f"{config.get_model_dir()}/features_schema.json"

    try:
        schema = read_json(schema_path)
        expected_columns = schema.get("columns", [])
        print(f"Model expects {len(expected_columns)} columns")
        print(f"First 10: {expected_columns[:10]}\n")
    except FileNotFoundError:
        return {"error": f"Schema file not found: {schema_path}"}

    # Check dataset
    try:
        df = DataLoader.load_csv_strict(dataset_path)
        dataset_columns = list(df.columns)

        result = {
            "model_columns_count": len(expected_columns),
            "dataset_columns_count": len(dataset_columns),
            "dataset_shape": df.shape,
            "model_first_10": expected_columns[:10],
            "dataset_first_10": dataset_columns[:10],
            "columns_match": len(expected_columns) == len(dataset_columns),
        }

        print(f"Dataset has {len(dataset_columns)} columns")
        print(f"Dataset shape: {df.shape}")
        print(f"First 10 columns: {dataset_columns[:10]}")

        return result

    except FileNotFoundError:
        return {"error": f"Dataset file not found: {dataset_path}"}

if __name__ == "__main__":
    check_feature_schema()
