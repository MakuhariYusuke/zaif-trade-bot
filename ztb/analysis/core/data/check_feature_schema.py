import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.utils.config import TypedConfig

# Check model schema - use config-based path
config = TypedConfig()
schema_path = f"{config.get_model_dir()}/features_schema.json"
schema = json.load(open(schema_path))
print(f"Model expects {len(schema.get('columns', []))} columns")
print(f"First 10: {schema.get('columns', [])[:10]}\n")

# Check dataset
df = pd.read_csv("ml-dataset-enhanced.csv")
print(f"Dataset has {len(df.columns)} columns")
print(f"Dataset shape: {df.shape}")
print(f"First 10 columns: {list(df.columns)[:10]}")
