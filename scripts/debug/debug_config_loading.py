import json

from ztb.config.unified_config import UnifiedConfig

config_path = "config/sac_v446_base_template.json"
print(f"Loading {config_path}")

try:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"JSON loaded. Keys: {list(data.keys())}")
    if "environment" in data:
        print(f"Environment keys: {list(data['environment'].keys())}")
    else:
        print("Environment key missing in JSON data")

    unified_config = UnifiedConfig.from_file(config_path)
    print("UnifiedConfig loaded")
    print(f"UnifiedConfig environment keys: {list(unified_config.environment.keys())}")

except Exception as e:
    print(f"Error: {e}")
