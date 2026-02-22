import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.io.text_io import write_text
from ztb.utils.config import TypedConfig


def show_model_features(
    model_path: Optional[str] = None, detailed: bool = False
) -> dict:
    """Show model features information.

    Args:
        model_path: Path to the model (unused for now, kept for compatibility)
        detailed: Whether to show detailed information

    Returns:
        Dictionary with feature information
    """
    # Load scaler feature names - use config-based path
    config = TypedConfig()
    scaler_path = f"{config.get_model_dir()}/scaler.npz"
    scaler_data = np.load(scaler_path, allow_pickle=True)
    feature_names = scaler_data["feature_names"].tolist()

    result = {
        "total_features": len(feature_names),
        "feature_names": feature_names,
        "first_20": feature_names[:20],
        "last_20": feature_names[-20:],
    }

    print(f"Total features in model: {len(feature_names)}\n")
    print("First 20 features:")
    for i, name in enumerate(feature_names[:20]):
        print(f"  {i+1:3d}. {name}")

    if detailed:
        print("\nLast 20 features:")
        for i, name in enumerate(feature_names[-20:], start=len(feature_names) - 19):
            print(f"  {i:3d}. {name}")

    # Save to file for reference
    content = "\n".join(f"{i:3d}. {name}" for i, name in enumerate(feature_names, 1))
    write_text("model_features_110.txt", content + "\n")

    print("\n✅ Full feature list saved to: model_features_110.txt")

    return result


if __name__ == "__main__":
    show_model_features()
