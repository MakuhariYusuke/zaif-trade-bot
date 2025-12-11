import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Fix DLL error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ztb.utils.checkpoint import CheckpointManager


def inspect_checkpoint(path):
    print(f"Loading checkpoint from {path}")

    # Create a dummy manager to access load methods
    manager = CheckpointManager(save_dir=".")

    try:
        # Try to load using the manager's internal method
        # We need to access the private method _load_raw_checkpoint
        # But first, let's see if we can just read and decompress manually using the manager's logic

        with open(path, "rb") as f:
            compressed_data = f.read()

        print(f"Read {len(compressed_data)} bytes.")

        try:
            data = manager._decompress_data(compressed_data)
            print("Decompressed successfully.")
        except Exception as e:
            print(f"Decompression failed: {e}")
            # Try loading as plain pickle
            import pickle

            try:
                data = pickle.loads(compressed_data)
                print("Loaded as plain pickle.")
            except Exception as e2:
                print(f"Plain pickle load failed: {e2}")
                return

        # Now unpickle if it was compressed
        if "data" in locals() and isinstance(data, bytes):
            import pickle

            try:
                checkpoint_data = pickle.loads(data)
                print("Unpickled successfully.")
            except Exception as e:
                print(f"Unpickle failed: {e}")
                return
        elif "data" in locals():
            checkpoint_data = data

        # Inspect the data
        if isinstance(checkpoint_data, dict):
            print("Keys:", checkpoint_data.keys())

            # Check for config
            if "config" in checkpoint_data:
                print("Config found in root.")
                config = checkpoint_data["config"]
            elif (
                "obj" in checkpoint_data
                and isinstance(checkpoint_data["obj"], dict)
                and "config" in checkpoint_data["obj"]
            ):
                print("Config found in obj.")
                config = checkpoint_data["obj"]["config"]
            elif "obj" in checkpoint_data:
                print(
                    "Checking obj keys:",
                    checkpoint_data["obj"].keys()
                    if isinstance(checkpoint_data["obj"], dict)
                    else type(checkpoint_data["obj"]),
                )
                if (
                    isinstance(checkpoint_data["obj"], dict)
                    and "env_state" in checkpoint_data["obj"]
                ):
                    print("Env state found in obj.")
                    env_state = checkpoint_data["obj"]["env_state"]
                    if "features" in env_state:
                        print("Features in env_state:", env_state["features"])
                        print("Count:", len(env_state["features"]))

            if "config" in locals():
                print(f"Config type: {type(config)}")
                if isinstance(config, dict):
                    print("Config keys:", config.keys())
                    if "training" in config:
                        training_conf = config["training"]
                        print("Training config keys:", training_conf.keys())
                        if "environment" in training_conf:
                            print("Environment config:", training_conf["environment"])
                        if "data_config" in training_conf:
                            print("Data config:", training_conf["data_config"])

            if "model_state" in checkpoint_data:
                print("Checking model state...")
                model_state = checkpoint_data["model_state"]
                print(f"Model state type: {type(model_state)}")

                keys = list(model_state.keys())
                print(f"Model state keys: {keys[:10]}")

                # Check if it's nested
                first_val = model_state[keys[0]]
                print(f"Type of first value ({keys[0]}): {type(first_val)}")

                if isinstance(first_val, dict):
                    print("Nested state dict detected.")
                    # Iterate through nested dict
                    for k, v in first_val.items():
                        if hasattr(v, "shape"):
                            print(f"  {k}: {v.shape}")
                            if len(v.shape) == 2 and v.shape[1] in [138, 143]:
                                print(
                                    f"  FOUND INPUT LAYER in nested: {k} with input dim {v.shape[1]}"
                                )
                else:
                    # Flat state dict
                    for key in keys:
                        val = model_state[key]
                        if hasattr(val, "shape"):
                            if len(val.shape) == 2:
                                print(f"Layer: {key}, Shape: {val.shape}")
                                if val.shape[1] in [138, 143]:
                                    print(
                                        f"FOUND INPUT LAYER: {key} with input dim {val.shape[1]}"
                                    )
                        print(f"Layer: {key}, Shape: {model_state[key].shape}")
                        if model_state[key].shape[1] in [138, 143]:
                            print(
                                f"FOUND INPUT LAYER: {key} with input dim {model_state[key].shape[1]}"
                            )

            if "env_state" in checkpoint_data:
                print("Env state found in root.")
                env_state = checkpoint_data["env_state"]
                if "features" in env_state:
                    print("Features in env_state:", env_state["features"])
                    print("Count:", len(env_state["features"]))

        else:
            print("Checkpoint data is not a dict:", type(checkpoint_data))

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    path = r"c:\Users\Admin\dev\zaif-trade-bot\checkpoints\v451\phase7\training_state_20000_20251211_192328.pkl"
    inspect_checkpoint(path)
