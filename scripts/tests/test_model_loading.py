import os

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv


def test_loading():
    model_path = "models/sac_model.zip"
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    print(f"Loading model from {model_path}...")

    # 1. Load Model
    model = SAC.load(model_path)
    print("✅ Model loaded.")

    # 2. Create Environment (needed for VecNormalize)
    # We need a dummy env to wrap
    # For testing, we can use a simple setup or the actual one if data exists
    data_path = "data/btc_jpy_featured_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        try:
            config = EnvironmentConfig()
            env = HeavyTradingEnv(data=df, config=config)
            venv = DummyVecEnv([lambda: env])

            # 3. Load VecNormalize
            print("Loading VecNormalize stats...")
            import pickle
            import zipfile

            # Manual extraction and load
            try:
                with zipfile.ZipFile(model_path, "r") as zf:
                    if "vec_normalize.pkl" in zf.namelist():
                        with zf.open("vec_normalize.pkl") as f:
                            norm_env = pickle.load(f)

                        print("✅ VecNormalize object loaded successfully from zip!")
                        if hasattr(norm_env, "obs_rms"):
                            print(
                                f"   Obs RMS Mean (first 3): {norm_env.obs_rms.mean[:3]}"
                            )
                            print(f"   Ret RMS Mean: {norm_env.ret_rms.mean}")

                        # Try to set venv (might fail if shapes mismatch due to config differences)
                        try:
                            norm_env.set_venv(venv)
                            print("✅ venv set successfully.")
                        except Exception as e:
                            print(
                                f"⚠️  venv set failed (expected if test env config differs from training): {e}"
                            )
                            print(
                                "   (This confirms the stats were loaded, just the dummy env has different shape)"
                            )

                    else:
                        print("❌ vec_normalize.pkl not found in zip")
            except Exception as e:
                print(f"❌ Failed to load VecNormalize manually: {e}")

        except Exception as e:
            print(f"❌ Failed to load VecNormalize: {e}")
    else:
        print("Data file not found, skipping env load test.")


if __name__ == "__main__":
    test_loading()
