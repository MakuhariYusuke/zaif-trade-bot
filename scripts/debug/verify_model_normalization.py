import pickle
import zipfile

from stable_baselines3 import SAC

model_path = "models/sac_model.zip"

print(f"Checking {model_path}...")

# 1. Check zip content
try:
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Files in zip: {file_list}")
        if "vec_normalize.pkl" in file_list:
            print("✅ vec_normalize.pkl found in zip")
        else:
            print("❌ vec_normalize.pkl NOT found in zip")
except Exception as e:
    print(f"Error inspecting zip: {e}")

# 2. Try to load the model and inspect
try:
    # We need a dummy env to load the model, but we are interested in the file content mostly
    # However, SB3 loads vec_normalize if it's in the zip
    model = SAC.load(model_path)
    print("✅ Model loaded successfully")

    # Check if we can extract the stats manually from the file
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        if "vec_normalize.pkl" in zip_ref.namelist():
            with zip_ref.open("vec_normalize.pkl") as f:
                stats = pickle.load(f)
                print("Stats loaded from pickle:")
                # print(stats)
                if hasattr(stats, "obs_rms"):
                    print(f"  obs_rms.mean: {stats.obs_rms.mean[:5]}...")
                    print(f"  obs_rms.var: {stats.obs_rms.var[:5]}...")
                if hasattr(stats, "ret_rms"):
                    print(f"  ret_rms.mean: {stats.ret_rms.mean}")
                    print(f"  ret_rms.var: {stats.ret_rms.var}")

except Exception as e:
    print(f"Error loading model: {e}")
