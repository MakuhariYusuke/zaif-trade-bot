import os
import zipfile

model_path = "models/sac_v446_fixed.zip"

if os.path.exists(model_path):
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        print(f"Contents of {model_path}:")
        for file_name in zip_ref.namelist():
            print(f" - {file_name}")
else:
    print(f"File not found: {model_path}")
