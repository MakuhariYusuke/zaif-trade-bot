import zipfile

model_path = "models/sac_model.zip"
try:
    with zipfile.ZipFile(model_path, "r") as zf:
        print(zf.namelist())
except Exception as e:
    print(e)
