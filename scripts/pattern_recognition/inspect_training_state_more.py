from pprint import pprint

pkl_path = r"c:\Users\Admin\dev\zaif-trade-bot\models\training_states\training_state_1000_20251101_222846.pkl"
print("Path:", pkl_path)
with open(pkl_path, "rb") as f:
    header = f.read(128)
    print("Header bytes:", header[:64])
    print("Header repr:", repr(header[:64]))

# try different loaders
try:
    import joblib

    print("\nTrying joblib.load...")
    obj = joblib.load(pkl_path)
    print("joblib.load ok, type:", type(obj))
    try:
        pprint(list(obj.keys()))
    except Exception:
        pass
except Exception as e:
    print("joblib failed:", type(e), e)

try:
    import cloudpickle

    print("\nTrying cloudpickle.load...")
    with open(pkl_path, "rb") as f:
        obj = cloudpickle.load(f)
    print("cloudpickle ok, type:", type(obj))
except Exception as e:
    print("cloudpickle failed:", type(e), e)

print("\nDone")
