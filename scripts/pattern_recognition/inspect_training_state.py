import pickle
import pprint

pkl_path = r"c:\Users\Admin\dev\zaif-trade-bot\models\training_states\training_state_1000_20251101_222846.pkl"
print("Loading", pkl_path)
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("\nType:", type(data))
try:
    pprint.pprint(list(data.keys()))
except Exception as e:
    print("Cannot list keys:", e)

# Try to find action entries
candidates = [
    "actions",
    "action_history",
    "discrete_actions",
    "action_distribution",
    "stats",
    "statistics",
]
found = False
for k in getattr(data, "__dict__", {}) or (
    data.keys() if hasattr(data, "keys") else []
):
    if any(ci in str(k).lower() for ci in candidates):
        print("\nCandidate key:", k)
        try:
            val = data[k]
        except Exception:
            try:
                val = getattr(data, k)
            except Exception as e:
                val = f"Error retrieving: {e}"
        print(type(val))
        try:
            if hasattr(val, "__len__"):
                print("len =", len(val))
                if len(val) > 0:
                    print("sample:", val[:5])
        except Exception:
            pprint.pprint(val)
        found = True

# Fallback: search nested dicts for arrays of length 1000
if not found and isinstance(data, dict):
    for k, v in data.items():
        try:
            if hasattr(v, "__len__") and len(v) >= 1000:
                print(
                    "\nFound long candidate key:", k, "len=", len(v), "type=", type(v)
                )
                print("sample:", v[:5])
                found = True
                break
        except Exception:
            continue

print("\nDone")
