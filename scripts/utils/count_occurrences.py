path = r"c:\Users\Admin\dev\zaif-trade-bot\logs\temp_training_1000_run.log"
with open(path, "rb") as f:
    data = f.read().decode("utf-8", "replace")
print("Occurrences (exact):", data.count("SAC continuous action:"))
print("Occurrences (lower):", data.lower().count("sac continuous action:"))
idx = data.find("SAC continuous action:")
print("first idx:", idx)
if idx != -1:
    print(data[idx : idx + 200])
