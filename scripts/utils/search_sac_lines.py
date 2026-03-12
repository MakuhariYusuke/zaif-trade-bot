path = r"c:\Users\Admin\dev\zaif-trade-bot\logs\temp_training_1000_run.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16")
lines = text.splitlines()
matches = [(i + 1, l) for i, l in enumerate(lines) if "SAC continuous action" in l]
print("Found", len(matches), "lines with SAC continuous action")
for m in matches[:10]:
    print(m[0], m[1])
# print sample around first match
if matches:
    idx = matches[0][0]
    for i in range(idx - 3, idx + 3):
        if 1 <= i <= len(lines):
            print(i, lines[i - 1])
