path = r"c:\Users\Admin\dev\zaif-trade-bot\logs\temp_training_1000_run.log"
with open(path, "rb") as f:
    b = f.read(2048)
print(repr(b[:1000]))
# print bytes where ASCII 'SAC' appears
for i in range(len(b) - 3):
    if b[i : i + 3] == b"SAC":
        print("found at", i, b[i - 20 : i + 40])
        break
else:
    print("No literal SAC in first 2048 bytes")
