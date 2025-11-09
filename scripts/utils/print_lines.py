path = r"c:\Users\Admin\dev\zaif-trade-bot\logs\temp_training_1000_run.log"
with open(path, "rb") as f:
    raw = f.read()
for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
    try:
        text = raw.decode(enc)
        print("--- decoded with", enc, "---")
        lines = text.splitlines()
        for i, l in enumerate(lines[:40]):
            print(i + 1, repr(l))
        break
    except Exception as e:
        print("decoding with", enc, "failed:", e)
