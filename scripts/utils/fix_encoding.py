file_path = r"c:\Users\Admin\dev\zaif-trade-bot\ztb\features\models\sac\sac_v427_feature_engineering.py"

with open(file_path, "rb") as f:
    content = f.read()

# Remove null bytes
content = content.replace(b"\x00", b"")

# Decode
text = content.decode("utf-8", errors="ignore")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed file encoding.")
