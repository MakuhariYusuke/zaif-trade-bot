import os
import re


def fix_files() -> None:
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Replace pd.Series with pd.Series
                    new_content = re.sub(r"pd\.Series\[float\]", "pd.Series", content)

                    if new_content != content:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"Fixed {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    fix_files()
