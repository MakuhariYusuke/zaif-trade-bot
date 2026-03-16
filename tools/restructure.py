"""
Repository restructure helper (dry-run)
Scans top-level directories, reports sizes, and suggests grouping for artifacts/cache/models.
Run locally and inspect output before applying any changes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".github", "docs", "tools", "ztb", "tests", "src", "notebooks"}

EXT_GROUPS = {
    "models": [".pt", ".pth", ".h5", ".onnx", ".pkl"],
    "archives": [".zip", ".tar.gz", ".tar"],
    "binaries": [".exe", ".dll", ".so"],
}


def scan(root: Path):
    candidates = []
    for p in root.iterdir():
        if p.name in IGNORED:
            continue
        size = 0
        try:
            if p.is_file():
                size = p.stat().st_size
            else:
                for pp in p.rglob("*"):
                    if pp.is_file():
                        size += pp.stat().st_size
        except Exception:
            size = -1
        candidates.append({"path": str(p), "is_dir": p.is_dir(), "size_bytes": size})
    return candidates


def suggest_groups(candidates):
    groups = {"models": [], "checkpoints": [], "cache": [], "others": []}
    for c in candidates:
        path = c["path"]
        lower = path.lower()
        if any(ext in lower for exts in EXT_GROUPS.values() for ext in exts):
            groups["models"].append(path)
        elif (
            "cache" in lower
            or ".mypy_cache" in lower
            or ".pytest_cache" in lower
            or "cache" in path
        ):
            groups["cache"].append(path)
        elif "checkpoint" in lower or "checkpoints" in lower:
            groups["checkpoints"].append(path)
        else:
            groups["others"].append(path)
    return groups


if __name__ == "__main__":
    cand = scan(ROOT)
    groups = suggest_groups(cand)
    out = {"candidates": cand, "groups": groups}
    print(json.dumps(out, indent=2))
