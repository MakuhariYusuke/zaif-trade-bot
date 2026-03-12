"""Minimal dotenv shim to support tests that import `dotenv.load_dotenv`.
This is a lightweight compatibility shim and not a full replacement for `python-dotenv`.
It attempts to load environment variables from a `.env` file in the current working
directory if present. If not present, it returns False.
"""
import os
from pathlib import Path


def load_dotenv(dotenv_path: str | None = None) -> bool:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    Returns True if a file was found and loaded, False otherwise.
    """
    path = Path(dotenv_path or ".env")
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
        return True
    except Exception:
        return False


__all__ = ["load_dotenv"]
