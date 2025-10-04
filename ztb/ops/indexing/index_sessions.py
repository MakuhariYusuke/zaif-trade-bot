#!/usr/bin/env python3
"""
Session indexer for Zaif Trade Bot.

Builds index of existing sessions for discoverability.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


def get_session_status(corr_dir: Path) -> str:
    """Determine session status."""
    stop_file = corr_dir / "ztb.stop"
    if stop_file.exists():
        return "stopped"

    metrics_path = corr_dir / "metrics.json"
    if metrics_path.exists():
        try:
            mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(minutes=10):
                return "running"
            elif datetime.now() - mtime > timedelta(minutes=30):
                return "stopped"
        except Exception:
            pass

    return "unknown"


def index_sessions(root: Path) -> Dict[str, Any]:
    """Index sessions."""
    sessions: List[Dict[str, Any]] = []

    for item in root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            corr_id = item.name
            metadata_path = item / "run_metadata.json"
            summary_path = item / "summary.json"
            best_marker = item / "best.marker"

            session: Dict[str, Any] = {
                "correlation_id": corr_id,
                "created_at": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                "size_mb": sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                / (1024 * 1024),
                "latest_step": 0,
                "status": get_session_status(item),
                "is_best": best_marker.exists(),
                "metrics": {},
                "paths": {},
            }

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    session["paths"]["metadata"] = str(metadata_path)
                except Exception:
                    pass

            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    session["latest_step"] = summary.get("summary", {}).get(
                        "global_step", 0
                    )
                    # Extract key metrics
                    if "summary" in summary:
                        session["metrics"] = {
                            "loss": summary["summary"].get("loss"),
                            "accuracy": summary["summary"].get("accuracy"),
                            "val_loss": summary["summary"].get("val_loss"),
                            "val_accuracy": summary["summary"].get("val_accuracy"),
                        }
                    session["paths"]["summary"] = str(summary_path)
                except Exception:
                    pass

            sessions.append(session)

    # Sort by modified_at descending (newest first)
    sessions.sort(key=lambda s: str(s["modified_at"]), reverse=True)
    latest_session = sessions[0]["correlation_id"] if sessions else None

    # Find best sessions
    best_sessions = [s["correlation_id"] for s in sessions if s["is_best"]]

    return {
        "generated_at": datetime.now().isoformat(),
        "latest": latest_session,
        "best": best_sessions,
        "sessions": sessions,
    }


def main() -> int:
    root = Path("artifacts")
    index = index_sessions(root)

    index_path = root / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Indexed {len(index['sessions'])} sessions to {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
