"""Evaluation logging helpers used by feature evaluation flows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


class EvaluationLogger:
    """Append evaluation results to JSONL for offline inspection."""

    def __init__(self, output_path: str | Path = "logs/evaluation_results.jsonl") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_evaluation(self, payload: Mapping[str, object]) -> None:
        record: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        record.update(payload)
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Logging must be non-fatal in analysis paths.
            return
