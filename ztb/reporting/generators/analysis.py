"""
ReportGenerator: Unified report generation for experiments and quality gates.
"""

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from ztb.types.common import ConfigDict
from ztb.io.csv_io import write_csv_dicts
from ztb.io.json_io import write_json
from ztb.io.text_io import write_text
from ztb.utils.path_utils import ensure_dir

class ReportGenerator:
    """Unified report generator."""

    def generate_csv(self, results: list[dict[str, Any]], file_path: str) -> None:
        """Generate CSV report."""
        if not results:
            return
        write_csv_dicts(file_path, results)

    def generate_json(self, results: list[dict[str, Any]], file_path: str) -> None:
        """Generate JSON report."""
        write_json(file_path, results, indent=2, ensure_ascii=False, default=str)

    def generate_markdown(self, results: list[dict[str, Any]], file_path: str) -> None:
        """Generate Markdown report."""
        if not results:
            return
        ensure_dir(Path(file_path).parent)

        all_keys: set[str] = set()
        for result in results:
            all_keys.update(result.keys())

        content_lines = ["# Report", "", f"Total results: {len(results)}", ""]
        content_lines.append("| " + " | ".join(sorted(all_keys)) + " |")
        content_lines.append("| " + " | ".join(["---"] * len(all_keys)) + " |")

        for result in results:
            row = []
            for key in sorted(all_keys):
                value = result.get(key, "")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            content_lines.append("| " + " | ".join(row) + " |")

        write_text(file_path, "\n".join(content_lines))

    def save_experiment_dump(
        self,
        experiment_id: str,
        config: ConfigDict,
        error: Exception | None = None,
    ) -> None:
        """Save minimal experiment dump on failure."""
        dump_dir = Path("logs/dumps")
        ensure_dir(dump_dir)

        dump_data = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "error": str(error) if error else None,
            "traceback": traceback.format_exc() if error else None,
        }

        dump_file = dump_dir / f"dump-{experiment_id}.json"
        write_json(dump_file, dump_data, indent=2, ensure_ascii=False, default=str)
