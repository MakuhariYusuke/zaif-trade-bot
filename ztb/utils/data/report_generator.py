"""
ReportGenerator: Unified report generation for experiments and quality gates.

Outputs in CSV/JSON/Markdown formats, saves QualityGates and ExperimentResult in standard formats.

Usage:
    from ztb.utils.report_generator import ReportGenerator

    generator = ReportGenerator()
    generator.generate_csv(results, "report.csv")
    generator.generate_markdown(results, "report.md")
"""

import json
import csv
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class ReportGenerator:
    """Unified report generator"""

    def __init__(self) -> None:
        pass

    def generate_csv(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """Generate CSV report"""
        if not results:
            return

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Get all unique keys
        all_keys: set[str] = set()
        for result in results:
            all_keys.update(result.keys())

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def generate_json(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """Generate JSON report"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=2, ensure_ascii=False, default=str)

    def generate_markdown(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """Generate Markdown report"""
        if not results:
            return

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Get all unique keys
        all_keys: set[str] = set()
        for result in results:
            all_keys.update(result.keys())

        with open(file_path, 'w', encoding='utf-8') as mdfile:
            mdfile.write("# Report\n\n")
            mdfile.write(f"Total results: {len(results)}\n\n")

            if results:
                # Table header
                mdfile.write("| " + " | ".join(sorted(all_keys)) + " |\n")
                mdfile.write("| " + " | ".join(["---"] * len(all_keys)) + " |\n")

                # Table rows
                for result in results:
                    row = []
                    for key in sorted(all_keys):
                        value = result.get(key, "")
                        # Format floats
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    mdfile.write("| " + " | ".join(row) + " |\n")

    def save_experiment_dump(self, experiment_id: str, config: Dict[str, Any], error: Optional[Exception] = None) -> None:
        """Save minimal experiment dump on failure"""
        dump_dir = Path("logs/dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)

        dump_data = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "error": str(error) if error else None,
            "traceback": traceback.format_exc() if error else None
        }

        dump_file = dump_dir / f"dump-{experiment_id}.json"
        with open(dump_file, 'w', encoding='utf-8') as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False, default=str)