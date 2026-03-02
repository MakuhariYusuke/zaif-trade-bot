#!/usr/bin/env python3
"""
Metrics collection mixin for training classes.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

class MetricsCollectionMixin:
    """Mixin class providing common metrics collection functionality."""

    def __init__(self):
        self.metrics_csv_path: Path | None = None
        self.metrics_csv_writer: Any | None = None
        self.metrics_csv_file: Any | None = None

    def initialize_metrics_collection(
        self, output_dir: str = "results", filename_prefix: str = "training_metrics"
    ) -> None:
        """Initialize metrics collection with CSV output."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            self.metrics_csv_path = Path(output_dir) / csv_filename

            self.metrics_csv_file = open(
                self.metrics_csv_path, "w", newline="", encoding="utf-8"
            )
            self.metrics_csv_writer = csv.writer(self.metrics_csv_file)

            # Write header - this should be overridden by subclasses for specific metrics
            self.metrics_csv_writer.writerow(["timestamp", "step", "custom_metrics..."])

            self.logger.info(f"Initialized metrics collection: {self.metrics_csv_path}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize metrics collection: {e}")
            self.metrics_csv_writer = None
            self.metrics_csv_file = None

    def log_metrics_to_csv(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics to CSV file."""
        if self.metrics_csv_writer is None:
            return

        try:
            timestamp = datetime.now().isoformat()
            row = [timestamp, step]

            # Add metrics in consistent order - subclasses should override for specific format
            for key, value in sorted(metrics.items()):
                row.append(value)

            self.metrics_csv_writer.writerow(row)
            if self.metrics_csv_file:
                self.metrics_csv_file.flush()
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to CSV: {e}")

    def cleanup_metrics_collection(self) -> None:
        """Clean up metrics collection resources."""
        if self.metrics_csv_file:
            try:
                self.metrics_csv_file.close()
            except Exception as e:
                self.logger.warning(f"Failed to close metrics CSV file: {e}")

    @property
    def logger(self):
        """Get logger - should be implemented by classes using this mixin."""
        raise NotImplementedError(
            "Classes using MetricsCollectionMixin must provide a logger property"
        )
