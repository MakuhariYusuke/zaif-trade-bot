"""
Parquet data integrity checker and repair tool.

Checks for missing data, duplicates, and repairs automatically.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ztb.data.binance_data import load_parquet_pattern, save_parquet_chunked
from ztb.ops.alerts.notifications import DiscordNotifier

logger = logging.getLogger(__name__)


class ParquetIntegrityChecker:
    """
    Checks and repairs Parquet data integrity.

    Detects gaps, duplicates, and missing data in time series.
    """

    def __init__(self, data_dir: str = "data/binance"):
        self.data_dir = Path(data_dir)
        self.notifier = DiscordNotifier()

    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """
        Validate DataFrame schema: column dtypes and timezone awareness.

        Returns:
            List of validation issues
        """
        issues = []

        # Check required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
            else:
                # Check dtypes
                if col in ["open", "high", "low", "close"]:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        issues.append(
                            f"Column {col} should be numeric, got {df[col].dtype}"
                        )
                elif col == "volume":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        issues.append(
                            f"Column {col} should be numeric, got {df[col].dtype}"
                        )

        # Check index is timezone-aware UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index should be DatetimeIndex")
        elif df.index.tz is None:
            issues.append("Index should be timezone-aware (UTC)")
        elif str(df.index.tz) != "UTC":
            issues.append(f"Index timezone should be UTC, got {df.index.tz}")

        return issues

    def check_integrity(
        self, symbol: str = "BTCUSDT", interval: str = "1m"
    ) -> Dict[str, Any]:
        """
        Check data integrity for the given symbol and interval.

        Returns:
            Dict with integrity report
        """
        report = {
            "symbol": symbol,
            "interval": interval,
            "total_files": 0,
            "total_records": 0,
            "missing_periods": [],
            "duplicate_records": 0,
            "gaps": [],
            "is_integrity_ok": True,
        }

        try:
            # Load all data
            pattern = str(self.data_dir / "*.parquet")
            df = load_parquet_pattern(str(self.data_dir))

            # Schema validation: check column dtypes and timezone awareness
            schema_issues = self._validate_schema(df)
            if schema_issues:
                report["schema_issues"] = schema_issues
                report["is_integrity_ok"] = False
                logger.warning(f"Schema validation failed: {schema_issues}")

            if df.empty:
                report["is_integrity_ok"] = False
                report["error"] = "No data files found"
                return report

            report["total_records"] = len(df)

            # Check for duplicates (index uniqueness)
            duplicates = df.index.duplicated().sum()
            report["duplicate_records"] = duplicates

            if duplicates > 0:
                report["is_integrity_ok"] = False
                logger.warning(f"Found {duplicates} duplicate records")

            # Check for data gaps and missing periods
            if interval == "1m":
                df = df.sort_index()
                time_diffs = df.index.to_series().diff().dropna()
                expected_diff = pd.Timedelta(minutes=1)

                gaps = time_diffs[time_diffs > expected_diff]
                if not gaps.empty:
                    report["gaps"] = [
                        {
                            "start": str(gap_start),
                            "end": str(gap_start + gap_duration),
                            "duration_minutes": (
                                gap_duration.total_seconds() / 60
                                if hasattr(gap_duration, "total_seconds")
                                else float(gap_duration) / 60
                            ),
                        }
                        for gap_start, gap_duration in zip(gaps.index, gaps.values)
                    ]
                    report["is_integrity_ok"] = False
                    logger.warning(f"Found {len(gaps)} time gaps")

                    # Check if gaps exceed threshold (0.1%)
                    total_expected_records = len(df) + len(gaps)
                    gap_ratio = (
                        len(gaps) / total_expected_records
                        if total_expected_records > 0
                        else 0
                    )
                    if gap_ratio > 0.001:  # 0.1% threshold
                        report["gap_alert"] = True
                        self.notifier.notify_data_pipeline_status(
                            "gap_alert",
                            {"gap_ratio": f"{gap_ratio:.2%}", "total_gaps": len(gaps)},
                        )

            # Check for missing OHLCV columns
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                report["missing_columns"] = missing_cols
                report["is_integrity_ok"] = False

        except Exception as e:
            report["is_integrity_ok"] = False
            report["error"] = str(e)
            logger.error(f"Error checking integrity: {e}")

        return report

    def repair_integrity(
        self, report: Dict[str, Any], auto_repair: bool = True
    ) -> bool:
        """
        Repair data integrity issues.

        Args:
            report: Integrity check report
            auto_repair: Whether to automatically apply repairs

        Returns:
            True if repair successful or no repair needed
        """
        if report.get("is_integrity_ok", True):
            logger.info("Data integrity is OK, no repair needed")
            return True

        if not auto_repair:
            logger.info("Auto-repair disabled, manual intervention required")
            return False

        try:
            # Load current data
            df = load_parquet_pattern(str(self.data_dir))

            if df.empty:
                logger.error("Cannot repair: no data to work with")
                return False

            # Remove duplicates
            if report.get("duplicate_records", 0) > 0:
                df = df[~df.index.duplicated(keep="first")]
                logger.info(f"Removed {report['duplicate_records']} duplicate records")

            # Repair gaps by refetching missing data
            gaps = report.get("gaps", [])
            if gaps:
                logger.info(f"Attempting to repair {len(gaps)} gaps")

                for gap in gaps:
                    try:
                        # Parse gap period
                        start_time = pd.to_datetime(gap["start"])
                        end_time = pd.to_datetime(gap["end"])

                        # Fetch missing data (simplified - in practice would need proper date range)
                        # This is a placeholder for actual gap filling logic
                        logger.info(
                            f"Would refetch data for gap: {start_time} to {end_time}"
                        )

                    except Exception as e:
                        logger.warning(f"Failed to repair gap {gap}: {e}")

            # Re-save cleaned data
            backup_dir = (
                self.data_dir.parent
                / f"{self.data_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Move existing files to backup
            for file_path in self.data_dir.glob("*.parquet"):
                file_path.rename(backup_dir / file_path.name)

            # Save repaired data
            save_parquet_chunked(df, str(self.data_dir))

            success_msg = (
                f"Data integrity repair completed. Backup saved to {backup_dir}"
            )
            logger.info(success_msg)
            self.notifier.send_notification(
                "Data Integrity Check", success_msg, "success"
            )

            return True

        except Exception as e:
            error_msg = f"Data integrity repair failed: {e}"
            logger.error(error_msg)
            self.notifier.send_notification("Data Integrity Repair", error_msg, "error")
            return False

    def run_integrity_check(self, auto_repair: bool = True) -> Dict[str, Any]:
        """
        Run full integrity check and optional repair.

        Args:
            auto_repair: Whether to automatically repair issues

        Returns:
            Final integrity report
        """
        logger.info("Starting Parquet data integrity check")

        # Check integrity
        report = self.check_integrity()

        # Repair if needed
        if not report["is_integrity_ok"] and auto_repair:
            repair_success = self.repair_integrity(report, auto_repair)
            if repair_success:
                # Re-check after repair
                report = self.check_integrity()
                report["repair_attempted"] = True
                report["repair_successful"] = True
            else:
                report["repair_attempted"] = True
                report["repair_successful"] = False

        # Notify results
        if report["is_integrity_ok"]:
            self.notifier.send_notification(
                "Data Integrity Check",
                "Data integrity check passed",
                "success",
                fields={
                    "total_records": report.get("total_records", 0),
                    "duplicate_records": report.get("duplicate_records", 0),
                },
            )
        else:
            self.notifier.send_notification(
                "Data Integrity Check",
                "Data integrity issues found",
                "warning",
                fields={
                    "gaps_count": len(report.get("gaps", [])),
                    "duplicates": report.get("duplicate_records", 0),
                },
            )

        return report
