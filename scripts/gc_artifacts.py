#!/usr/bin/env python3
"""
Checkpoint Garbage Collection (GC) for 1M Long-Run Training.

This script manages checkpoint lifecycle to prevent disk space exhaustion
during long training runs (1M steps = ~40 checkpoints @ 25k interval).

Retention Policy:
  - Keep last N checkpoints (default: 4 = last 100k steps)
  - Keep best M checkpoints by Sharpe ratio (default: 3)
  - Delete checkpoints older than TTL days (default: 14)

Usage:
    # Dry-run mode (show what would be deleted)
    python scripts/gc_artifacts.py --checkpoint-dir checkpoints/ensemble_C_1M --dry-run

    # Execute cleanup
    python scripts/gc_artifacts.py --checkpoint-dir checkpoints/ensemble_C_1M

    # Custom retention policy
    python scripts/gc_artifacts.py --checkpoint-dir checkpoints/ensemble_C_1M \
        --keep-last 6 --keep-best 5 --ttl-days 21

Integration with PPO Trainer:
    The GC logic can be called after each checkpoint save via callbacks.
    See ztb/training/callbacks.py for integration example.

Design Principles:
    - Conservative: Never delete if unsure (keep on error)
    - Observable: Log all deletion decisions
    - Flexible: Configurable retention thresholds
    - Testable: Dry-run mode for validation
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.training.ppo_config import CHECKPOINT_INTERVAL

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint."""

    path: Path
    step: int
    timestamp: datetime
    sharpe_ratio: Optional[float] = None
    size_bytes: int = 0
    manifest: Optional[Dict[str, Any]] = None
    keep_reason: List[str] = field(default_factory=list)

    @property
    def age_days(self) -> float:
        """Age in days from now."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 86400

    @property
    def is_marked_for_keep(self) -> bool:
        """Whether this checkpoint has any keep reason."""
        return len(self.keep_reason) > 0

    def add_keep_reason(self, reason: str) -> None:
        """Add a reason to keep this checkpoint."""
        if reason not in self.keep_reason:
            self.keep_reason.append(reason)


class CheckpointGarbageCollector:
    """
    Garbage collector for training checkpoints.

    Retention Logic:
      1. Mark "keep_last" most recent checkpoints
      2. Mark "keep_best" highest Sharpe ratio checkpoints
      3. Mark checkpoints within TTL days
      4. Delete all unmarked checkpoints (conservatively)
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 4,
        keep_best: int = 3,
        ttl_days: int = 14,
        dry_run: bool = False,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.ttl_days = ttl_days
        self.dry_run = dry_run

        # Validate inputs
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1, got {keep_last}")
        if keep_best < 0:
            raise ValueError(f"keep_best must be >= 0, got {keep_best}")
        if ttl_days < 0:
            raise ValueError(f"ttl_days must be >= 0, got {ttl_days}")

        logger.info(
            f"Initialized CheckpointGC: keep_last={keep_last}, "
            f"keep_best={keep_best}, ttl_days={ttl_days}, dry_run={dry_run}"
        )

    def scan_checkpoints(self) -> List[CheckpointInfo]:
        """
        Scan checkpoint directory and extract metadata.

        Returns:
            List of CheckpointInfo sorted by step (ascending)
        """
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return []

        checkpoints: List[CheckpointInfo] = []

        # Scan for checkpoint files (pattern: checkpoint_STEP.zip or checkpoint_STEP/)
        for path in self.checkpoint_dir.iterdir():
            if path.name.startswith("checkpoint_"):
                info = self._extract_checkpoint_info(path)
                if info:
                    checkpoints.append(info)

        # Sort by step (ascending)
        checkpoints.sort(key=lambda x: x.step)

        logger.info(f"Found {len(checkpoints)} checkpoints in {self.checkpoint_dir}")
        return checkpoints

    def _extract_checkpoint_info(self, path: Path) -> Optional[CheckpointInfo]:
        """
        Extract metadata from checkpoint path.

        Supports:
          - checkpoint_25000.zip (compressed)
          - checkpoint_25000/ (directory with manifest.json)
        """
        try:
            # Extract step number from path name
            name = path.stem if path.suffix == ".zip" else path.name
            step_str = name.replace("checkpoint_", "")
            step = int(step_str)

            # Get file/directory metadata
            if path.is_file():
                size_bytes = path.stat().st_size
                timestamp = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                )
            elif path.is_dir():
                # Sum all files in directory
                size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                timestamp = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                )
            else:
                return None

            # Try to load manifest.json for Sharpe ratio
            manifest_path = path / "manifest.json" if path.is_dir() else None
            manifest = None
            sharpe_ratio = None

            if manifest_path and manifest_path.exists():
                try:
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                        sharpe_ratio = manifest.get("metrics", {}).get("sharpe_ratio")
                except Exception as e:
                    logger.warning(f"Failed to load manifest from {manifest_path}: {e}")

            return CheckpointInfo(
                path=path,
                step=step,
                timestamp=timestamp,
                sharpe_ratio=sharpe_ratio,
                size_bytes=size_bytes,
                manifest=manifest,
            )

        except (ValueError, OSError) as e:
            logger.warning(f"Failed to parse checkpoint path {path}: {e}")
            return None

    def mark_keepers(self, checkpoints: List[CheckpointInfo]) -> None:
        """
        Mark checkpoints to keep based on retention policy.

        Modifies checkpoints in-place by adding keep_reason.
        """
        if not checkpoints:
            return

        # 1. Mark last N checkpoints
        for info in checkpoints[-self.keep_last :]:
            info.add_keep_reason(f"last_{self.keep_last}")

        # 2. Mark best M by Sharpe ratio
        if self.keep_best > 0:
            # Filter checkpoints with valid Sharpe ratio
            with_sharpe = [c for c in checkpoints if c.sharpe_ratio is not None]
            # Sort by Sharpe ratio (descending)
            with_sharpe.sort(key=lambda x: x.sharpe_ratio or float("-inf"), reverse=True)
            # Mark top M
            for info in with_sharpe[: self.keep_best]:
                info.add_keep_reason(f"top_{self.keep_best}_sharpe")

        # 3. Mark checkpoints within TTL
        if self.ttl_days > 0:
            for info in checkpoints:
                if info.age_days <= self.ttl_days:
                    info.add_keep_reason(f"within_ttl_{self.ttl_days}d")

        # Log summary
        keep_count = sum(1 for c in checkpoints if c.is_marked_for_keep)
        delete_count = len(checkpoints) - keep_count
        logger.info(
            f"Retention policy: {keep_count} to keep, {delete_count} to delete"
        )

    def execute_cleanup(self, checkpoints: List[CheckpointInfo]) -> Tuple[int, int, int]:
        """
        Delete unmarked checkpoints.

        Returns:
            (deleted_count, kept_count, bytes_freed)
        """
        deleted_count = 0
        kept_count = 0
        bytes_freed = 0

        for info in checkpoints:
            if info.is_marked_for_keep:
                kept_count += 1
                logger.debug(
                    f"KEEP: {info.path.name} (step={info.step}, "
                    f"reasons={info.keep_reason})"
                )
            else:
                # Delete checkpoint
                if self.dry_run:
                    logger.info(
                        f"[DRY-RUN] Would delete: {info.path.name} "
                        f"(step={info.step}, age={info.age_days:.1f}d, "
                        f"size={info.size_bytes / 1024 / 1024:.1f}MB)"
                    )
                else:
                    try:
                        self._delete_checkpoint(info.path)
                        logger.info(
                            f"Deleted: {info.path.name} "
                            f"(step={info.step}, age={info.age_days:.1f}d, "
                            f"size={info.size_bytes / 1024 / 1024:.1f}MB)"
                        )
                        deleted_count += 1
                        bytes_freed += info.size_bytes
                    except Exception as e:
                        logger.error(f"Failed to delete {info.path}: {e}")

        return deleted_count, kept_count, bytes_freed

    def _delete_checkpoint(self, path: Path) -> None:
        """Delete checkpoint file or directory."""
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            # Recursively delete directory
            import shutil

            shutil.rmtree(path)
        else:
            raise ValueError(f"Unknown checkpoint type: {path}")

    def run(self) -> None:
        """Execute full GC cycle: scan -> mark -> delete."""
        logger.info("=" * 80)
        logger.info("Starting Checkpoint Garbage Collection")
        logger.info(f"Directory: {self.checkpoint_dir}")
        logger.info(f"Policy: keep_last={self.keep_last}, keep_best={self.keep_best}, ttl_days={self.ttl_days}")
        logger.info(f"Mode: {'DRY-RUN' if self.dry_run else 'EXECUTE'}")
        logger.info("=" * 80)

        # 1. Scan
        checkpoints = self.scan_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found. Nothing to do.")
            return

        # 2. Mark
        self.mark_keepers(checkpoints)

        # 3. Delete
        deleted, kept, bytes_freed = self.execute_cleanup(checkpoints)

        # 4. Summary
        logger.info("=" * 80)
        logger.info("Checkpoint GC Summary:")
        logger.info(f"  Total checkpoints: {len(checkpoints)}")
        logger.info(f"  Kept: {kept}")
        logger.info(f"  Deleted: {deleted}")
        logger.info(f"  Bytes freed: {bytes_freed / 1024 / 1024:.1f} MB")
        if self.dry_run:
            logger.info("  (DRY-RUN mode - no files were actually deleted)")
        logger.info("=" * 80)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Checkpoint Garbage Collector for 1M Long-Run Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=4,
        help="Keep last N checkpoints (default: 4 = 100k steps @ 25k interval)",
    )
    parser.add_argument(
        "--keep-best",
        type=int,
        default=3,
        help="Keep best M checkpoints by Sharpe ratio (default: 3)",
    )
    parser.add_argument(
        "--ttl-days",
        type=int,
        default=14,
        help="Keep checkpoints newer than N days (default: 14)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run GC
    gc = CheckpointGarbageCollector(
        checkpoint_dir=args.checkpoint_dir,
        keep_last=args.keep_last,
        keep_best=args.keep_best,
        ttl_days=args.ttl_days,
        dry_run=args.dry_run,
    )
    gc.run()


if __name__ == "__main__":
    main()
