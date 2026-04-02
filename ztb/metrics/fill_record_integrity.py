from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ztb.trading.common.cancel_reasons import AUDIT_CANCEL_REASONS
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class IntegrityFillRecord(Protocol):
    timestamp: float
    run_id: str | None
    pid: int | None
    git_sha: str | None
    side: str
    order_price: float | None
    order_quantity: float | None
    cancel_reason: str | None


def detect_split_brain(
    records: Iterable[IntegrityFillRecord],
    *,
    overlap_window_sec: float = 300.0,
) -> list[dict[str, object]]:
    """Detect overlapping writers that suggest split-brain execution."""
    materialized = list(records)
    if len(materialized) < 2:
        return []

    events: list[dict[str, object]] = []
    for i in range(1, len(materialized)):
        prev, curr = materialized[i - 1], materialized[i]
        if prev.run_id and curr.run_id and prev.run_id != curr.run_id:
            gap = abs(curr.timestamp - prev.timestamp)
            if gap <= overlap_window_sec:
                events.append(
                    {
                        "timestamp": curr.timestamp,
                        "run_id_a": prev.run_id,
                        "run_id_b": curr.run_id,
                        "pid_a": prev.pid,
                        "pid_b": curr.pid,
                        "gap_sec": gap,
                    }
                )
        elif (
            prev.pid is not None
            and curr.pid is not None
            and prev.pid != curr.pid
            and prev.run_id == curr.run_id
        ):
            gap = abs(curr.timestamp - prev.timestamp)
            if gap <= overlap_window_sec:
                events.append(
                    {
                        "timestamp": curr.timestamp,
                        "run_id_a": prev.run_id,
                        "run_id_b": curr.run_id,
                        "pid_a": prev.pid,
                        "pid_b": curr.pid,
                        "gap_sec": gap,
                    }
                )

    if events:
        logger.critical(
            "[286# SPLIT-BRAIN] %s overlapping process events detected! "
            "Multiple processes wrote to the same JSONL. First event: "
            "run_ids=%s/%s, pids=%s/%s",
            len(events),
            events[0].get("run_id_a"),
            events[0].get("run_id_b"),
            events[0].get("pid_a"),
            events[0].get("pid_b"),
        )
    return events


def quarantine_reason(record: IntegrityFillRecord) -> str | None:
    """Return quarantine reason for a fill record, or None when clean."""
    if not (record.git_sha and record.git_sha.strip()):
        return "blank_git_sha"
    if not (record.run_id and record.run_id.strip()):
        return "blank_run_id"

    is_audit = (
        record.cancel_reason in AUDIT_CANCEL_REASONS
        and record.side in ("none", "buy", "sell")
    )
    if record.side not in ("buy", "sell"):
        if is_audit:
            return None
        return f"invalid_side={record.side}"
    if (not record.order_price or record.order_price <= 0) and not is_audit:
        return "invalid_order_price"
    if (not record.order_quantity or record.order_quantity <= 0) and not is_audit:
        return "invalid_order_quantity"
    return None


def partition_clean_records(
    records: Iterable[IntegrityFillRecord],
    *,
    require_git_sha: bool = True,
) -> tuple[list[IntegrityFillRecord], list[IntegrityFillRecord]]:
    """Split records into clean and quarantine buckets."""
    if not require_git_sha:
        if isinstance(records, list):
            return records, []
        return list(records), []

    clean: list[IntegrityFillRecord] = []
    quarantine: list[IntegrityFillRecord] = []
    total = 0
    for record in records:
        total += 1
        reason = quarantine_reason(record)
        if reason:
            quarantine.append(record)
        else:
            clean.append(record)

    if quarantine:
        logger.info(
            "[quarantine] %s/%s records quarantined. clean=%s",
            len(quarantine),
            total,
            len(clean),
        )
    return clean, quarantine


def filter_clean_records(
    records: list[IntegrityFillRecord],
    *,
    require_git_sha: bool = True,
) -> tuple[list[IntegrityFillRecord], list[IntegrityFillRecord]]:
    """Backward-compatible list API for clean/quarantine partitioning."""
    return partition_clean_records(records, require_git_sha=require_git_sha)


__all__ = [
    "detect_split_brain",
    "filter_clean_records",
    "partition_clean_records",
    "quarantine_reason",
]
