from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ztb.metrics.fill_gate_reports import (
    build_g1_1_exec_judgment,
    build_g1_1_quick_judgment,
    build_g1_2_full_judgment,
)

if TYPE_CHECKING:
    from ztb.metrics.fill_metric_results import FillMetrics
    from ztb.metrics.fill_quality import FillRecord


def g1_1_judgment(
    metrics: "FillMetrics",
    thresholds: Mapping[str, object],
    records: list["FillRecord"] | None = None,
) -> dict[str, object]:
    """G1.1 Gate 合否判定."""
    return build_g1_1_exec_judgment(
        metrics=metrics,
        thresholds=thresholds,
        records=records,
    )


def g1_1_quick_judgment(
    metrics: "FillMetrics",
    thresholds: Mapping[str, object],
    cumulative_loss_jpy: float = 0.0,
) -> dict[str, object]:
    """G1.1-quick (72h Kill Gate) 判定."""
    return build_g1_1_quick_judgment(
        metrics=metrics,
        thresholds=thresholds,
        cumulative_loss_jpy=cumulative_loss_jpy,
    )


def g1_2_full_judgment(
    metrics: "FillMetrics",
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """G1.2-full (168h Qualification Gate) 判定."""
    return build_g1_2_full_judgment(
        metrics=metrics,
        thresholds=thresholds,
    )


__all__ = [
    "g1_1_judgment",
    "g1_1_quick_judgment",
    "g1_2_full_judgment",
]
