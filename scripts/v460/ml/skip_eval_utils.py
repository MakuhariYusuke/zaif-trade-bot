"""Skip simulation helpers for training / deploy / retrain paths.

PnL 改善評価の percentile ロジックを一元化し、
NaN-only / empty keep の退避を共通化する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SkipSliceStats:
    """単一 skip percentile の評価結果."""

    threshold: float
    n_keep: int
    baseline_pnl30: float
    baseline_pnl120: float
    kept_pnl30: float
    kept_pnl120: float
    pnl30_improvement: float
    pnl120_improvement: float
    keep_mask: np.ndarray


def safe_finite_mean(values: np.ndarray) -> float:
    """有限値のみ平均し、空集合では 0.0 を返す."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def compute_skip_slice_metrics(
    scores: np.ndarray,
    pnl30: np.ndarray,
    pnl120: np.ndarray,
    *,
    skip_pct: float,
    skip_low_scores: bool,
    require_pnl30_for_selection: bool = False,
) -> SkipSliceStats:
    """percentile ベースの skip 評価を返す.

    Args:
        scores: スキップ順位付けスコア.
        pnl30: 30s PnL 配列.
        pnl120: 120s PnL 配列.
        skip_pct: 除外する割合 (例: 20 -> 上位/下位20%を skip).
        skip_low_scores:
            True なら低スコア側を skip し、高スコアを keep.
            False なら高スコア側を skip し、低スコアを keep.
        require_pnl30_for_selection:
            True なら PnL30 が有限な行だけで閾値を計算する.
    """
    score_arr = np.asarray(scores, dtype=np.float64)
    pnl30_arr = np.asarray(pnl30, dtype=np.float64)
    pnl120_arr = np.asarray(pnl120, dtype=np.float64)
    empty_keep = np.zeros(score_arr.shape, dtype=bool)

    valid = np.isfinite(score_arr)
    if require_pnl30_for_selection:
        valid &= np.isfinite(pnl30_arr)

    if not np.any(valid):
        return SkipSliceStats(
            threshold=0.0,
            n_keep=0,
            baseline_pnl30=0.0,
            baseline_pnl120=0.0,
            kept_pnl30=0.0,
            kept_pnl120=0.0,
            pnl30_improvement=0.0,
            pnl120_improvement=0.0,
            keep_mask=empty_keep,
        )

    scores_eval = score_arr[valid]
    pnl30_eval = pnl30_arr[valid]
    pnl120_eval = pnl120_arr[valid]

    baseline_30 = safe_finite_mean(pnl30_eval)
    baseline_120 = safe_finite_mean(pnl120_eval)

    clamped_skip_pct = min(max(float(skip_pct), 0.0), 100.0)
    percentile = clamped_skip_pct if skip_low_scores else 100.0 - clamped_skip_pct
    threshold = float(np.percentile(scores_eval, percentile))

    keep_eval = scores_eval >= threshold if skip_low_scores else scores_eval < threshold
    full_keep = np.zeros(score_arr.shape, dtype=bool)
    full_keep[valid] = keep_eval

    if not np.any(keep_eval):
        kept_30 = baseline_30
        kept_120 = baseline_120
    else:
        kept_30 = safe_finite_mean(pnl30_eval[keep_eval])
        kept_120 = safe_finite_mean(pnl120_eval[keep_eval])

    return SkipSliceStats(
        threshold=threshold,
        n_keep=int(np.count_nonzero(keep_eval)),
        baseline_pnl30=baseline_30,
        baseline_pnl120=baseline_120,
        kept_pnl30=kept_30,
        kept_pnl120=kept_120,
        pnl30_improvement=kept_30 - baseline_30,
        pnl120_improvement=kept_120 - baseline_120,
        keep_mask=full_keep,
    )
