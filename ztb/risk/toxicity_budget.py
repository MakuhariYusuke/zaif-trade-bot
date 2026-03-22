"""543# Toxicity Budget: 段階的逆選択応答の独立モジュール.

Glosten-Milgrom (1985) の情報非対称性モデルに基づく 4 段階応答。
sell_dynamic_kill.py からの kill ロジック非依存で Toxicity 評価を提供する。

Usage:
    assessment = assess_toxicity_score(
        score=0.5,
        warn_level=0.3,
        caution_level=0.7,
        warn_offset_mult=1.0,
        caution_offset_mult=2.0,
        kill_offset_mult=3.0,
        caution_min_participation=0.33,
    )
"""

from __future__ import annotations

from ztb.risk.toxicity_types import ToxicityAssessment, ToxicityLevel


def assess_toxicity_score(
    *,
    score: float,
    warn_level: float = 0.3,
    caution_level: float = 0.7,
    warn_offset_mult: float = 1.0,
    caution_offset_mult: float = 2.0,
    kill_offset_mult: float = 3.0,
    caution_min_participation: float = 0.33,
    threshold_used: float = -0.5,
    rolling_mean: float | None = None,
) -> ToxicityAssessment:
    """Score-based toxicity assessment (kill ロジック非依存).

    Args:
        score: 正規化 toxicity スコア [0, ∞)。
               0=安全, 1.0=kill 閾値。
               典型的には ``max(0, rolling_pnl_mean / threshold_bps)`` で算出。

    Returns:
        ToxicityAssessment (immutable)
    """
    if score >= 1.0:
        return ToxicityAssessment(
            level=ToxicityLevel.KILL,
            score=score,
            offset_mult=kill_offset_mult,
            participation_rate=0.0,
            threshold_used=threshold_used,
            rolling_mean=rolling_mean,
        )

    if score >= caution_level:
        t = (score - caution_level) / (1.0 - caution_level) if caution_level < 1.0 else 0.0
        offset_m = caution_offset_mult + t * (kill_offset_mult - caution_offset_mult)
        participation = 1.0 - t * (1.0 - caution_min_participation)
        return ToxicityAssessment(
            level=ToxicityLevel.ORANGE,
            score=score,
            offset_mult=offset_m,
            participation_rate=participation,
            threshold_used=threshold_used,
            rolling_mean=rolling_mean,
        )

    if score >= warn_level:
        t = (score - warn_level) / (caution_level - warn_level) if caution_level > warn_level else 0.0
        offset_m = warn_offset_mult + t * (caution_offset_mult - warn_offset_mult)
        return ToxicityAssessment(
            level=ToxicityLevel.YELLOW,
            score=score,
            offset_mult=offset_m,
            participation_rate=1.0,
            threshold_used=threshold_used,
            rolling_mean=rolling_mean,
        )

    return ToxicityAssessment(
        level=ToxicityLevel.GREEN,
        score=score,
        offset_mult=1.0,
        participation_rate=1.0,
        threshold_used=threshold_used,
        rolling_mean=rolling_mean,
    )
