from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


JudgmentVerdict = Literal["pass", "fail", "insufficient"]


@dataclass(frozen=True, slots=True)
class CriterionAssessment:
    name: str
    verdict: JudgmentVerdict
    value: float | None
    threshold: float | None
    detail: str


def assess_fill_rate(
    *,
    variant_fill_rate: float,
    control_fill_rate: float,
    fill_rate_min: float,
    fill_rate_degradation_tolerance: float,
) -> CriterionAssessment:
    verdict: JudgmentVerdict = "pass"
    detail_parts: list[str] = []
    if variant_fill_rate < fill_rate_min:
        verdict = "fail"
        detail_parts.append(f"below absolute min {fill_rate_min:.1%}")
    if (
        control_fill_rate > 0
        and variant_fill_rate < control_fill_rate * (1 - fill_rate_degradation_tolerance)
    ):
        verdict = "fail"
        degradation = (control_fill_rate - variant_fill_rate) / control_fill_rate
        detail_parts.append(f"degraded {degradation:.1%} vs control")
    if not detail_parts:
        detail_parts.append("OK")
    return CriterionAssessment(
        name="fill_rate",
        verdict=verdict,
        value=variant_fill_rate,
        threshold=fill_rate_min,
        detail=(
            f"variant={variant_fill_rate:.1%} control={control_fill_rate:.1%} — "
            f"{'; '.join(detail_parts)}"
        ),
    )


def assess_avg_pnl30(
    *,
    variant_avg_pnl30_bps: float,
    control_avg_pnl30_bps: float,
    avg_pnl30_min_bps: float,
    avg_pnl30_must_improve: bool,
) -> CriterionAssessment:
    verdict: JudgmentVerdict = "pass"
    detail_parts: list[str] = []
    if variant_avg_pnl30_bps < avg_pnl30_min_bps:
        verdict = "fail"
        detail_parts.append(f"below absolute min {avg_pnl30_min_bps:+.2f}")
    if avg_pnl30_must_improve and variant_avg_pnl30_bps < control_avg_pnl30_bps:
        verdict = "fail"
        detail_parts.append(f"no improvement vs control ({control_avg_pnl30_bps:+.4f})")
    if not detail_parts:
        detail_parts.append("OK")
    return CriterionAssessment(
        name="avg_pnl30",
        verdict=verdict,
        value=variant_avg_pnl30_bps,
        threshold=avg_pnl30_min_bps,
        detail=(
            f"variant={variant_avg_pnl30_bps:+.4f} control={control_avg_pnl30_bps:+.4f} "
            f"bps — {'; '.join(detail_parts)}"
        ),
    )


def assess_downside_p10(
    *,
    variant_downside_p10_bps: float,
    control_downside_p10_bps: float,
    downside_p10_min_bps: float,
    downside_p10_degradation_max_bps: float,
) -> CriterionAssessment:
    verdict: JudgmentVerdict = "pass"
    detail_parts: list[str] = []
    if variant_downside_p10_bps < downside_p10_min_bps:
        verdict = "fail"
        detail_parts.append(f"below absolute min {downside_p10_min_bps:+.2f}")
    degradation_bps = control_downside_p10_bps - variant_downside_p10_bps
    if degradation_bps > downside_p10_degradation_max_bps:
        verdict = "fail"
        detail_parts.append(
            "degraded "
            f"{degradation_bps:+.2f}bps vs control "
            f"(max {downside_p10_degradation_max_bps:+.2f})"
        )
    if not detail_parts:
        detail_parts.append("OK")
    return CriterionAssessment(
        name="downside_p10",
        verdict=verdict,
        value=variant_downside_p10_bps,
        threshold=downside_p10_min_bps,
        detail=(
            f"variant={variant_downside_p10_bps:+.4f} "
            f"control={control_downside_p10_bps:+.4f} "
            f"bps — {'; '.join(detail_parts)}"
        ),
    )


def combine_assessment_verdicts(
    assessments: list[CriterionAssessment],
) -> JudgmentVerdict:
    verdicts = [assessment.verdict for assessment in assessments]
    if any(verdict == "fail" for verdict in verdicts):
        return "fail"
    if any(verdict == "insufficient" for verdict in verdicts):
        return "insufficient"
    return "pass"


def build_insufficient_assessment(
    *,
    name: str,
    value: float | None,
    threshold: float | None,
    detail: str,
) -> CriterionAssessment:
    """Build a canonical insufficient assessment payload."""
    return CriterionAssessment(
        name=name,
        verdict="insufficient",
        value=value,
        threshold=threshold,
        detail=detail,
    )
