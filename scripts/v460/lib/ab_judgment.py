"""160# P0-B/C: A/B 判定基準の固定 + trending_down sell 実測評価.

159# §3.1/§4.1 準拠:
- sell offset A/B 判定を fill_rate + avg_pnl30 + downside_tail の3指標必須で管理
- fill_rate 単独最適化を防止するため、3指標すべての合格を要件化
- trending_down sell 効果測定の固定テンプレート (日次出力)

既存資産活用:
- ztb.adaptation.ab_test.analyzer.ABTestAnalyzer (t検定・効果量)
- scripts.v460.analysis.side_regime_dashboard._compute_side_metrics (3指標算出)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# P0-B: A/B 判定基準
# ======================================================================


class Verdict(str, Enum):
    """判定結果."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"  # サンプル不足で判定不能


@dataclass
class ABJudgmentCriteria:
    """A/B 判定の3指標閾値 (159# §3.1 / §9.3 準拠).

    YAML `judgment.ab_criteria` からロード可能。
    全指標 PASS で「variant 採用可」、1つでも FAIL なら「不採用」。
    """

    # --- 最低サンプル要件 ---
    min_filled_records: int = 50  # filled レコード最小数 (これ未満は INSUFFICIENT)
    min_calendar_days: int = 2  # 最低暦日数

    # --- 3指標閾値 ---
    # fill_rate: variant >= control × (1 - tolerance) で PASS
    fill_rate_min: float = 0.30  # 絶対下限 (30% 未満は無条件 FAIL)
    fill_rate_degradation_tolerance: float = 0.05  # control 比 5% 以上悪化で FAIL

    # avg_pnl30: variant の平均 PnL30 が下限以上で PASS
    avg_pnl30_min_bps: float = -1.0  # 絶対下限 (bps)
    avg_pnl30_must_improve: bool = False  # True: control よりも改善を要求

    # downside_tail: p10 (worst decile) が下限以上で PASS
    downside_p10_min_bps: float = -5.0  # 絶対下限 (bps)
    downside_p10_degradation_max_bps: float = 2.0  # control 比 2bps 以上悪化で FAIL

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ABJudgmentCriteria:
        """辞書から生成 (YAML 用)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CriterionResult:
    """単一指標の判定結果."""

    name: str
    verdict: Verdict
    value: float | None = None
    threshold: float | None = None
    detail: str = ""


@dataclass
class ABJudgmentResult:
    """A/B 判定の総合結果."""

    overall: Verdict
    criteria: list[CriterionResult] = field(default_factory=list)
    variant_label: str = ""
    control_label: str = ""
    n_variant: int = 0
    n_control: int = 0

    # 統計検定結果 (ztb.adaptation.ab_test.analyzer 利用)
    pnl30_p_value: float | None = None
    pnl30_effect_size: float | None = None  # Cohen's d

    def summary(self) -> str:
        """人間可読サマリ."""
        lines = [
            f"[A/B Judgment] {self.overall.value.upper()}",
            f"  variant={self.variant_label} (n={self.n_variant})"
            f" vs control={self.control_label} (n={self.n_control})",
        ]
        for c in self.criteria:
            flag = "✅" if c.verdict == Verdict.PASS else ("⚠️" if c.verdict == Verdict.INSUFFICIENT else "❌")
            lines.append(f"  {flag} {c.name}: {c.detail}")
        if self.pnl30_p_value is not None:
            lines.append(
                f"  [stat] pnl30 p={self.pnl30_p_value:.4f}, "
                f"Cohen's d={self.pnl30_effect_size:.3f}"
            )
        return "\n".join(lines)


def _safe_finite(val: Any) -> float | None:
    """有限浮動小数点への安全変換."""
    if val is None:
        return None
    try:
        v = float(val)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def _extract_pnl30_array(records: list[dict[str, Any]]) -> np.ndarray:
    """filled レコードから PnL30 配列を抽出."""
    vals = []
    for r in records:
        if not r.get("filled"):
            continue
        v = _safe_finite(r.get("post_fill_30s_pnl"))
        if v is not None:
            vals.append(v)
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def _compute_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """レコード群から判定用メトリクスを算出.

    side_regime_dashboard._compute_side_metrics と互換性のある出力。
    """
    n_total = len(records)
    filled = [r for r in records if r.get("filled")]
    n_filled = len(filled)
    fill_rate = n_filled / n_total if n_total > 0 else 0.0

    pnl_vals = [_safe_finite(r.get("post_fill_30s_pnl")) for r in filled]
    pnl_clean: list[float] = [v for v in pnl_vals if v is not None]

    if pnl_clean:
        arr = np.array(pnl_clean)
        avg_pnl30 = float(np.mean(arr))
        p10 = float(np.percentile(arr, 10))
        p05 = float(np.percentile(arr, 5))
    else:
        avg_pnl30 = 0.0
        p10 = 0.0
        p05 = 0.0

    # カレンダー日数
    timestamps = []
    for r in filled:
        ts = _safe_finite(r.get("timestamp"))
        if ts is not None:
            timestamps.append(ts)
    if timestamps:
        from datetime import datetime, timezone
        days = set()
        for ts in timestamps:
            try:
                days.add(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d"))
            except (ValueError, OSError):
                continue
        calendar_days = len(days)
    else:
        calendar_days = 0

    return {
        "n_total": n_total,
        "n_filled": n_filled,
        "fill_rate": fill_rate,
        "avg_pnl30_bps": avg_pnl30,
        "downside_p10_bps": p10,
        "downside_p05_bps": p05,
        "calendar_days": calendar_days,
    }


def evaluate_ab_variant(
    variant_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    criteria: ABJudgmentCriteria | None = None,
    *,
    variant_label: str = "variant",
    control_label: str = "control",
) -> ABJudgmentResult:
    """A/B variant を3指標基準で判定.

    Args:
        variant_records: variant 群の全レコード (filled + cancelled).
        control_records: control 群の全レコード.
        criteria: 判定基準 (None=デフォルト).
        variant_label: variant 表示名.
        control_label: control 表示名.

    Returns:
        ABJudgmentResult: 総合判定 + 各指標の詳細.
    """
    if criteria is None:
        criteria = ABJudgmentCriteria()

    vm = _compute_metrics(variant_records)
    cm = _compute_metrics(control_records)

    result = ABJudgmentResult(
        overall=Verdict.PASS,
        variant_label=variant_label,
        control_label=control_label,
        n_variant=int(vm["n_filled"]),
        n_control=int(cm["n_filled"]),
    )

    # --- サンプル充足チェック ---
    if vm["n_filled"] < criteria.min_filled_records:
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(CriterionResult(
            name="sample_size",
            verdict=Verdict.INSUFFICIENT,
            value=float(vm["n_filled"]),
            threshold=float(criteria.min_filled_records),
            detail=f"variant filled={vm['n_filled']} < min={criteria.min_filled_records}",
        ))
        return result

    if vm["calendar_days"] < criteria.min_calendar_days:
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(CriterionResult(
            name="calendar_days",
            verdict=Verdict.INSUFFICIENT,
            value=float(vm["calendar_days"]),
            threshold=float(criteria.min_calendar_days),
            detail=f"variant days={vm['calendar_days']} < min={criteria.min_calendar_days}",
        ))
        return result

    # --- 1. fill_rate 判定 ---
    fr_verdict = Verdict.PASS
    fr_detail_parts: list[str] = []
    vfr = vm["fill_rate"]
    cfr = cm["fill_rate"]

    if vfr < criteria.fill_rate_min:
        fr_verdict = Verdict.FAIL
        fr_detail_parts.append(f"below absolute min {criteria.fill_rate_min:.1%}")

    if cfr > 0 and vfr < cfr * (1 - criteria.fill_rate_degradation_tolerance):
        fr_verdict = Verdict.FAIL
        degradation = (cfr - vfr) / cfr
        fr_detail_parts.append(f"degraded {degradation:.1%} vs control")

    if not fr_detail_parts:
        fr_detail_parts.append("OK")

    result.criteria.append(CriterionResult(
        name="fill_rate",
        verdict=fr_verdict,
        value=vfr,
        threshold=criteria.fill_rate_min,
        detail=f"variant={vfr:.1%} control={cfr:.1%} — {'; '.join(fr_detail_parts)}",
    ))

    # --- 2. avg_pnl30 判定 ---
    pnl_verdict = Verdict.PASS
    pnl_detail_parts: list[str] = []
    vpnl = vm["avg_pnl30_bps"]
    cpnl = cm["avg_pnl30_bps"]

    if vpnl < criteria.avg_pnl30_min_bps:
        pnl_verdict = Verdict.FAIL
        pnl_detail_parts.append(f"below absolute min {criteria.avg_pnl30_min_bps:+.2f}")

    if criteria.avg_pnl30_must_improve and vpnl < cpnl:
        pnl_verdict = Verdict.FAIL
        pnl_detail_parts.append(f"no improvement vs control ({cpnl:+.4f})")

    if not pnl_detail_parts:
        pnl_detail_parts.append("OK")

    result.criteria.append(CriterionResult(
        name="avg_pnl30",
        verdict=pnl_verdict,
        value=vpnl,
        threshold=criteria.avg_pnl30_min_bps,
        detail=f"variant={vpnl:+.4f} control={cpnl:+.4f} bps — {'; '.join(pnl_detail_parts)}",
    ))

    # --- 3. downside_tail (p10) 判定 ---
    ds_verdict = Verdict.PASS
    ds_detail_parts: list[str] = []
    vp10 = vm["downside_p10_bps"]
    cp10 = cm["downside_p10_bps"]

    if vp10 < criteria.downside_p10_min_bps:
        ds_verdict = Verdict.FAIL
        ds_detail_parts.append(f"below absolute min {criteria.downside_p10_min_bps:+.2f}")

    degradation_bps = cp10 - vp10  # 正値 = variant が悪化
    if degradation_bps > criteria.downside_p10_degradation_max_bps:
        ds_verdict = Verdict.FAIL
        ds_detail_parts.append(
            f"degraded {degradation_bps:+.2f}bps vs control "
            f"(max {criteria.downside_p10_degradation_max_bps:+.2f})"
        )

    if not ds_detail_parts:
        ds_detail_parts.append("OK")

    result.criteria.append(CriterionResult(
        name="downside_p10",
        verdict=ds_verdict,
        value=vp10,
        threshold=criteria.downside_p10_min_bps,
        detail=f"variant={vp10:+.4f} control={cp10:+.4f} bps — {'; '.join(ds_detail_parts)}",
    ))

    # --- 統計検定 (ztb.adaptation.ab_test.analyzer 活用) ---
    v_pnl = _extract_pnl30_array(variant_records)
    c_pnl = _extract_pnl30_array(control_records)
    if len(v_pnl) >= 10 and len(c_pnl) >= 10:
        try:
            from ztb.adaptation.ab_test.analyzer import ABTestAnalyzer
            analyzer = ABTestAnalyzer()
            stat_result = analyzer.analyze_parallel(c_pnl, v_pnl)
            result.pnl30_p_value = stat_result.p_value
            result.pnl30_effect_size = stat_result.effect_size
        except Exception as e:
            logger.debug(f"[ab_judgment] Statistical test failed (non-fatal): {e}")

    # --- 総合判定 ---
    verdicts = [c.verdict for c in result.criteria]
    if any(v == Verdict.FAIL for v in verdicts):
        result.overall = Verdict.FAIL
    elif any(v == Verdict.INSUFFICIENT for v in verdicts):
        result.overall = Verdict.INSUFFICIENT
    else:
        result.overall = Verdict.PASS

    return result


# ======================================================================
# P0-C: trending_down sell 実測評価
# ======================================================================


@dataclass
class TrendingEvalCriteria:
    """trending_down sell 継続判定の閾値 (160# P0-C).

    156# D-4 で trending_down sell を開放した施策の有効性を判定。
    YAML `judgment.trending_down_sell` からロード可能。
    """

    # 最低サンプル
    min_filled: int = 10  # filled trending_down sell 最小数
    target_filled: int = 30  # 統計的に十分なサンプル数

    # 判定閾値
    avg_pnl30_min_bps: float = -0.5  # 平均 PnL30 下限 (bps)
    downside_p10_min_bps: float = -5.0  # p10 下限 (bps)

    # 対照群 (trending sell skip = 約定していた場合の期待値)
    # trending sell の期待損失: -0.66 bps (160# §1.3)
    # D-4 opening が正味プラスであれば、期待値改善と判定
    counterfactual_pnl30_bps: float = -0.66

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrendingEvalCriteria:
        """辞書から生成."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrendingEvalResult:
    """trending_down sell 評価結果."""

    verdict: Verdict
    n_filled: int = 0
    n_total: int = 0
    avg_pnl30_bps: float = 0.0
    downside_p10_bps: float | None = None
    downside_p05_bps: float | None = None
    profitable_rate: float = 0.0
    counterfactual_gain_bps: float = 0.0  # 実測 - カウンターファクチュアル
    detail: str = ""
    daily_breakdown: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """人間可読サマリ."""
        flag = "✅" if self.verdict == Verdict.PASS else (
            "⚠️" if self.verdict == Verdict.INSUFFICIENT else "❌"
        )
        lines = [
            f"[Trending Down Sell Eval] {flag} {self.verdict.value.upper()}",
            f"  n_filled={self.n_filled} (total={self.n_total})",
            f"  avg_pnl30={self.avg_pnl30_bps:+.4f} bps",
        ]
        if self.downside_p10_bps is not None:
            lines.append(f"  downside_p10={self.downside_p10_bps:+.4f} bps")
        lines.append(f"  profitable={self.profitable_rate:.1%}")
        lines.append(f"  CF gain={self.counterfactual_gain_bps:+.4f} bps vs skip")
        lines.append(f"  {self.detail}")
        if self.daily_breakdown:
            lines.append("  --- Daily ---")
            for d in self.daily_breakdown:
                avg = d.get("avg_pnl30_bps")
                avg_str = f"{avg:+.4f}" if avg is not None else "N/A"
                lines.append(f"    {d['day']}: n={d['n_filled']}, avg_pnl30={avg_str} bps")
        return "\n".join(lines)


def evaluate_trending_down_sell(
    records: list[dict[str, Any]],
    criteria: TrendingEvalCriteria | None = None,
) -> TrendingEvalResult:
    """trending_down sell の実測評価 (160# P0-C).

    Args:
        records: 全 fill_records (フィルタは内部で実施).
        criteria: 判定基準 (None=デフォルト).

    Returns:
        TrendingEvalResult: 判定 + 日次内訳.
    """
    if criteria is None:
        criteria = TrendingEvalCriteria()

    # trending_down × sell のみ抽出
    td_sell_all = [
        r for r in records
        if r.get("regime") == "trending_down" and r.get("side") == "sell"
    ]
    td_sell_filled = [r for r in td_sell_all if r.get("filled")]

    result = TrendingEvalResult(
        verdict=Verdict.INSUFFICIENT,  # 仮設定 (後で上書き)
        n_total=len(td_sell_all),
        n_filled=len(td_sell_filled),
    )

    # サンプル不足チェック
    if result.n_filled < criteria.min_filled:
        result.verdict = Verdict.INSUFFICIENT
        progress = result.n_filled / criteria.target_filled * 100
        result.detail = (
            f"Insufficient: {result.n_filled}/{criteria.min_filled} min, "
            f"{result.n_filled}/{criteria.target_filled} target ({progress:.0f}%)"
        )
        return result

    # PnL30 配列
    pnl_vals = [_safe_finite(r.get("post_fill_30s_pnl")) for r in td_sell_filled]
    pnl_clean = [v for v in pnl_vals if v is not None]

    if not pnl_clean:
        result.verdict = Verdict.INSUFFICIENT
        result.detail = "No valid PnL30 data"
        return result

    arr = np.array(pnl_clean)
    result.avg_pnl30_bps = float(np.mean(arr))
    result.downside_p10_bps = float(np.percentile(arr, 10))
    result.downside_p05_bps = float(np.percentile(arr, 5))
    result.profitable_rate = float(np.sum(arr > 0) / len(arr))
    result.counterfactual_gain_bps = result.avg_pnl30_bps - criteria.counterfactual_pnl30_bps

    # 日次内訳 (P0-C 固定テンプレート)
    from collections import defaultdict
    from datetime import datetime, timezone
    daily_groups: dict[str, list[float]] = defaultdict(list)
    for r in td_sell_filled:
        ts = _safe_finite(r.get("timestamp"))
        if ts is None:
            continue
        pnl = _safe_finite(r.get("post_fill_30s_pnl"))
        if pnl is None:
            continue
        try:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")
        except (ValueError, OSError):
            continue
        daily_groups[day].append(pnl)

    for day in sorted(daily_groups.keys()):
        vals = daily_groups[day]
        entry: dict[str, Any] = {
            "day": day,
            "n_filled": len(vals),
            "avg_pnl30_bps": round(float(np.mean(vals)), 4),
        }
        if len(vals) >= 3:
            entry["p10_bps"] = round(float(np.percentile(vals, 10)), 4)
        result.daily_breakdown.append(entry)

    # 判定
    fail_reasons: list[str] = []
    if result.avg_pnl30_bps < criteria.avg_pnl30_min_bps:
        fail_reasons.append(
            f"avg_pnl30={result.avg_pnl30_bps:+.4f} < min={criteria.avg_pnl30_min_bps:+.2f}"
        )

    if (
        result.downside_p10_bps is not None
        and result.downside_p10_bps < criteria.downside_p10_min_bps
    ):
        fail_reasons.append(
            f"p10={result.downside_p10_bps:+.4f} < min={criteria.downside_p10_min_bps:+.2f}"
        )

    if fail_reasons:
        result.verdict = Verdict.FAIL
        result.detail = "FAIL: " + "; ".join(fail_reasons)
    else:
        # サンプルが target に達していない場合は慎重に PASS
        if result.n_filled < criteria.target_filled:
            result.verdict = Verdict.PASS
            result.detail = (
                f"PROVISIONAL PASS (n={result.n_filled}/{criteria.target_filled}): "
                f"metrics within thresholds but sample insufficient for full confidence"
            )
        else:
            result.verdict = Verdict.PASS
            result.detail = (
                f"PASS: avg_pnl30={result.avg_pnl30_bps:+.4f} >= "
                f"{criteria.avg_pnl30_min_bps:+.2f}, "
                f"p10={result.downside_p10_bps:+.4f} >= "
                f"{criteria.downside_p10_min_bps:+.2f}"
            )

    return result
