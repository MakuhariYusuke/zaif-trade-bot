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
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import TypedDict

logger = logging.getLogger(__name__)

import numpy as np

from scripts.v460.lib.metrics_utils import MetricsAccumulator, compute_base_metrics
from ztb.io.json_io import JSONObject
from ztb.metrics.fill_quality import format_utc_day
from ztb.utils.dataclass_utils import filter_known_dataclass_fields
from ztb.utils.safety import safe_to_finite

FillRecord = JSONObject


class JudgmentMetrics(TypedDict):
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: float
    downside_p10_bps: float
    downside_p05_bps: float
    calendar_days: int


class DailyBreakdownRow(TypedDict, total=False):
    day: str
    n_filled: int
    avg_pnl30_bps: float
    p10_bps: float


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
    min_filled_records: int = 50  # variant filled レコード最小数 (これ未満は INSUFFICIENT)
    min_control_filled_records: int = 30  # control filled レコード最小数 (10.4: 対称制約)
    min_calendar_days: int = 2  # 最低暦日数

    # --- regime フィルタ ---
    exclude_regimes: list[str] = field(
        default_factory=lambda: ["none"],
    )  # warmup / legacy ノイズ除外 (空=全 regime 含む)

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
    def from_dict(cls, d: JSONObject) -> ABJudgmentCriteria:
        """辞書から生成 (YAML 用)."""
        filtered = filter_known_dataclass_fields(cls, d)
        # exclude_regimes: YAML リスト → そのまま渡す
        if "exclude_regimes" in filtered and filtered["exclude_regimes"] is None:
            filtered["exclude_regimes"] = []
        return cls(**filtered)


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

    # 297# F-4: ノンパラメトリック検定結果
    mann_whitney_p_value: float | None = None
    cliffs_delta_value: float | None = None
    cliffs_delta_interpretation: str = ""
    holm_significant: list[bool] | None = None  # [ttest, mann_whitney]

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
            holm_t = ""
            if self.holm_significant is not None and len(self.holm_significant) >= 1:
                holm_t = " (Holm ✓)" if self.holm_significant[0] else " (Holm ✗)"
            lines.append(
                f"  [stat] Welch t: p={self.pnl30_p_value:.4f}{holm_t}, "
                f"Cohen's d={self.pnl30_effect_size:.3f}"
            )
        if self.mann_whitney_p_value is not None:
            holm_mw = ""
            if self.holm_significant is not None and len(self.holm_significant) >= 2:
                holm_mw = " (Holm ✓)" if self.holm_significant[1] else " (Holm ✗)"
            lines.append(
                f"  [stat] Mann-Whitney: p={self.mann_whitney_p_value:.4f}{holm_mw}, "
                f"Cliff's δ={self.cliffs_delta_value:.3f} "
                f"({self.cliffs_delta_interpretation})"
            )
        return "\n".join(lines)





def _extract_pnl30_array(records: list[FillRecord]) -> np.ndarray:
    """filled レコードから PnL30 配列を抽出."""
    vals = []
    for r in records:
        if not r.get("filled"):
            continue
        v = safe_to_finite(r.get("post_fill_30s_pnl"))
        if v is not None:
            vals.append(v)
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def _compute_metrics(records: list[FillRecord]) -> JudgmentMetrics:
    """レコード群から判定用メトリクスを算出.

    161# DRY: compute_base_metrics に委譲。
    """
    base = compute_base_metrics(records)
    return {
        "n_total": base["n_total"],
        "n_filled": base["n_filled"],
        "fill_rate": base["fill_rate"],
        "avg_pnl30_bps": base["avg_pnl30_bps"],
        "downside_p10_bps": base["downside_p10_bps"],
        "downside_p05_bps": base["downside_p05_bps"],
        "calendar_days": base["calendar_days"],
    }


def _compute_metrics_with_pnl(
    records: list[FillRecord],
) -> tuple[JudgmentMetrics, np.ndarray]:
    """判定用メトリクスと PnL30 配列を同時計算（再走査回避）."""
    base = compute_base_metrics(records)
    metrics: JudgmentMetrics = {
        "n_total": base["n_total"],
        "n_filled": base["n_filled"],
        "fill_rate": base["fill_rate"],
        "avg_pnl30_bps": base["avg_pnl30_bps"],
        "downside_p10_bps": base["downside_p10_bps"],
        "downside_p05_bps": base["downside_p05_bps"],
        "calendar_days": base["calendar_days"],
    }
    return metrics, base["pnl30_array"]


@lru_cache(maxsize=1)
def _resolve_ab_test_analyzer_class() -> object | None:
    """ABTestAnalyzer クラスを遅延解決してキャッシュ."""
    try:
        from ztb.adaptation.ab_test.analyzer import ABTestAnalyzer
        return ABTestAnalyzer
    except Exception as e:
        logger.debug("ABTestAnalyzer import failed: %s", e)
        return None


def _cohen_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Cohen's d (sample_b - sample_a)."""
    mean_a = float(np.mean(sample_a))
    mean_b = float(np.mean(sample_b))
    std_a = float(np.std(sample_a, ddof=1))
    std_b = float(np.std(sample_b, ddof=1))
    pooled = math.sqrt((std_a * std_a + std_b * std_b) / 2.0)
    if pooled <= 0.0 or not math.isfinite(pooled):
        return 0.0
    return (mean_b - mean_a) / pooled


# --- 297# F-4: ノンパラメトリック検定 + 多重比較補正 ---
# gate_c3_comparison.py から cherry-pick (pure Python, scipy 不要)


def _norm_cdf(z: float) -> float:
    """標準正規分布の CDF (Abramowitz & Stegun 7.1.26 erfc 近似)."""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5 = -1.453152027, 1.061405429
    p = 0.3275911
    x = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    erfc_val = (
        ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1
    ) * t * math.exp(-x * x)
    if z >= 0:
        return 1.0 - 0.5 * erfc_val
    return 0.5 * erfc_val


def _mann_whitney_u(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float]:
    """Mann-Whitney U 検定 (正規近似, O(n*m)).

    Returns:
        (U 統計量, 近似 p 値)
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, 1.0

    # 全ペア比較
    u = 0.0
    for xi in x:
        for yi in y:
            if xi > yi:
                u += 1
            elif xi == yi:
                u += 0.5

    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    if sigma <= 0.0:
        return u, 1.0

    z = (u - mu) / sigma
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return u, p_value


def _cliffs_delta(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, str]:
    """Cliff's Delta (ノンパラメトリック効果量).

    Returns:
        (delta, 解釈) — 解釈は negligible / small / medium / large
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, "negligible"

    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1

    delta = (more - less) / (nx * ny)
    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"
    return delta, interp


def _holm_bonferroni(
    p_values: list[float], alpha: float = 0.05,
) -> list[bool]:
    """Holm-Bonferroni 多重比較補正.

    Returns:
        各検定が alpha 水準で有意かどうかのリスト
    """
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = sorted(range(n), key=lambda i: p_values[i])
    significant = [False] * n
    for rank, idx in enumerate(sorted_idx):
        adjusted_alpha = alpha / (n - rank)
        if p_values[idx] <= adjusted_alpha:
            significant[idx] = True
        else:
            break
    return significant


def _compute_statistical_comparison(
    control_pnl: np.ndarray,
    variant_pnl: np.ndarray,
) -> tuple[float | None, float | None]:
    """P値と効果量を計算（軽量経路優先、必要時のみ互換fallback）。"""
    try:
        from scipy import stats

        _, p_value = stats.ttest_ind(control_pnl, variant_pnl, equal_var=False)
        effect_size = _cohen_d(control_pnl, variant_pnl)
        p = float(p_value)
        eff = float(effect_size)
        return (
            p if math.isfinite(p) else None,
            eff if math.isfinite(eff) else None,
        )
    except Exception as e:
        logger.debug("scipy ttest_ind failed, trying ABTestAnalyzer: %s", e)
        analyzer_cls = _resolve_ab_test_analyzer_class()
        if analyzer_cls is None:
            return None, None
        try:
            analyzer = analyzer_cls()
            stat_result = analyzer.analyze_parallel(control_pnl, variant_pnl)
            p = float(stat_result.p_value)
            eff = float(stat_result.effect_size)
            return (
                p if math.isfinite(p) else None,
                eff if math.isfinite(eff) else None,
            )
        except Exception as e:
            logger.debug("ABTestAnalyzer.analyze_parallel failed: %s", e)
            return None, None


def evaluate_ab_variant(
    variant_records: list[FillRecord],
    control_records: list[FillRecord],
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

    # --- regime フィルタ (warmup / legacy ノイズ除外) ---
    if criteria.exclude_regimes:
        excl = set(criteria.exclude_regimes)
        variant_records = [
            r for r in variant_records
            if str(r.get("regime") or "none") not in excl
        ]
        control_records = [
            r for r in control_records
            if str(r.get("regime") or "none") not in excl
        ]

    vm, v_pnl = _compute_metrics_with_pnl(variant_records)
    cm, c_pnl = _compute_metrics_with_pnl(control_records)

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

    if cm["n_filled"] < criteria.min_control_filled_records:
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(CriterionResult(
            name="control_sample_size",
            verdict=Verdict.INSUFFICIENT,
            value=float(cm["n_filled"]),
            threshold=float(criteria.min_control_filled_records),
            detail=f"control filled={cm['n_filled']} < min={criteria.min_control_filled_records}",
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

    # --- 0. PnL データ有効性チェック (160# bugfix) ---
    if math.isnan(vm["avg_pnl30_bps"]):
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(CriterionResult(
            name="pnl_data",
            verdict=Verdict.INSUFFICIENT,
            value=float(vm["n_filled"]),
            threshold=1.0,
            detail=f"variant filled={vm['n_filled']} but no valid PnL data",
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
    if len(v_pnl) >= 10 and len(c_pnl) >= 10:
        p_value, effect_size = _compute_statistical_comparison(c_pnl, v_pnl)
        result.pnl30_p_value = p_value
        result.pnl30_effect_size = effect_size

        # 297# F-4: ノンパラメトリック検定 + Holm-Bonferroni 補正
        try:
            _, mw_p = _mann_whitney_u(c_pnl, v_pnl)
            cd_val, cd_interp = _cliffs_delta(c_pnl, v_pnl)
            if math.isfinite(mw_p):
                result.mann_whitney_p_value = mw_p
            if math.isfinite(cd_val):
                result.cliffs_delta_value = cd_val
                result.cliffs_delta_interpretation = cd_interp

            # Holm-Bonferroni: 2 検定 (Welch t + Mann-Whitney) を補正
            collected_p: list[float] = []
            if p_value is not None and math.isfinite(p_value):
                collected_p.append(p_value)
            if result.mann_whitney_p_value is not None:
                collected_p.append(result.mann_whitney_p_value)
            if len(collected_p) >= 2:
                result.holm_significant = _holm_bonferroni(collected_p)
        except Exception as e:
            logger.debug("Nonparametric test failed: %s", e)

    # --- 総合判定 ---
    verdicts = [c.verdict for c in result.criteria]
    if any(v == Verdict.FAIL for v in verdicts):
        result.overall = Verdict.FAIL
    elif any(v == Verdict.INSUFFICIENT for v in verdicts):
        result.overall = Verdict.INSUFFICIENT
    else:
        result.overall = Verdict.PASS

    return result


def _filter_by_regime(
    records: list[FillRecord], regime: str,
) -> list[FillRecord]:
    """指定 regime のレコードのみ抽出."""
    return [r for r in records if str(r.get("regime") or "none") == regime]


@dataclass
class PerRegimeResult:
    """Regime 別 A/B 判定結果."""

    regime: str
    result: ABJudgmentResult


def evaluate_per_regime(
    variant_records: list[FillRecord],
    control_records: list[FillRecord],
    criteria: ABJudgmentCriteria | None = None,
    *,
    variant_label: str = "variant",
    control_label: str = "control",
    target_regimes: list[str] | None = None,
) -> list[PerRegimeResult]:
    """Regime 別に A/B 判定を実行.

    集約値ではデータ汚染が見えない問題を解決するため、
    regime 単位で3指標判定を行う。

    Args:
        variant_records: variant 群の全レコード.
        control_records: control 群の全レコード.
        criteria: 判定基準 (None=デフォルト). exclude_regimes は無視される。
        variant_label: variant 表示名.
        control_label: control 表示名.
        target_regimes: 対象 regime リスト (None=全 regime).

    Returns:
        list[PerRegimeResult]: regime ごとの判定結果.
    """
    if criteria is None:
        criteria = ABJudgmentCriteria()

    # regime 別に分類
    all_regimes: set[str] = set()
    for r in variant_records:
        all_regimes.add(str(r.get("regime") or "none"))
    for r in control_records:
        all_regimes.add(str(r.get("regime") or "none"))

    if target_regimes is not None:
        all_regimes = all_regimes & set(target_regimes)

    # per-regime 判定: exclude_regimes を無効化して個別 regime を評価
    per_regime_criteria = ABJudgmentCriteria(
        min_filled_records=criteria.min_filled_records,
        min_control_filled_records=criteria.min_control_filled_records,
        min_calendar_days=criteria.min_calendar_days,
        fill_rate_min=criteria.fill_rate_min,
        fill_rate_degradation_tolerance=criteria.fill_rate_degradation_tolerance,
        avg_pnl30_min_bps=criteria.avg_pnl30_min_bps,
        avg_pnl30_must_improve=criteria.avg_pnl30_must_improve,
        downside_p10_min_bps=criteria.downside_p10_min_bps,
        downside_p10_degradation_max_bps=criteria.downside_p10_degradation_max_bps,
        exclude_regimes=[],  # regime filtering は外側で行う
    )

    results: list[PerRegimeResult] = []
    for regime in sorted(all_regimes):
        v_filtered = _filter_by_regime(variant_records, regime)
        c_filtered = _filter_by_regime(control_records, regime)

        # どちらかが空なら INSUFFICIENT
        if not v_filtered and not c_filtered:
            continue

        ab_result = evaluate_ab_variant(
            variant_records=v_filtered,
            control_records=c_filtered,
            criteria=per_regime_criteria,
            variant_label=f"{variant_label}[{regime}]",
            control_label=f"{control_label}[{regime}]",
        )
        results.append(PerRegimeResult(regime=regime, result=ab_result))

    return results


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
    def from_dict(cls, d: JSONObject) -> TrendingEvalCriteria:
        """辞書から生成."""
        return cls(**filter_known_dataclass_fields(cls, d))


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
    daily_breakdown: list[DailyBreakdownRow] = field(default_factory=list)

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
    records: list[FillRecord],
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

    metrics = MetricsAccumulator()
    daily_groups: dict[str, MetricsAccumulator] = defaultdict(MetricsAccumulator)

    for record in records:
        if record.get("regime") != "trending_down" or record.get("side") != "sell":
            continue
        metrics.add(record)
        if not record.get("filled"):
            continue

        day = format_utc_day(safe_to_finite(record.get("timestamp")))
        if day is None:
            continue
        daily_groups[day].add(record)

    result = TrendingEvalResult(
        verdict=Verdict.INSUFFICIENT,  # 仮設定 (後で上書き)
        n_total=metrics.n_total,
        n_filled=metrics.n_filled,
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

    base_metrics = metrics.to_base_metrics()
    pnl_arr = base_metrics["pnl30_array"]
    if pnl_arr.size == 0:
        result.verdict = Verdict.INSUFFICIENT
        result.detail = "No valid PnL30 data"
        return result

    result.avg_pnl30_bps = base_metrics["avg_pnl30_bps"]
    result.downside_p10_bps = float(np.percentile(pnl_arr, 10))
    result.downside_p05_bps = float(np.percentile(pnl_arr, 5))
    result.profitable_rate = base_metrics["profitable_rate"]
    result.counterfactual_gain_bps = result.avg_pnl30_bps - criteria.counterfactual_pnl30_bps

    # 日次内訳 (P0-C 固定テンプレート)
    for day in sorted(daily_groups.keys()):
        daily = daily_groups[day].to_base_metrics()
        entry: DailyBreakdownRow = {
            "day": day,
            "n_filled": daily["n_filled"],
            "avg_pnl30_bps": round(daily["avg_pnl30_bps"], 4),
        }
        if daily["n_filled"] >= 3 and daily["downside_p10_bps"] == daily["downside_p10_bps"]:
            entry["p10_bps"] = round(daily["downside_p10_bps"], 4)
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
