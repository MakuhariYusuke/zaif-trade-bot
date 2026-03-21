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

from ztb.adaptation.ab_test.judgment_rules import (
    CriterionAssessment as _CriterionAssessment,
    assess_avg_pnl30 as _assess_avg_pnl30,
    assess_downside_p10 as _assess_downside_p10,
    assess_fill_rate as _assess_fill_rate,
    build_insufficient_assessment as _build_insufficient_assessment,
    combine_assessment_verdicts as _combine_assessment_verdicts,
)
from ztb.io.json_io import JSONObject
from ztb.metrics.fill_quality import format_utc_day
from ztb.metrics.record_metrics import MetricsAccumulator, compute_base_metrics
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


def _to_criterion_result(assessment: _CriterionAssessment) -> CriterionResult:
    return CriterionResult(
        name=assessment.name,
        verdict=Verdict(assessment.verdict),
        value=assessment.value,
        threshold=assessment.threshold,
        detail=assessment.detail,
    )


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

    # 306# 301#-F2 改善: block bootstrap + matched comparison
    bootstrap_mean_diff: float | None = None  # buy-sell PnL30 差の bootstrap 推定値
    bootstrap_ci_lower: float | None = None  # 95% CI 下限
    bootstrap_ci_upper: float | None = None  # 95% CI 上限
    bootstrap_p_value: float | None = None  # bootstrap permutation p 値
    matched_n_pairs: int = 0  # 時間近接 matched pair 数
    matched_mean_diff: float | None = None  # matched pair PnL30 差の平均
    matched_ci_lower: float | None = None  # matched 95% CI 下限
    matched_ci_upper: float | None = None  # matched 95% CI 上限
    matched_p_value: float | None = None  # matched Wilcoxon signed-rank p 値

    def summary(self) -> str:
        """人間可読サマリ."""
        lines = [
            f"[Side Comparison] {self.overall.value.upper()}",
            f"  variant={self.variant_label} (n={self.n_variant})"
            f" vs control={self.control_label} (n={self.n_control})",
            "  ※ 観察比較であり、ランダム割当の A/B テストではない (301# F2)",
        ]
        for c in self.criteria:
            flag = "✅" if c.verdict == Verdict.PASS else ("⚠️" if c.verdict == Verdict.INSUFFICIENT else "❌")
            lines.append(f"  {flag} {c.name}: {c.detail}")
        if self.pnl30_p_value is not None:
            holm_t = ""
            if self.holm_significant is not None and len(self.holm_significant) >= 1:
                holm_t = " (Holm ✓)" if self.holm_significant[0] else " (Holm ✗)"
            eff_str = f"{self.pnl30_effect_size:.3f}" if self.pnl30_effect_size is not None else "N/A"
            lines.append(
                f"  [stat] Welch t: p={self.pnl30_p_value:.4f}{holm_t}, "
                f"Cohen's d={eff_str}"
            )
        if self.mann_whitney_p_value is not None:
            holm_mw = ""
            if self.holm_significant is not None and len(self.holm_significant) >= 2:
                holm_mw = " (Holm ✓)" if self.holm_significant[1] else " (Holm ✗)"
            cd_str = f"{self.cliffs_delta_value:.3f}" if self.cliffs_delta_value is not None else "N/A"
            lines.append(
                f"  [stat] Mann-Whitney: p={self.mann_whitney_p_value:.4f}{holm_mw}, "
                f"Cliff's δ={cd_str} "
                f"({self.cliffs_delta_interpretation})"
            )
        # 306# block bootstrap CI
        if self.bootstrap_mean_diff is not None:
            ci_lo = f"{self.bootstrap_ci_lower:+.4f}" if self.bootstrap_ci_lower is not None else "N/A"
            ci_hi = f"{self.bootstrap_ci_upper:+.4f}" if self.bootstrap_ci_upper is not None else "N/A"
            bp = f"p={self.bootstrap_p_value:.4f}" if self.bootstrap_p_value is not None else "p=N/A"
            lines.append(
                f"  [stat] Block Bootstrap: diff={self.bootstrap_mean_diff:+.4f} bps, "
                f"95%CI=[{ci_lo}, {ci_hi}], {bp}"
            )
        # 306# matched temporal comparison
        if self.matched_n_pairs > 0:
            md = f"{self.matched_mean_diff:+.4f}" if self.matched_mean_diff is not None else "N/A"
            mci_lo = f"{self.matched_ci_lower:+.4f}" if self.matched_ci_lower is not None else "N/A"
            mci_hi = f"{self.matched_ci_upper:+.4f}" if self.matched_ci_upper is not None else "N/A"
            mp = f"p={self.matched_p_value:.4f}" if self.matched_p_value is not None else "p=N/A"
            lines.append(
                f"  [stat] Matched Pairs (n={self.matched_n_pairs}): "
                f"diff={md} bps, 95%CI=[{mci_lo}, {mci_hi}], {mp}"
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
    """標準正規分布の CDF (Abramowitz & Stegun 7.1.26 erfc 近似).

    Reference: Abramowitz, M. & Stegun, I. A. (1964),
    *Handbook of Mathematical Functions*, eq. 7.1.26.
    最大絶対誤差 ≤ 1.5×10⁻⁷.
    """
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


def _pairwise_counts(
    x: np.ndarray, y: np.ndarray,
) -> tuple[int, int, int]:
    """全ペア比較で (more, less, equal) カウントを返す (numpy ベクトル化).

    x[i] > y[j] → more, x[i] < y[j] → less, x[i] == y[j] → equal.
    O(n*m) メモリだが Python ループ比 ~100x 高速。
    """
    # diff[i,j] = x[i] - y[j]
    diff = x[:, None] - y[None, :]
    more = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    equal = int(np.sum(diff == 0))
    return more, less, equal


def _mann_whitney_u(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float]:
    """Mann-Whitney U 検定 (正規近似, numpy ベクトル化).

    Returns:
        (U 統計量, 近似 p 値)
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, 1.0

    more, _less, equal = _pairwise_counts(x, y)
    u = float(more) + 0.5 * float(equal)

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
    """Cliff's Delta (ノンパラメトリック効果量, numpy ベクトル化).

    Returns:
        (delta, 解釈) — 解釈は negligible / small / medium / large
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, "negligible"

    more, less, _equal = _pairwise_counts(x, y)
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


# --- 306# 301#-F2: Block Bootstrap + Matched Temporal Comparison ---


def _benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05,
) -> list[bool]:
    """Benjamini-Hochberg FDR 多重比較補正 (301# F5).

    Holm-Bonferroni より検出力が高い。regime 横断テスト向け。

    Returns:
        各検定が FDR-adjusted alpha 水準で有意かどうかのリスト
    """
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = sorted(range(n), key=lambda i: p_values[i])
    significant = [False] * n
    # 最大 k s.t. p_(k) <= k/n * alpha を見つける
    max_significant_rank = -1
    for rank, idx in enumerate(sorted_idx):
        bh_threshold = (rank + 1) / n * alpha
        if p_values[idx] <= bh_threshold:
            max_significant_rank = rank
    # max_significant_rank 以下の全検定を有意とする
    if max_significant_rank >= 0:
        for rank in range(max_significant_rank + 1):
            significant[sorted_idx[rank]] = True
    return significant


def _bootstrap_sample_means(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    sample_size: int,
) -> np.ndarray:
    """IID bootstrap の標本平均をベクトル化計算する."""
    idx = rng.integers(0, len(arr), size=(n_bootstrap, sample_size))
    return np.mean(arr[idx], axis=1, dtype=float)


def _block_bootstrap_sample_means(
    arr: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    block_size: int,
) -> np.ndarray:
    """移動ブロックブートストラップの標本平均をベクトル化計算する."""
    n = len(arr)
    if n <= block_size:
        return _bootstrap_sample_means(
            arr,
            rng=rng,
            n_bootstrap=n_bootstrap,
            sample_size=n,
        )

    n_blocks = max(1, int(math.ceil(n / block_size)))
    starts = rng.integers(0, n - block_size + 1, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size, dtype=int)
    indices = (starts[..., None] + offsets).reshape(n_bootstrap, -1)[:, :n]
    return np.mean(arr[indices], axis=1, dtype=float)


def _block_bootstrap_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    block_size: int = 10,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Block Bootstrap for mean difference (306# 301#-F2).

    時系列自己相関を尊重するため、連続ブロック単位でリサンプリングする。
    Künsch (1989) の移動ブロックブートストラップ (MBB) を実装。

    Args:
        x: variant (sell) PnL30 配列
        y: control (buy) PnL30 配列
        n_bootstrap: ブートストラップ反復数
        block_size: ブロックサイズ (自己相関長の目安)
        seed: 乱数シード

    Returns:
        (mean_diff, ci_lower, ci_upper, p_value)
        mean_diff = mean(x) - mean(y)
        p_value = bootstrap permutation p値
    """
    rng = np.random.default_rng(seed)
    observed_diff = float(np.mean(x)) - float(np.mean(y))
    x_boot_means = _block_bootstrap_sample_means(
        x,
        rng=rng,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )
    y_boot_means = _block_bootstrap_sample_means(
        y,
        rng=rng,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )
    boot_diffs = x_boot_means - y_boot_means

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Bootstrap permutation p-value: centeringして帰無仮説検定
    centered = boot_diffs - float(np.mean(boot_diffs))
    p_value = float(np.mean(np.abs(centered) >= abs(observed_diff)))

    return observed_diff, ci_lower, ci_upper, p_value


def _matched_temporal_comparison(
    variant_records: list[FillRecord],
    control_records: list[FillRecord],
    *,
    max_gap_sec: float = 600.0,
) -> tuple[int, float | None, float | None, float | None, float | None]:
    """時間近接 Matched Pair 比較 (306# 301#-F2).

    同じ regime 内で時間的に近い variant/control fill をペアリングし、
    市場条件をマッチさせた上でペア差を検定する。

    Args:
        variant_records: variant (sell) の filled レコード
        control_records: control (buy) の filled レコード
        max_gap_sec: 最大時間差 (秒)

    Returns:
        (n_pairs, mean_diff, ci_lower, ci_upper, p_value)
    """
    # filled only + timestamp/pnl30 抽出
    def _extract_ts_pnl(records: list[FillRecord]) -> list[tuple[float, float]]:
        result = []
        for r in records:
            if not r.get("filled"):
                continue
            ts = safe_to_finite(r.get("timestamp"))
            pnl = safe_to_finite(r.get("post_fill_30s_pnl"))
            if ts is None or pnl is None:
                continue
            result.append((float(ts), float(pnl)))
        return sorted(result, key=lambda x: x[0])

    v_data = _extract_ts_pnl(variant_records)
    c_data = _extract_ts_pnl(control_records)

    if not v_data or not c_data:
        return 0, None, None, None, None

    # Greedy nearest-neighbor matching (regime 内時間近接)
    used_c = set[int]()
    pairs: list[tuple[float, float]] = []  # (v_pnl, c_pnl)

    c_idx = 0
    for v_ts, v_pnl in v_data:
        best_ci = -1
        best_gap = max_gap_sec + 1.0
        # c_data 内で v_ts に最も近い未使用ペアを探す
        for ci in range(max(0, c_idx - 5), len(c_data)):
            if ci in used_c:
                continue
            gap = abs(c_data[ci][0] - v_ts)
            if gap < best_gap:
                best_gap = gap
                best_ci = ci
            # c_data はソート済みなので、gap が増加し始めたら打ち切り
            if c_data[ci][0] > v_ts + max_gap_sec:
                break
        if best_ci >= 0 and best_gap <= max_gap_sec:
            pairs.append((v_pnl, c_data[best_ci][1]))
            used_c.add(best_ci)
            # c_idx を進めて探索窓を制限
            while c_idx < len(c_data) and c_idx in used_c:
                c_idx += 1

    n_pairs = len(pairs)
    if n_pairs < 10:
        return n_pairs, None, None, None, None

    diffs = np.array([v - c for v, c in pairs], dtype=float)
    mean_diff = float(np.mean(diffs))

    # Bootstrap CI for paired differences
    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_means = _bootstrap_sample_means(
        diffs,
        rng=rng,
        n_bootstrap=n_boot,
        sample_size=n_pairs,
    )
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Wilcoxon signed-rank test (ノンパラメトリック, 正規近似)
    p_value = _wilcoxon_signed_rank(diffs)

    return n_pairs, mean_diff, ci_lower, ci_upper, p_value


def _wilcoxon_signed_rank(diffs: np.ndarray) -> float:
    """Wilcoxon signed-rank test (正規近似, pure Python/numpy).

    ゼロ差は除外し、残りに対して正規近似 z-test を行う。

    Returns:
        両側 p 値
    """
    # ゼロ差を除外
    nonzero = diffs[diffs != 0.0]
    n = len(nonzero)
    if n < 5:
        return 1.0

    abs_vals = np.abs(nonzero)
    ranks = np.empty(n, dtype=float)
    sorted_idx = np.argsort(abs_vals)
    # 平均順位 (tie 処理)
    i = 0
    while i < n:
        j = i
        while j < n and abs_vals[sorted_idx[j]] == abs_vals[sorted_idx[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j

    # 正の差の順位和
    w_plus = float(np.sum(ranks[nonzero > 0]))
    # 正規近似
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma <= 0.0:
        return 1.0
    z = (w_plus - mu) / sigma
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return p_value


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
        result.criteria.append(
            _to_criterion_result(
                _build_insufficient_assessment(
                    name="sample_size",
                    value=float(vm["n_filled"]),
                    threshold=float(criteria.min_filled_records),
                    detail=(
                        f"variant filled={vm['n_filled']} < min={criteria.min_filled_records}"
                    ),
                )
            )
        )
        return result

    if cm["n_filled"] < criteria.min_control_filled_records:
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(
            _to_criterion_result(
                _build_insufficient_assessment(
                    name="control_sample_size",
                    value=float(cm["n_filled"]),
                    threshold=float(criteria.min_control_filled_records),
                    detail=(
                        f"control filled={cm['n_filled']} < min={criteria.min_control_filled_records}"
                    ),
                )
            )
        )
        return result

    if vm["calendar_days"] < criteria.min_calendar_days:
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(
            _to_criterion_result(
                _build_insufficient_assessment(
                    name="calendar_days",
                    value=float(vm["calendar_days"]),
                    threshold=float(criteria.min_calendar_days),
                    detail=(
                        f"variant days={vm['calendar_days']} < min={criteria.min_calendar_days}"
                    ),
                )
            )
        )
        return result

    # --- 0. PnL データ有効性チェック (160# bugfix) ---
    if math.isnan(vm["avg_pnl30_bps"]):
        result.overall = Verdict.INSUFFICIENT
        result.criteria.append(
            _to_criterion_result(
                _build_insufficient_assessment(
                    name="pnl_data",
                    value=float(vm["n_filled"]),
                    threshold=1.0,
                    detail=f"variant filled={vm['n_filled']} but no valid PnL data",
                )
            )
        )
        return result

    fill_rate_assessment = _assess_fill_rate(
        variant_fill_rate=vm["fill_rate"],
        control_fill_rate=cm["fill_rate"],
        fill_rate_min=criteria.fill_rate_min,
        fill_rate_degradation_tolerance=criteria.fill_rate_degradation_tolerance,
    )
    avg_pnl30_assessment = _assess_avg_pnl30(
        variant_avg_pnl30_bps=vm["avg_pnl30_bps"],
        control_avg_pnl30_bps=cm["avg_pnl30_bps"],
        avg_pnl30_min_bps=criteria.avg_pnl30_min_bps,
        avg_pnl30_must_improve=criteria.avg_pnl30_must_improve,
    )
    downside_assessment = _assess_downside_p10(
        variant_downside_p10_bps=vm["downside_p10_bps"],
        control_downside_p10_bps=cm["downside_p10_bps"],
        downside_p10_min_bps=criteria.downside_p10_min_bps,
        downside_p10_degradation_max_bps=criteria.downside_p10_degradation_max_bps,
    )
    result.criteria.extend(
        [
            _to_criterion_result(fill_rate_assessment),
            _to_criterion_result(avg_pnl30_assessment),
            _to_criterion_result(downside_assessment),
        ]
    )

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

        # 306# 301#-F2: Block Bootstrap for mean difference CI
        try:
            diff, ci_lo, ci_hi, bp = _block_bootstrap_mean_diff(v_pnl, c_pnl)
            if math.isfinite(diff):
                result.bootstrap_mean_diff = diff
            if math.isfinite(ci_lo):
                result.bootstrap_ci_lower = ci_lo
            if math.isfinite(ci_hi):
                result.bootstrap_ci_upper = ci_hi
            if math.isfinite(bp):
                result.bootstrap_p_value = bp
        except Exception as e:
            logger.debug("Block bootstrap failed: %s", e)

        # 306# 301#-F2: Matched temporal comparison
        try:
            mp_n, mp_diff, mp_ci_lo, mp_ci_hi, mp_p = _matched_temporal_comparison(
                variant_records, control_records,
            )
            result.matched_n_pairs = mp_n
            if mp_diff is not None and math.isfinite(mp_diff):
                result.matched_mean_diff = mp_diff
            if mp_ci_lo is not None and math.isfinite(mp_ci_lo):
                result.matched_ci_lower = mp_ci_lo
            if mp_ci_hi is not None and math.isfinite(mp_ci_hi):
                result.matched_ci_upper = mp_ci_hi
            if mp_p is not None and math.isfinite(mp_p):
                result.matched_p_value = mp_p
        except Exception as e:
            logger.debug("Matched temporal comparison failed: %s", e)

    # --- 総合判定 ---
    result.overall = Verdict(
        _combine_assessment_verdicts(
            [
                fill_rate_assessment,
                avg_pnl30_assessment,
                downside_assessment,
            ]
        )
    )

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
    # trending sell の期待損失: -0.66 bps (160# §1.3, 2026-02 データ)
    # D-4 opening が正味プラスであれば、期待値改善と判定
    # NOTE: 市場条件変動で陳腐化リスクあり — 動的推定は将来課題 (P3)
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
