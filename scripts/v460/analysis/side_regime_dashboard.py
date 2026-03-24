"""159# P0-B/C: side × regime 3指標ダッシュボード + trending 日次テンプレート.

sell offset A/B 評価で fill_rate 単独最適化は危険 (159# §3.1)。
最低限 fill_rate / avg_pnl30 / downside_tail (p10) の3指標同時管理が必要。

160# P0-B/C:
- A/B 判定基準の固定 (3指標 PASS/FAIL) → ab_judgment 統合
- trending_down sell 固定テンプレート日次評価 → TrendingEvalResult 統合

Usage:
    .venv\\Scripts\\python.exe scripts/v460/analysis/side_regime_dashboard.py
    .venv\\Scripts\\python.exe scripts/v460/analysis/side_regime_dashboard.py --results-dir results/v460/fill_test
    .venv\\Scripts\\python.exe scripts/v460/analysis/side_regime_dashboard.py --with-judgment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, cast

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import format_utc_day, iter_fill_record_objects_glob
from ztb.metrics.record_metrics import (
    ExtendedMetrics,
    MetricRecord,
    MetricsAccumulator,
    compute_extended_metrics,
)

# 160# P0-B/C: judgment 統合
from ztb.io.yaml_io import read_yaml
from ztb.utils.safety import safe_to_finite
from scripts.v460.lib.ab_judgment import (
    ABJudgmentCriteria,
    ABJudgmentResult,
    PerRegimeResult,
    TrendingEvalCriteria,
    TrendingEvalResult,
    Verdict,
    evaluate_ab_variant,
    evaluate_per_regime,
    evaluate_trending_down_sell,
)


class SideMetrics(TypedDict, total=False):
    """Side 別メトリクス (159# §3.1 準拠)."""

    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: float
    std_pnl30_bps: float
    downside_p10_bps: float  # p10 = worst decile
    downside_p05_bps: float  # p5
    profitable_rate: float
    as_rate: float
    avg_as_loss_bps: float
    # 159# P1-A/P1-C: 新フィールド集計
    reprice_rate: float  # reprice 発生率 (filled 中)
    avg_reprice_drift_bps: float  # reprice drift 平均
    vg_trigger_rate: float  # VG trigger 率 (filled 中)


class RegimeSideMetrics(TypedDict, total=False):
    """Regime × Side メトリクス."""

    regime: str
    side: str
    metrics: SideMetrics


class DashboardResult(TypedDict, total=False):
    """ダッシュボード結果."""

    timestamp: str
    results_dir: str
    total_records: int
    total_filled: int
    overall_fill_rate: float
    side_summary: dict[str, SideMetrics]
    regime_side_detail: list[RegimeSideMetrics]
    trending_daily: list[dict[str, object]]
    # 160# P0-B/C: judgment 統合
    ab_judgment: dict[str, object] | None
    ab_judgment_incl_none: dict[str, object] | None  # 303# A: none 含有版
    trending_eval: dict[str, object] | None
    per_regime_judgment: list[dict[str, object]] | None


# 161# DRY: _to_finite -> ztb.utils.safety.safe_to_finite に統合


def _ab_result_to_dict(ab_result: ABJudgmentResult) -> dict[str, object]:
    """ABJudgmentResult を dict に変換 (DRY: 303# A)."""
    return {
        "overall": ab_result.overall.value,
        "variant_label": ab_result.variant_label,
        "control_label": ab_result.control_label,
        "n_variant": ab_result.n_variant,
        "n_control": ab_result.n_control,
        "criteria": [
            {
                "name": c.name,
                "verdict": c.verdict.value,
                "value": c.value,
                "threshold": c.threshold,
                "detail": c.detail,
            }
            for c in ab_result.criteria
        ],
        "pnl30_p_value": ab_result.pnl30_p_value,
        "pnl30_effect_size": ab_result.pnl30_effect_size,
        # 298# F-4: ノンパラメトリック検定結果
        "mann_whitney_p_value": ab_result.mann_whitney_p_value,
        "cliffs_delta_value": ab_result.cliffs_delta_value,
        "cliffs_delta_interpretation": ab_result.cliffs_delta_interpretation,
        "holm_significant": ab_result.holm_significant,
        # 306# 301#-F2: Block Bootstrap + Matched Temporal Comparison
        "bootstrap_mean_diff": ab_result.bootstrap_mean_diff,
        "bootstrap_ci_lower": ab_result.bootstrap_ci_lower,
        "bootstrap_ci_upper": ab_result.bootstrap_ci_upper,
        "bootstrap_p_value": ab_result.bootstrap_p_value,
        "matched_n_pairs": ab_result.matched_n_pairs,
        "matched_mean_diff": ab_result.matched_mean_diff,
        "matched_ci_lower": ab_result.matched_ci_lower,
        "matched_ci_upper": ab_result.matched_ci_upper,
        "matched_p_value": ab_result.matched_p_value,
        "summary": ab_result.summary(),
    }


def _compute_side_metrics(records: list[MetricRecord]) -> SideMetrics:
    """レコード群から SideMetrics を計算.

    161# DRY: compute_extended_metrics に委譲。
    """
    ext = compute_extended_metrics(records)
    return _side_metrics_from_extended(ext)


def _side_metrics_from_extended(ext: ExtendedMetrics) -> SideMetrics:
    """拡張メトリクスを丸め込み、SideMetrics へ変換する."""
    return {
        "n_total": ext["n_total"],
        "n_filled": ext["n_filled"],
        "fill_rate": round(ext["fill_rate"], 4),
        "avg_pnl30_bps": round(ext["avg_pnl30_bps"], 4) if ext["avg_pnl30_bps"] == ext["avg_pnl30_bps"] else 0.0,
        "std_pnl30_bps": round(ext["std_pnl30_bps"], 4),
        "downside_p10_bps": round(ext["downside_p10_bps"], 4) if ext["downside_p10_bps"] == ext["downside_p10_bps"] else 0.0,
        "downside_p05_bps": round(ext["downside_p05_bps"], 4) if ext["downside_p05_bps"] == ext["downside_p05_bps"] else 0.0,
        "profitable_rate": round(ext["profitable_rate"], 4),
        "as_rate": round(ext["as_rate"], 4),
        "avg_as_loss_bps": round(ext["avg_as_loss_bps"], 4),
        "reprice_rate": round(ext["reprice_rate"], 4),
        "avg_reprice_drift_bps": round(ext["avg_reprice_drift_bps"], 4),
        "vg_trigger_rate": round(ext["vg_trigger_rate"], 4),
    }


def _load_all_records(results_dir: Path) -> list[MetricRecord]:
    """fill_records JSONL を全読み込み."""
    return [
        cast(MetricRecord, record)
        for record in iter_fill_record_objects_glob(
            results_dir,
            include_emergency=False,
        )
    ]


def _yaml_file_signature(path: Path) -> tuple[int, int]:
    """Return file signature for cached YAML reads."""
    try:
        st = path.stat()
    except OSError:
        return -1, -1
    return st.st_mtime_ns, st.st_size


@lru_cache(maxsize=8)
def _read_judgment_mapping_cached(
    path_str: str,
    mtime_ns: int,
    size: int,
) -> dict[str, object]:
    del mtime_ns, size  # cache key only
    raw = read_yaml(path_str)
    return raw if isinstance(raw, dict) else {}


def _read_judgment_mapping(path: Path) -> dict[str, object]:
    mtime_ns, size = _yaml_file_signature(path)
    return _read_judgment_mapping_cached(str(path), mtime_ns, size)


def run_dashboard(
    results_dir: str = "results/v460/fill_test",
    *,
    with_judgment: bool = False,
    ab_criteria: ABJudgmentCriteria | None = None,
    trending_criteria: TrendingEvalCriteria | None = None,
) -> DashboardResult:
    """3指標ダッシュボードを生成.

    Args:
        results_dir: fill_records ディレクトリ.
        with_judgment: True で P0-B/C 判定を実行.
        ab_criteria: A/B 判定基準 (None=YAML or default).
        trending_criteria: trending_down sell 判定基準.

    Returns:
        DashboardResult: fill_rate / avg_pnl30 / downside_p10 を side×regime で算出。
    """
    results_path = Path(results_dir)
    records = _load_all_records(results_path)

    result: DashboardResult = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_path),
        "total_records": len(records),
    }

    side_groups: dict[str, MetricsAccumulator] = defaultdict(MetricsAccumulator)
    regime_side_groups: dict[str, MetricsAccumulator] = defaultdict(MetricsAccumulator)
    td_by_day: dict[str, MetricsAccumulator] = defaultdict(MetricsAccumulator)
    filled_count = 0
    for r in records:
        side = str(r.get("side", "unknown"))
        regime = str(r.get("regime") or "none")
        side_groups[side].add(r)
        regime_side_groups[f"{regime}:{side}"].add(r)
        if not r.get("filled"):
            continue
        filled_count += 1
        if regime != "trending_down" or side != "sell":
            continue
        day = format_utc_day(safe_to_finite(r.get("timestamp")))
        if day is None:
            continue
        td_by_day[day].add(r)

    result["total_filled"] = filled_count
    result["overall_fill_rate"] = round(filled_count / len(records), 4) if records else 0.0

    # === Side 別サマリー (159# §3.1: 3指標) ===
    side_summary: dict[str, SideMetrics] = {}
    for side in ["buy", "sell"]:
        if side in side_groups:
            side_summary[side] = _side_metrics_from_extended(
                side_groups[side].to_extended_metrics()
            )
    result["side_summary"] = side_summary

    # === Regime × Side 詳細 ===
    detail: list[RegimeSideMetrics] = []
    for key in sorted(regime_side_groups.keys()):
        regime, side = key.split(":", 1)
        detail.append({
            "regime": regime,
            "side": side,
            "metrics": _side_metrics_from_extended(
                regime_side_groups[key].to_extended_metrics()
            ),
        })
    result["regime_side_detail"] = detail

    # === P0-C: trending 日次テンプレート ===
    # trending_down × sell の日別集計
    trending_daily: list[dict[str, object]] = []
    for day in sorted(td_by_day.keys()):
        day_metrics = td_by_day[day].to_base_metrics()
        trending_daily.append({
            "day": day,
            "n_filled": day_metrics["n_filled"],
            "avg_pnl30_bps": (
                round(day_metrics["avg_pnl30_bps"], 4)
                if day_metrics["avg_pnl30_bps"] == day_metrics["avg_pnl30_bps"]
                else None
            ),
            "p10_bps": (
                round(day_metrics["downside_p10_bps"], 4)
                if day_metrics["n_filled"] >= 3
                and day_metrics["downside_p10_bps"] == day_metrics["downside_p10_bps"]
                else None
            ),
        })
    result["trending_daily"] = trending_daily

    # === 160# P0-B/C: judgment 統合 ===
    result["ab_judgment"] = None
    result["ab_judgment_incl_none"] = None  # 303# A: none 含有版
    result["trending_eval"] = None
    result["per_regime_judgment"] = None

    if with_judgment:
        # P0-B: side=sell を variant, side=buy を control として3指標判定
        # (sell offset 最適化が主テーマのため sell が variant 扱い)
        sell_records = [dict(r) for r in records if str(r.get("side", "unknown")) == "sell"]
        buy_records = [dict(r) for r in records if str(r.get("side", "unknown")) == "buy"]
        if sell_records:
            ab_result = evaluate_ab_variant(
                variant_records=sell_records,
                control_records=buy_records,
                criteria=ab_criteria,
                variant_label="sell",
                control_label="buy",
            )
            result["ab_judgment"] = _ab_result_to_dict(ab_result)

            # 303# A: none レジーム含有版 (301# F1 楽観バイアス除去)
            from copy import copy
            if ab_criteria is not None:
                criteria_incl = copy(ab_criteria)
                criteria_incl.exclude_regimes = []
            else:
                criteria_incl = ABJudgmentCriteria(exclude_regimes=[])
            ab_result_incl = evaluate_ab_variant(
                variant_records=sell_records,
                control_records=buy_records,
                criteria=criteria_incl,
                variant_label="sell",
                control_label="buy",
            )
            result["ab_judgment_incl_none"] = _ab_result_to_dict(ab_result_incl)

        # P0-C: trending_down sell 実測評価
        all_records_dict = [dict(r) for r in records]
        trending_result = evaluate_trending_down_sell(
            records=all_records_dict,
            criteria=trending_criteria,
        )
        result["trending_eval"] = {
            "verdict": trending_result.verdict.value,
            "n_filled": trending_result.n_filled,
            "n_total": trending_result.n_total,
            "avg_pnl30_bps": trending_result.avg_pnl30_bps,
            "downside_p10_bps": trending_result.downside_p10_bps,
            "profitable_rate": trending_result.profitable_rate,
            "counterfactual_gain_bps": trending_result.counterfactual_gain_bps,
            "detail": trending_result.detail,
            "daily_breakdown": trending_result.daily_breakdown,
            "summary": trending_result.summary(),
        }

        # Per-regime A/B judgment (regime フィルタで隠れた健全性を可視化)
        per_regime_results = evaluate_per_regime(
            variant_records=sell_records,
            control_records=buy_records,
            criteria=ab_criteria,
            variant_label="sell",
            control_label="buy",
            target_regimes=["ranging", "trending", "trending_down", "trending_up"],
        )
        result["per_regime_judgment"] = [
            {
                "regime": pr.regime,
                "overall": pr.result.overall.value,
                "n_variant": pr.result.n_variant,
                "n_control": pr.result.n_control,
                "criteria": [
                    {
                        "name": c.name,
                        "verdict": c.verdict.value,
                        "value": c.value,
                        "threshold": c.threshold,
                        "detail": c.detail,
                    }
                    for c in pr.result.criteria
                ],
                "summary": pr.result.summary(),
            }
            for pr in per_regime_results
        ]

    return result


def _print_dashboard(result: DashboardResult) -> None:
    """ダッシュボードを人間可読形式で出力."""
    print("\n" + "=" * 74)
    print("  159# Side × Regime 3指標ダッシュボード")
    print("=" * 74)
    print(f"  Total: {result['total_records']} records, "
          f"{result['total_filled']} filled ({result['overall_fill_rate']:.1%})")

    for side in ["buy", "sell"]:
        sm = result.get("side_summary", {}).get(side)
        if not sm:
            continue
        print(f"\n  --- {side.upper()} ---")
        print(f"    fill_rate:     {sm['fill_rate']:.1%} ({sm['n_filled']}/{sm['n_total']})")
        print(f"    avg_pnl30:     {sm['avg_pnl30_bps']:+.4f} bps")
        print(f"    downside_p10:  {sm['downside_p10_bps']:+.4f} bps")
        print(f"    downside_p05:  {sm['downside_p05_bps']:+.4f} bps")
        print(f"    profitable:    {sm['profitable_rate']:.1%}")
        print(f"    AS rate:       {sm['as_rate']:.1%}, avg AS loss: {sm['avg_as_loss_bps']:+.4f} bps")
        rr = sm.get('reprice_rate', 0.0)
        rd = sm.get('avg_reprice_drift_bps', 0.0)
        vg = sm.get('vg_trigger_rate', 0.0)
        if rr > 0 or vg > 0:
            print(f"    reprice:       {rr:.1%} (drift {rd:+.4f} bps), VG trigger: {vg:.1%}")

    print(f"\n  --- Regime × Side Detail ---")
    for item in result.get("regime_side_detail", []):
        m = item["metrics"]
        filled_n = m["n_filled"]
        if filled_n == 0:
            continue
        print(f"    {item['regime']:15s} {item['side']:4s}  "
              f"fill={m['fill_rate']:.1%}  "
              f"pnl30={m['avg_pnl30_bps']:+.4f}  "
              f"p10={m['downside_p10_bps']:+.4f}  "
              f"AS={m['as_rate']:.1%}  "
              f"n={filled_n}")

    td = result.get("trending_daily", [])
    if td:
        print(f"\n  --- Trending Down Sell (Daily) ---")
        for entry in td:
            avg = entry.get("avg_pnl30_bps")
            avg_str = f"{avg:+.4f}" if avg is not None else "N/A"
            print(f"    {entry['day']}: n={entry['n_filled']}, avg_pnl30={avg_str} bps")

    # 160# P0-B/C: judgment 結果出力
    ab_j = result.get("ab_judgment")
    if ab_j:
        print(f"\n  --- Side Comparison (excl. none) ---")
        print(f"  {ab_j['summary']}")

    # 303# A: none 含有版
    ab_j_incl = result.get("ab_judgment_incl_none")
    if ab_j_incl:
        print(f"\n  --- Side Comparison (incl. none) ---")
        print(f"  {ab_j_incl['summary']}")

    te = result.get("trending_eval")
    if te:
        print(f"\n  --- P0-C: Trending Down Sell Evaluation ---")
        print(f"  {te['summary']}")

    prj = result.get("per_regime_judgment")
    if prj:
        print(f"\n  --- Per-Regime Side Comparison ---")
        for entry in prj:
            regime = str(entry["regime"])
            overall = str(entry["overall"])
            flag = "✅" if overall == "pass" else (
                "⚠️" if overall == "insufficient" else "❌"
            )
            nv = entry.get("n_variant", 0)
            nc = entry.get("n_control", 0)
            print(f"    {flag} {regime:15s}  [{overall.upper()}]  "
                  f"sell(n={nv}) vs buy(n={nc})")
            criteria_list = cast(list[dict[str, object]], entry.get("criteria", []))
            for c in criteria_list:
                cv = str(c.get("verdict", ""))
                cf = "✅" if cv == "pass" else "❌"
                print(f"      {cf} {c['name']}: {c['detail']}")

    print("=" * 74)


def _load_judgment_config(
    config_path: str | None = None,
) -> tuple[ABJudgmentCriteria | None, TrendingEvalCriteria | None]:
    """YAML から judgment 設定をロード."""
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml")
    try:
        cfg = _read_judgment_mapping(Path(config_path))
        judgment = cast(dict[str, object], cfg.get("judgment", {}))
        ab = None
        trending = None
        if "ab_criteria" in judgment:
            ab = ABJudgmentCriteria.from_dict(
                cast(dict[str, object], judgment["ab_criteria"]),
            )
        if "trending_down_sell" in judgment:
            trending = TrendingEvalCriteria.from_dict(
                cast(dict[str, object], judgment["trending_down_sell"]),
            )
        return ab, trending
    except Exception as e:
        logger.debug("Failed to parse judgment: %s", e)
        return None, None


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(
        description="159# P0-B/C: Side × Regime 3指標ダッシュボード",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/v460/fill_test",
        help="fill_records ディレクトリ",
    )
    parser.add_argument("--json", action="store_true", help="JSON 出力")
    parser.add_argument("--with-judgment", action="store_true",
                        help="160# P0-B/C: A/B判定 + trending_down sell 評価を実行")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 設定ファイルパス (judgment section を参照)")
    args = parser.parse_args()

    ab_criteria = None
    trending_criteria = None
    if args.with_judgment:
        ab_criteria, trending_criteria = _load_judgment_config(args.config)

    result = run_dashboard(
        results_dir=args.results_dir,
        with_judgment=args.with_judgment,
        ab_criteria=ab_criteria,
        trending_criteria=trending_criteria,
    )
    _print_dashboard(result)

    if args.json:
        print(f"\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
