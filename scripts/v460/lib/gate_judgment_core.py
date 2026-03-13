"""G2/G3 Gate 判定 — 共通コアロジック.

399# 重複統合:
  run_experiment.py (_evaluate_g2_from_results / _evaluate_g3_from_results) と
  run_gate_check.py (run_g2_judgment / run_g3_judgment) で二重実装されていた
  G2/G3 判定ロジックを単一モジュールに抽出。

  各呼び出し元は:
    - run_gate_check.py: ファイルパス → JSON読み込み → 本モジュール呼び出し
    - run_experiment.py: dict (in-memory) → 本モジュール呼び出し

  384# HIGH-1 修正 (error_seeds チェック) も統合版に包含。
  run_gate_check.py 側にはこの修正が欠落していたバグも同時に解消。
"""

from __future__ import annotations

import statistics as _stats


def evaluate_g2_checks(
    seed_results: list[dict],
    convergence: dict,
    thresholds: dict,
) -> dict[str, object]:
    """G2-train gate 判定コアロジック.

    Args:
        seed_results: [{seed, gross_roi, ...}, ...]
        convergence: {roi_variance_pct_after_30k: float}
        thresholds: g2_train 閾値 dict

    Returns:
        {gate, gate_result, checks, n_seeds} dict
    """
    if not seed_results:
        return {
            "gate": "G2-train",
            "gate_result": "NO_DATA",
            "checks": {},
            "error": "No seed results",
        }

    # 384# HIGH-1: seed crash → ROI=0.0 がゲートを通過する問題を修正
    error_seeds = [s for s in seed_results if "error" in s]
    if error_seeds:
        error_info = [
            f"seed={s.get('seed', '?')}: {s.get('error', '?')}"
            for s in error_seeds
        ]
        return {
            "gate": "G2-train",
            "gate_result": "ERROR",
            "checks": {},
            "reason": f"{len(error_seeds)} seed(s) crashed: {'; '.join(error_info)}",
            "n_seeds": len(seed_results),
        }

    checks: dict[str, dict[str, object]] = {}

    # E1: gross > 0 の seed 比率 >= 75%
    min_ratio = float(thresholds.get("min_positive_seed_ratio", 0.75))
    positive_seeds = sum(
        1 for s in seed_results if float(s.get("gross_roi", 0)) > 0
    )
    ratio = positive_seeds / len(seed_results)
    checks["positive_seed_ratio"] = {
        "value": ratio,
        "threshold": min_ratio,
        "detail": f"{positive_seeds}/{len(seed_results)}",
        "pass": ratio >= min_ratio,
    }

    # E2: seed 間標準偏差チェック
    if "max_roi_seed_std" in thresholds:
        max_seed_std = float(thresholds.get("max_roi_seed_std", 0.03))
        seed_std_values = [float(s.get("gross_roi", 0)) for s in seed_results]
        check_name = "roi_seed_std"
    else:
        max_seed_std = float(thresholds.get("max_ic_seed_std", 0.03))
        seed_std_values = [float(s.get("ic_mean", 0)) for s in seed_results]
        check_name = "ic_seed_std"
    seed_std = (
        _stats.stdev(seed_std_values) if len(seed_std_values) >= 2 else 0.0
    )
    checks[check_name] = {
        "value": seed_std,
        "threshold": max_seed_std,
        "pass": seed_std <= max_seed_std,
    }

    # E3: 30K以降の ROI 変動 <= 5%
    max_roi_var = float(thresholds.get("max_roi_variance_pct", 5.0))
    roi_var = (
        float(convergence.get("roi_variance_pct_after_30k", 0.0))
        if isinstance(convergence, dict)
        else 0.0
    )
    checks["convergence"] = {
        "value": roi_var,
        "threshold": max_roi_var,
        "pass": roi_var <= max_roi_var,
    }

    # E4: worst-seed ROI > threshold
    worst_min = float(thresholds.get("worst_seed_min_roi", -0.02))
    roi_list = [float(s.get("gross_roi", 0)) for s in seed_results]
    worst_roi = min(roi_list) if roi_list else 0.0
    checks["worst_seed_roi"] = {
        "value": worst_roi,
        "threshold": worst_min,
        "pass": worst_roi > worst_min,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "gate": "G2-train",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "n_seeds": len(seed_results),
    }


def evaluate_g3_checks(
    seed_metrics: list[dict],
    thresholds: dict,
) -> dict[str, object]:
    """G3-pnl gate 判定コアロジック.

    Args:
        seed_metrics: [{pf, sharpe_annual, max_drawdown, avg_gross_per_trade, avg_fee_per_trade}, ...]
        thresholds: g3_pnl 閾値 dict

    Returns:
        {gate, gate_result, checks, n_seeds} dict
    """
    if not seed_metrics:
        return {
            "gate": "G3-pnl",
            "gate_result": "NO_DATA",
            "checks": {},
            "error": "No seed metrics",
        }

    checks: dict[str, dict[str, object]] = {}

    # E1: PF median > 1.05
    min_pf_median = float(thresholds.get("min_pf_median", 1.05))
    pfs = sorted(float(s.get("pf", 0)) for s in seed_metrics)
    pf_median = _stats.median(pfs)
    checks["pf_median"] = {
        "value": pf_median,
        "threshold": min_pf_median,
        "pass": pf_median > min_pf_median,
    }

    # E2: PF worst > 0.95
    min_pf_worst = float(thresholds.get("min_pf_worst", 0.95))
    pf_worst = min(pfs)
    checks["pf_worst"] = {
        "value": pf_worst,
        "threshold": min_pf_worst,
        "pass": pf_worst > min_pf_worst,
    }

    # E3: gross > fee
    gross_gt_fee_required = bool(thresholds.get("gross_gt_fee", True))
    total_gross = sum(
        float(s.get("avg_gross_per_trade", 0)) for s in seed_metrics
    )
    total_fee = sum(
        float(s.get("avg_fee_per_trade", 0)) for s in seed_metrics
    )
    gross_gt_fee = total_gross > total_fee
    checks["gross_gt_fee"] = {
        "value": gross_gt_fee,
        "threshold": gross_gt_fee_required,
        "pass": gross_gt_fee if gross_gt_fee_required else True,
    }

    # E4: Max DD < 15%
    max_dd_threshold = float(thresholds.get("max_drawdown", 0.15))
    worst_dd = max(float(s.get("max_drawdown", 0)) for s in seed_metrics)
    checks["max_drawdown"] = {
        "value": worst_dd,
        "threshold": max_dd_threshold,
        "pass": worst_dd < max_dd_threshold,
    }

    # E5: Sharpe annual median > 0.8
    min_sharpe = float(thresholds.get("min_sharpe_annual", 0.8))
    sharpes = [float(s.get("sharpe_annual", 0)) for s in seed_metrics]
    sharpe_median = _stats.median(sharpes)
    checks["sharpe_annual"] = {
        "value": sharpe_median,
        "threshold": min_sharpe,
        "pass": sharpe_median > min_sharpe,
    }

    # E6: reward-profit alignment — median corr > 0 (392# P0-3)
    min_corr = float(thresholds.get("min_reward_profit_corr_median", 0.0))
    corrs = [float(s.get("reward_profit_corr", 0)) for s in seed_metrics]
    if corrs:
        corr_median = _stats.median(corrs)
    else:
        corr_median = 0.0
    checks["reward_profit_corr_median"] = {
        "value": corr_median,
        "threshold": min_corr,
        "pass": corr_median > min_corr,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "gate": "G3-pnl",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "n_seeds": len(seed_metrics),
    }
