"""
v460 Gate 統計検定モジュール.

001# §5.1–§5.3 準拠. Holm-Bonferroni + p 平均法 + G1 判定アルゴリズム.

- holm_bonferroni_gate(): multi-target 判定 (Holm 補正付き)
- p_mean_gate(): fold/seed p 値の幾何平均統合
- g1_judgment(): target 単位 fold p-mean → Holm 補正 → AND 判定 (§5.3 厳密仕様)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's Delta effect size — O(n log n) via Mann-Whitney U.

    旧実装は O(n*m) のネストループで大規模データに非対応.
    Mann-Whitney U 統計量から d = 2U/(n1*n2) - 1 で導出.

    Args:
        x: Model scores.
        y: Baseline scores.

    Returns:
        Cliff's Delta in [-1, +1].
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    # Mann-Whitney U gives count of x[i] > y[j] pairs (+ 0.5 for ties)
    # Cliff's delta = 2U / (n1*n2) - 1
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(2.0 * u_stat / (n1 * n2) - 1.0)


def compare_single(
    model: list[float],
    baseline: list[float],
) -> tuple[float, float]:
    """Single comparison: Mann-Whitney U p-value + Cliff's Delta.

    Args:
        model: Model score samples.
        baseline: Baseline score samples.

    Returns:
        (p_value, cliff_d)
    """
    if len(model) < 2 or len(baseline) < 2:
        return 1.0, 0.0
    _, p = mannwhitneyu(model, baseline, alternative="greater")
    return p, cliffs_delta(model, baseline)


def holm_bonferroni_gate(
    results: dict[str, tuple[list[float], list[float]]],
    alpha: float = 0.05,
    min_effect: float = 0.33,
) -> dict[str, dict]:
    """G1 multi-target 判定: Holm-Bonferroni 補正付き.

    001# §5.1 準拠.

    Args:
        results: {target_name: (model_scores, baseline_scores)}
        alpha: Family-wise error rate.
        min_effect: Cliff's Delta minimum threshold.

    Returns:
        {target_name: {"pass": bool, "p_raw": float, "p_holm": float, "d": float}}
    """
    if not results:
        return {}

    # 1. Raw p + effect size for all targets
    raw: dict[str, tuple[float, float]] = {}
    for k, v in results.items():
        raw[k] = compare_single(*v)

    # 2. Sort by p ascending → Holm correction
    sorted_keys = sorted(raw, key=lambda k: raw[k][0])
    m = len(sorted_keys)
    out: dict[str, dict] = {}
    gate_open = True  # Holm: first non-rejection → all subsequent non-rejected

    for rank, key in enumerate(sorted_keys):
        p_raw, d = raw[key]
        holm_alpha = alpha / (m - rank)
        rejected = gate_open and (p_raw < holm_alpha) and (abs(d) > min_effect)
        if not (p_raw < holm_alpha):
            gate_open = False
        out[key] = {
            "pass": rejected,
            "p_raw": round(p_raw, 6),
            "p_holm": round(min(p_raw * (m - rank), 1.0), 6),
            "d": round(d, 4),
        }

    return out


def p_mean_gate(
    fold_p_values: list[float],
    alpha: float = 0.05,
) -> dict:
    """p 平均法による Gate 判定.

    001# §5.2 準拠. 既存 ztb.metrics.metrics.p_mean_method() を活用.

    Args:
        fold_p_values: Per-fold Mann-Whitney U p values.
        alpha: Significance level.

    Returns:
        {"p_geometric": float, "p_arithmetic": float, "n_folds": int, "pass": bool}
    """
    from ztb.metrics.metrics import p_mean_method

    if not fold_p_values:
        return {"p_geometric": 1.0, "p_arithmetic": 1.0, "n_folds": 0, "pass": False}

    p_geo = p_mean_method(fold_p_values, method="geometric")
    p_arith = p_mean_method(fold_p_values, method="arithmetic")
    return {
        "p_geometric": round(float(p_geo), 6),
        "p_arithmetic": round(float(p_arith), 6),
        "n_folds": len(fold_p_values),
        "pass": float(p_geo) < alpha,
    }


def g1_judgment(
    fold_results: dict[str, list[tuple[list[float], list[float]]]],
    alpha: float = 0.05,
    min_effect: float = 0.33,
) -> dict:
    """G1 判定: p-mean (target 単位) → Holm 補正 → AND.

    001# §5.3 厳密仕様:
      Step A: target 単位の fold/seed p 値統合 (p_mean_gate geometric)
      Step B: Holm-Bonferroni 補正 (family = len(targets))
      Step C: AND 判定 — holm_pass ∧ pmean_pass ∧ |d| > min_effect

    Args:
        fold_results: {target_name: [(model_fold1, baseline_fold1), ...]}
        alpha: Family-wise error rate.
        min_effect: Cliff's Delta minimum threshold.

    Returns:
        {"g1_pass": bool, "passed_targets": [...], "details": {...}}
    """
    from ztb.metrics.metrics import p_mean_method

    if not fold_results:
        return {"g1_pass": False, "passed_targets": [], "details": {}}

    # Step A: Target-level fold p-values → geometric mean
    target_p_geo: dict[str, float] = {}
    target_effects: dict[str, float] = {}
    target_pmean_pass: dict[str, bool] = {}

    for tgt, folds in fold_results.items():
        fold_ps: list[float] = []
        all_model: list[float] = []
        all_baseline: list[float] = []

        for model_scores, baseline_scores in folds:
            if len(model_scores) < 2 or len(baseline_scores) < 2:
                fold_ps.append(1.0)
                continue
            _, p = mannwhitneyu(model_scores, baseline_scores, alternative="greater")
            fold_ps.append(float(p))
            all_model.extend(model_scores)
            all_baseline.extend(baseline_scores)

        p_geo = float(p_mean_method(fold_ps, method="geometric")) if fold_ps else 1.0
        target_p_geo[tgt] = p_geo
        target_effects[tgt] = cliffs_delta(all_model, all_baseline) if all_model else 0.0
        target_pmean_pass[tgt] = p_geo < alpha

    # Step B: Holm-Bonferroni correction
    sorted_targets = sorted(target_p_geo, key=lambda t: target_p_geo[t])
    m = len(sorted_targets)
    holm_pass: dict[str, bool] = {}
    gate_open = True

    for rank, tgt in enumerate(sorted_targets):
        holm_alpha = alpha / (m - rank)
        rejected = gate_open and (target_p_geo[tgt] < holm_alpha)
        if not (target_p_geo[tgt] < holm_alpha):
            gate_open = False
        holm_pass[tgt] = rejected

    # Step C: AND
    passed_targets = [
        tgt for tgt in sorted_targets
        if holm_pass[tgt]
        and target_pmean_pass[tgt]
        and abs(target_effects[tgt]) > min_effect
    ]

    return {
        "g1_pass": len(passed_targets) > 0,
        "passed_targets": passed_targets,
        "details": {
            tgt: {
                "p_geo": round(target_p_geo[tgt], 6),
                "pmean_pass": target_pmean_pass[tgt],
                "holm_pass": holm_pass[tgt],
                "cliff_d": round(target_effects[tgt], 4),
            }
            for tgt in sorted_targets
        },
    }
