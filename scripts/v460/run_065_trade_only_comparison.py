#!/usr/bin/env python3
"""
065# Trade-Only AS-LR モデル比較検証.

問題: require_spread=True → 166 samples / 39 features (n/p=4.3, 過学習リスク)
改善: require_spread=False + OB列除外 → 284 samples / ~25 features (n/p=11.4)

058# Ridge Top5 のうち 4/5 が OB 不要 (trade + time + regime) の実績に基づく。

3つのモデル構成を比較:
  A) 現行 enriched (require_spread=True, 39 features, 166 samples)
  B) Trade-only enriched (require_spread=False, OB列除外, ~25 features, 284 samples)
  C) Base-only (require_spread=False, OB/trade micro不使用, ~10 features, 284 samples)

Usage:
    python scripts/v460/run_065_trade_only_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    INTERACTION_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    V2_FEATURE_COLS,
    build_enriched_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.walk_forward_as import run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = _PROJECT_ROOT / "docs/v460"

# OB必須列 — これらは板スナップショットがマッチしないと NaN
OB_DEPENDENT_COLS = {
    "spread_bps_ob",
    "depth_imbalance_ob",
    "side_aligned_imbalance",
    # return/momentum columns are OB mid-price derived
    "return_30s",
    "return_60s",
    "return_300s",
    "realized_vol_300s",
    "side_aligned_return_30s",
    "side_aligned_return_60s",
    "side_aligned_return_300s",
}

# spread_at_order 依存列 — fill record 中の spread/offset が前半で NaN
SPREAD_DEPENDENT_COLS = {
    "spread_jpy",
    "offset_ratio",
}


def drop_ob_columns(X: pd.DataFrame) -> pd.DataFrame:
    """OB 依存列 + spread依存列を除外した trade-only feature set を返す."""
    drop_cols = [c for c in X.columns if c in OB_DEPENDENT_COLS or c in SPREAD_DEPENDENT_COLS]
    X_trade = X.drop(columns=drop_cols, errors="ignore")
    logger.info(
        f"Dropped {len(drop_cols)} OB/spread-dependent columns: {drop_cols}"
    )
    return X_trade


def drop_micro_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Micro/V2/interaction 列をすべて除外した base-only feature set を返す."""
    micro_all = (
        set(MICRO_FEATURE_COLS) | set(V2_FEATURE_COLS) | set(INTERACTION_FEATURE_COLS)
        | OB_DEPENDENT_COLS | SPREAD_DEPENDENT_COLS
    )
    drop_cols = [c for c in X.columns if c in micro_all]
    X_base = X.drop(columns=drop_cols, errors="ignore")
    logger.info(
        f"Dropped {len(drop_cols)} micro/OB/spread columns. "
        f"Remaining: {X_base.columns.tolist()}"
    )
    return X_base


def run_model_config(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series | None,
    *,
    min_train: int = 50,
    step: int = 20,
    embargo: int = 2,
    k: int = 8,
) -> dict:
    """単一モデル構成で walk-forward 実行."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {name} ({X.shape[1]} features, {len(X)} samples)")
    logger.info(f"  Features: {X.columns.tolist()}")

    # k は特徴量数以下に制限
    k_adj = min(k, X.shape[1])

    wf = run_walk_forward(
        X, y, pnl,
        min_train=min_train,
        step=step,
        embargo=embargo,
        k=k_adj,
    )
    wf["name"] = name
    wf["n_features"] = X.shape[1]
    wf["n_samples"] = len(X)
    wf["feature_cols"] = X.columns.tolist()

    agg = wf.get("aggregate", {})
    skip = wf.get("skip_simulation", {})
    feat = wf.get("feature_stability", {})

    logger.info(f"  ROC-AUC: {agg.get('roc_auc_mean', 'N/A')}")
    logger.info(f"  Brier: {agg.get('brier_mean', 'N/A')}")
    if skip:
        logger.info(f"  Skip20%: {skip.get('skip20_improvement_bps', 0):+.3f} bps")
        logger.info(f"  Skip10%: {skip.get('skip10_improvement_bps', 0):+.3f} bps")
        logger.info(f"  Baseline PnL: {skip.get('baseline_pnl_bps', 0):.3f} bps")
    logger.info(f"  Jaccard stability: {feat.get('jaccard_stability', 0):.3f}")
    logger.info(f"  Always selected: {feat.get('always_selected', [])}")

    return wf


def generate_comparison_report(results: list[dict]) -> str:
    """比較レポート Markdown 生成."""
    lines = [
        "# 065# Trade-Only AS-LR モデル比較検証",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 背景",
        "",
        "- 現行 AS-LR: `require_spread=True` → 166 samples, 39 features (n/p=4.3, 過学習リスク)",
        "- 058# Ridge Top5 の 4/5 が OB 不要 (trade + time + regime) → trade-only で十分な可能性",
        "- 板データマッチ率 347/491 (30% 欠損) → OB 依存は汎用性が低い",
        "",
        "## 比較対象",
        "",
        "| Model | require_spread | OB cols | Features | Samples | n/p ratio |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        n_f = r["n_features"]
        n_s = r["n_samples"]
        np_ratio = n_s / n_f if n_f > 0 else 0
        lines.append(
            f"| {r['name']} | — | — | {n_f} | {n_s} | {np_ratio:.1f} |"
        )

    lines += [
        "",
        "## Walk-Forward 結果比較",
        "",
        "| Model | ROC-AUC | Brier | Skip20% (bps) | Skip10% (bps) | "
        "Baseline PnL | Jaccard | N folds |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        agg = r.get("aggregate", {})
        skip = r.get("skip_simulation", {})
        feat = r.get("feature_stability", {})
        roc = agg.get("roc_auc_mean")
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        brier = agg.get("brier_mean")
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        s20 = skip.get("skip20_improvement_bps", 0)
        s10 = skip.get("skip10_improvement_bps", 0)
        bl = skip.get("baseline_pnl_bps", 0)
        jac = feat.get("jaccard_stability", 0)
        nf = agg.get("n_folds", 0)
        lines.append(
            f"| {r['name']} | {roc_str} | {brier_str} | "
            f"{s20:+.3f} | {s10:+.3f} | {bl:.3f} | {jac:.3f} | {nf} |"
        )

    # Feature stability per model
    lines += [
        "",
        "## 特徴量安定性",
        "",
    ]
    for r in results:
        feat = r.get("feature_stability", {})
        lines += [
            f"### {r['name']}",
            f"- Always selected: {feat.get('always_selected', [])}",
            f"- Ever selected: {feat.get('ever_selected', [])}",
            f"- Jaccard stability: {feat.get('jaccard_stability', 0):.3f}",
            "",
        ]

    # Per-fold details per model
    lines += [
        "## Per-Fold 詳細",
        "",
    ]
    for r in results:
        folds = r.get("folds", [])
        lines += [
            f"### {r['name']}",
            "",
            "| Fold | Train | Test | ROC-AUC | Brier | AS rate | Selected |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in folds:
            roc = f.get("roc_auc")
            roc_s = f"{roc:.4f}" if roc is not None else "N/A"
            sel = f.get("selected_features", [])[:4]
            lines.append(
                f"| {f['fold']} | {f['n_train']} | {f['n_test']} | "
                f"{roc_s} | {f['brier']:.4f} | {f['as_rate_test']:.3f} | "
                f"{', '.join(sel)} |"
            )
        lines.append("")

    # Recommendation
    lines += [
        "## 判定・推奨",
        "",
    ]

    # Find best model by skip20 improvement
    best = max(
        results,
        key=lambda r: r.get("skip_simulation", {}).get(
            "skip20_improvement_bps", -999
        ),
    )
    lines += [
        f"**最良モデル (Skip20%改善)**: {best['name']}",
        f"- Skip20% improvement: "
        f"{best.get('skip_simulation', {}).get('skip20_improvement_bps', 0):+.3f} bps",
        f"- n/p ratio: {best['n_samples'] / best['n_features']:.1f}",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    # --- Load & enrich ---
    logger.info("Loading fill records")
    df = load_fill_records()
    logger.info(f"Raw: {len(df)}")

    enriched_df = enrich_fill_records(df)

    # --- Config A: Current enriched (require_spread=True) ---
    X_a, y_a = build_enriched_as_features(enriched_df, require_spread=True)
    filled_mask_a = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl_a = df.loc[filled_mask_a, "post_fill_30s_pnl"].astype(float)
    pnl_a = pnl_a.reindex(X_a.index)

    # --- Config B: Trade-only enriched (require_spread=False, drop OB cols) ---
    X_b_full, y_b = build_enriched_as_features(enriched_df, require_spread=False)
    X_b = drop_ob_columns(X_b_full)
    filled_mask_b = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl_b = df.loc[filled_mask_b, "post_fill_30s_pnl"].astype(float)
    pnl_b = pnl_b.reindex(X_b.index)

    # --- Config C: Base-only (require_spread=False, drop ALL micro) ---
    X_c_full, y_c = build_enriched_as_features(enriched_df, require_spread=False)
    X_c = drop_micro_columns(X_c_full)
    filled_mask_c = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl_c = df.loc[filled_mask_c, "post_fill_30s_pnl"].astype(float)
    pnl_c = pnl_c.reindex(X_c.index)

    # --- Config D: Full enriched features (require_spread=False, OB列は NaN impute) ---
    X_d, y_d = build_enriched_as_features(enriched_df, require_spread=False)
    pnl_d = df.loc[filled_mask_b, "post_fill_30s_pnl"].astype(float)
    pnl_d = pnl_d.reindex(X_d.index)

    # --- Config E: Trade-only k=5 (少ない特徴量で安定化) ---
    X_e = X_b.copy()
    y_e = y_b.copy()
    pnl_e = pnl_b.copy()

    # --- Config F: Trade-only k=3 (最小特徴量) ---
    X_f = X_b.copy()
    y_f = y_b.copy()
    pnl_f = pnl_b.copy()

    # --- Run Walk-Forward for each ---
    results: list[dict] = []

    result_a = run_model_config(
        "A: Enriched (OB+spread=True, 現行)",
        X_a, y_a, pnl_a,
    )
    results.append(result_a)

    result_b = run_model_config(
        "B: Trade-only (OB除外, spread=False)",
        X_b, y_b, pnl_b,
    )
    results.append(result_b)

    result_c = run_model_config(
        "C: Base-only (micro全除外, spread=False)",
        X_c, y_c, pnl_c,
    )
    results.append(result_c)

    result_d = run_model_config(
        "D: Full-39 + spread=False (NaN impute)",
        X_d, y_d, pnl_d,
    )
    results.append(result_d)

    result_e = run_model_config(
        "E: Trade-only k=5 (安定化)",
        X_e, y_e, pnl_e,
        k=5,
    )
    results.append(result_e)

    result_f = run_model_config(
        "F: Trade-only k=3 (最小)",
        X_f, y_f, pnl_f,
        k=3,
    )
    results.append(result_f)

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    for r in results:
        agg = r.get("aggregate", {})
        skip = r.get("skip_simulation", {})
        logger.info(
            f"  {r['name']}: "
            f"ROC={agg.get('roc_auc_mean', 'N/A')}, "
            f"Skip20%={skip.get('skip20_improvement_bps', 0):+.3f} bps, "
            f"{r['n_features']}f / {r['n_samples']}s"
        )

    # --- Report ---
    report = generate_comparison_report(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "065_trade_only_comparison.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"\nReport: {report_path}")

    # JSON
    json_path = REPORT_DIR / "065_trade_only_comparison.json"
    json_path.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"JSON: {json_path}")

    # --- Best model recommendation ---
    best = max(
        results,
        key=lambda r: r.get("skip_simulation", {}).get(
            "skip20_improvement_bps", -999
        ),
    )
    print(f"\n{'='*60}")
    print(f"  Best: {best['name']}")
    skip = best.get("skip_simulation", {})
    print(f"  Skip20%: {skip.get('skip20_improvement_bps', 0):+.3f} bps")
    print(f"  n/p ratio: {best['n_samples'] / best['n_features']:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
