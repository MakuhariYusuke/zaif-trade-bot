#!/usr/bin/env python3
"""
065# AS-LR SkipGate 学習 + Walk-Forward 検証 + skip_gate_as.pkl 保存.

060/061# で構築した AS-LR パイプラインを、
蓄積 fill records データで学習し、walk-forward で検証した上で
models/v460/skip_gate_as.pkl に保存する。

Usage:
    python scripts/v460/run_065_as_lr_prep.py [--as-threshold 0.65]
"""

from __future__ import annotations

import argparse
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
from scripts.v460.ml.skip_gate import train_and_save_as_skip_gate
from scripts.v460.ml.walk_forward_as import expanding_window_splits, run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = _PROJECT_ROOT / "docs/v460"
MODEL_PATH = _PROJECT_ROOT / "models/v460/skip_gate_as.pkl"


def generate_report(
    wf_results: dict,
    model_meta: dict,
    n_records: int,
    n_features: int,
    as_threshold: float,
) -> str:
    """Generate AS-LR preparation report."""
    agg = wf_results.get("aggregate", {})
    skip = wf_results.get("skip_simulation", {})
    feat = wf_results.get("feature_stability", {})
    folds = wf_results.get("folds", [])

    lines = [
        "# 065# AS-LR SkipGate 学習・検証レポート",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Fill Records**: {n_records} samples",
        f"**Features**: {n_features} features",
        f"**AS Threshold**: {as_threshold}",
        f"**Model Path**: `{MODEL_PATH.relative_to(_PROJECT_ROOT)}`",
        "",
        "## 1. Walk-Forward 検証結果",
        "",
        f"- **Folds**: {agg.get('n_folds', 0)}",
        f"- **ROC-AUC (mean)**: {agg.get('roc_auc_mean', 'N/A')}",
        f"- **ROC-AUC (std)**: {agg.get('roc_auc_std', 'N/A')}",
        f"- **PR-AUC (mean)**: {agg.get('pr_auc_mean', 'N/A')}",
        f"- **Brier (mean)**: {agg.get('brier_mean', 'N/A')}",
        "",
    ]

    # Skip simulation
    if skip:
        lines += [
            "## 2. Skip Simulation (OOF)",
            "",
            f"- **Baseline PnL**: {skip.get('baseline_pnl_bps', 0):.3f} bps",
            f"- **Skip 20% 改善**: {skip.get('skip20_improvement_bps', 0):+.3f} bps",
            f"- **Skip 10% 改善**: {skip.get('skip10_improvement_bps', 0):+.3f} bps",
            f"- **Valid samples**: {skip.get('n_valid', 0)}",
            "",
        ]

    # Feature stability
    lines += [
        "## 3. Feature Stability",
        "",
        f"- **Jaccard stability**: {feat.get('jaccard_stability', 0):.3f}",
        f"- **Always selected ({feat.get('n_always', 0)})**: {feat.get('always_selected', [])}",
        f"- **Ever selected ({feat.get('n_ever', 0)})**: {feat.get('ever_selected', [])}",
        "",
    ]

    # Per-fold details
    lines += [
        "## 4. Per-Fold Results",
        "",
        "| Fold | Train | Test | ROC-AUC | PR-AUC | Brier | AS rate (test) | Selected Features |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        roc = f.get("roc_auc")
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        pr = f.get("pr_auc")
        pr_str = f"{pr:.4f}" if pr is not None else "N/A"
        feats = f.get("selected_features", [])[:4]
        lines.append(
            f"| {f['fold']} | {f['n_train']} | {f['n_test']} | "
            f"{roc_str} | {pr_str} | {f['brier']:.4f} | "
            f"{f['as_rate_test']:.3f} | {', '.join(feats)}... |"
        )

    # Model metadata
    lines += [
        "",
        "## 5. 学習済みモデル情報",
        "",
        f"- **Total samples**: {model_meta.get('n_samples', 0)}",
        f"- **AS rate**: {model_meta.get('as_rate', 0):.3f}",
        f"- **Selected features**: {model_meta.get('selected_features', [])}",
        "",
        "### Feature Importances (LR coefficient abs)",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    fi = model_meta.get("feature_importances", {})
    for feat_name, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {feat_name} | {imp:.4f} |")

    # Deployment recommendation
    lines += [
        "",
        "## 6. ph2 投入設定",
        "",
        "```yaml",
        "# configs/v460/fill_test.yaml skip_gate section",
        "skip_gate:",
        "  enabled: true",
        "  mode: as",
        f"  model_path: {MODEL_PATH.relative_to(_PROJECT_ROOT)}",
        f"  as_threshold: {as_threshold}",
        "  max_skip_rate: 0.3",
        "```",
        "",
        "### 判定基準 (200 cycle 評価)",
        "",
        "| 指標 | 継続条件 | 中止条件 |",
        "|---|---|---|",
        "| post_fill_30s_pnl mean | baseline 比改善 | baseline 以下 |",
        "| AS ratio | baseline 比低下 | 増加 |",
        "| fill rate | 劣化軽微 | 大幅悪化 |",
        "| skip rate | 設定範囲内 | 上限張り付き |",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="065# AS-LR SkipGate Prep")
    parser.add_argument(
        "--as-threshold",
        type=float,
        default=0.65,
        help="AS probability skip threshold (conservative: 0.65)",
    )
    parser.add_argument("--k", type=int, default=8, help="SelectKBest k")
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--embargo", type=int, default=2)
    args = parser.parse_args()

    # --- Step 1: Load & enrich ---
    logger.info("Step 1: Loading fill records and enriching with microstructure features")
    df = load_fill_records()
    logger.info(f"Raw fill records: {len(df)}")

    enriched_df = enrich_fill_records(df)
    X, y = build_enriched_as_features(enriched_df)
    logger.info(f"AS features: {X.shape[1]} features, {len(X)} samples")
    logger.info(f"AS rate: {y.mean():.3f}")
    logger.info(f"Feature columns: {X.columns.tolist()}")

    # PnL for skip simulation
    filled_mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float)
    pnl = pnl.reindex(X.index)

    # --- Step 2: Walk-Forward Validation ---
    logger.info("=" * 60)
    logger.info("Step 2: Walk-forward validation")
    wf_results = run_walk_forward(
        X,
        y,
        pnl,
        min_train=args.min_train,
        step=args.step,
        embargo=args.embargo,
        k=args.k,
    )

    if "error" in wf_results:
        logger.error(f"Walk-forward failed: {wf_results['error']}")
        sys.exit(1)

    agg = wf_results.get("aggregate", {})
    skip = wf_results.get("skip_simulation", {})

    logger.info(f"  Folds: {agg.get('n_folds', 0)}")
    logger.info(f"  ROC-AUC (mean): {agg.get('roc_auc_mean', 'N/A')}")
    logger.info(f"  Brier (mean): {agg.get('brier_mean', 'N/A')}")
    if skip:
        logger.info(f"  Skip20% improvement: {skip.get('skip20_improvement_bps', 0):+.3f} bps")
        logger.info(f"  Baseline PnL: {skip.get('baseline_pnl_bps', 0):.3f} bps")

    # --- Step 3: Train full model ---
    logger.info("=" * 60)
    logger.info("Step 3: Training full AS-LR model")
    gate = train_and_save_as_skip_gate(
        output_path=MODEL_PATH,
        as_threshold=args.as_threshold,
        k=args.k,
    )
    model_meta = gate.metadata

    logger.info(f"  Model saved: {MODEL_PATH}")
    logger.info(f"  Selected features: {model_meta.get('selected_features', [])}")
    logger.info(f"  AS threshold: {args.as_threshold}")

    # --- Step 4: Generate report ---
    logger.info("=" * 60)
    logger.info("Step 4: Generating report")

    report = generate_report(
        wf_results,
        model_meta,
        n_records=len(X),
        n_features=X.shape[1],
        as_threshold=args.as_threshold,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "065_as_lr_prep.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved: {report_path}")

    # Save walk-forward JSON
    json_path = REPORT_DIR / "065_as_lr_wf_results.json"
    json_path.write_text(
        json.dumps(wf_results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"WF results JSON saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"  065# AS-LR SkipGate Preparation Complete")
    print(f"  Model: {MODEL_PATH}")
    print(f"  ROC-AUC: {agg.get('roc_auc_mean', 'N/A')}")
    if skip:
        print(f"  Skip20% improvement: {skip.get('skip20_improvement_bps', 0):+.3f} bps")
    print(f"  AS threshold: {args.as_threshold}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
