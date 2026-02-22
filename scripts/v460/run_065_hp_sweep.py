#!/usr/bin/env python3
"""
065# Phase 2: AS-LR ハイパーパラメータ最適化 + Two-Tier モデル.

Goals:
  1) 現行 model A (enriched, OB+spread) の C/k チューニング
  2) Curated trade-only fallback (058# 実績ベース特徴量)
  3) Two-tier: OB あり→A', OB なし→curated fallback

Usage:
    python scripts/v460/run_065_hp_sweep.py
"""

from __future__ import annotations

import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_enriched_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.walk_forward_as import run_walk_forward
from ztb.io.json_io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 058# 実績 + 比較検証で安定した OB-free features
CURATED_TRADE_FEATURES = [
    "log_queue_wait",  # always top, 058# GB top1
    "edge_bps",        # always selected in B/D/E/F
    "vpin_60s",        # 058# Ridge top4
    "vpin_30s",        # multi-timeframe
    "vpin_300s",       # multi-timeframe
    "price_velocity_60s",  # 058# Ridge top3
    "buy_ratio",       # trade composition
    "tfi_300s",        # longer-term TFI
    "hour_cos",        # 058# Ridge top5
    "hour_sin",        # time cyclical pair
    "side_aligned_tfi",    # 058# GB top4
    "side_aligned_velocity",  # interaction
]


def run_param(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series | None,
    k: int,
    min_train: int = 50,
    step: int = 20,
    embargo: int = 2,
) -> dict:
    """Walk-forward with given parameters."""
    k_adj = min(k, X.shape[1])
    wf = run_walk_forward(X, y, pnl, min_train=min_train, step=step, embargo=embargo, k=k_adj)
    wf["name"] = name
    wf["n_features"] = X.shape[1]
    wf["n_samples"] = len(X)
    agg = wf.get("aggregate", {})
    skip = wf.get("skip_simulation", {})
    feat = wf.get("feature_stability", {})
    roc = agg.get("roc_auc_mean")
    s20 = skip.get("skip20_improvement_bps", 0)
    s10 = skip.get("skip10_improvement_bps", 0)
    bl = skip.get("baseline_pnl_bps", 0)
    jac = feat.get("jaccard_stability", 0)
    always = feat.get("always_selected", [])
    logger.info(
        f"  {name}: ROC={roc}, Skip20%={s20:+.3f}, Skip10%={s10:+.3f}, "
        f"baseline={bl:.3f}, Jaccard={jac:.3f}, always={always[:3]}"
    )
    return wf


def main() -> None:
    logger.info("Loading fill records")
    df = load_fill_records()
    enriched_df = enrich_fill_records(df)

    # --- Dataset A: Enriched (require_spread=True) ---
    X_a, y_a = build_enriched_as_features(enriched_df, require_spread=True)
    filled_mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl_a = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float).reindex(X_a.index)

    # --- Dataset B: Curated trade-only (require_spread=False) ---
    X_full, y_full = build_enriched_as_features(enriched_df, require_spread=False)
    curated_cols = [c for c in CURATED_TRADE_FEATURES if c in X_full.columns]
    X_curated = X_full[curated_cols]
    pnl_full = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float).reindex(X_curated.index)

    results: list[dict] = []

    # ===== Part 1: Model A hyperparameter sweep =====
    logger.info("\n" + "=" * 60)
    logger.info("Part 1: Enriched model (A) — k sweep")
    logger.info("=" * 60)

    for k in [3, 5, 8, 12]:
        name = f"A(k={k})"
        r = run_param(name, X_a, y_a, pnl_a, k=k)
        results.append(r)

    # ===== Part 2: Curated trade-only =====
    logger.info("\n" + "=" * 60)
    logger.info(f"Part 2: Curated trade-only ({len(curated_cols)} features, 058# based)")
    logger.info(f"  Features: {curated_cols}")
    logger.info("=" * 60)

    for k in [3, 5, 8]:
        name = f"Curated(k={k})"
        r = run_param(name, X_curated, y_full, pnl_full, k=k)
        results.append(r)

    # Also try using ALL curated features without selection
    k_all = len(curated_cols)
    r_all = run_param(f"Curated(k=all-{k_all})", X_curated, y_full, pnl_full, k=k_all)
    results.append(r_all)

    # ===== Summary =====
    logger.info("\n" + "=" * 60)
    logger.info("FULL SWEEP SUMMARY")
    logger.info("=" * 60)

    # Sort by skip20 improvement
    for r in sorted(
        results,
        key=lambda x: x.get("skip_simulation", {}).get("skip20_improvement_bps", -999),
        reverse=True,
    ):
        agg = r.get("aggregate", {})
        skip = r.get("skip_simulation", {})
        feat = r.get("feature_stability", {})
        logger.info(
            f"  {r['name']:25s} | ROC={agg.get('roc_auc_mean', 0):.4f} | "
            f"Skip20%={skip.get('skip20_improvement_bps', 0):+.3f} bps | "
            f"Jaccard={feat.get('jaccard_stability', 0):.3f} | "
            f"{r['n_features']}f/{r['n_samples']}s"
        )

    # Best enriched for primary
    enriched_results = [r for r in results if r["name"].startswith("A(")]
    best_enriched = max(
        enriched_results,
        key=lambda r: r.get("skip_simulation", {}).get("skip20_improvement_bps", -999),
    )

    # Best curated for fallback
    curated_results = [r for r in results if r["name"].startswith("Curated(")]
    best_curated = max(
        curated_results,
        key=lambda r: r.get("skip_simulation", {}).get("skip20_improvement_bps", -999),
    )

    logger.info("\n" + "=" * 60)
    logger.info("TWO-TIER RECOMMENDATION")
    logger.info("=" * 60)
    skip_e = best_enriched.get("skip_simulation", {})
    skip_c = best_curated.get("skip_simulation", {})
    logger.info(
        f"  Primary (OB あり):  {best_enriched['name']} → "
        f"Skip20%={skip_e.get('skip20_improvement_bps', 0):+.3f} bps"
    )
    logger.info(
        f"  Fallback (OB なし): {best_curated['name']} → "
        f"Skip20%={skip_c.get('skip20_improvement_bps', 0):+.3f} bps"
    )

    # Save results
    out_dir = _PROJECT_ROOT / "docs/v460"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "065_hp_sweep.json"
    write_json(json_path, results, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
