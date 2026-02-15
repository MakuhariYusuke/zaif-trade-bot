#!/usr/bin/env python3
"""
065# Phase 2: Two-Tier AS-LR モデル保存.

HP sweep 結果に基づき:
  Primary:  A(k=12) — enriched + OB, require_spread=True, 166 samples
  Fallback: Curated(k=5) — OB-free trade features, require_spread=False, 284 samples

Usage:
    python scripts/v460/run_065_save_two_tier.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_enriched_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.skip_gate import SkipGate, SkipGateConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 058# 実績ベースのキュレーション特徴量
CURATED_TRADE_FEATURES = [
    "log_queue_wait",
    "edge_bps",
    "vpin_60s",
    "vpin_30s",
    "vpin_300s",
    "price_velocity_60s",
    "buy_ratio",
    "tfi_300s",
    "hour_cos",
    "hour_sin",
    "side_aligned_tfi",
    "side_aligned_velocity",
]


def train_model(
    X: "pd.DataFrame",
    y: "pd.Series",
    k: int,
) -> Pipeline:
    """LR(C=0.01) + SelectKBest pipeline を学習."""
    k_actual = min(k, X.shape[1])
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_classif, k=k_actual)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.01, max_iter=2000, class_weight="balanced", random_state=42,
        )),
    ])
    pipe.fit(X, y.values)
    return pipe


def get_selected_features(pipe: Pipeline, columns: "pd.Index") -> list[str]:
    """Pipeline から選択された特徴量名を取得."""
    imputer = pipe.named_steps["imputer"]
    survived_mask = np.isfinite(imputer.statistics_)
    survived_cols = columns[survived_mask]
    selector = pipe.named_steps["selector"]
    return survived_cols[selector.get_support()].tolist()


def save_gate(
    pipe: Pipeline,
    X: "pd.DataFrame",
    y: "pd.Series",
    output_path: Path,
    *,
    as_threshold: float,
    label: str,
    k: int,
) -> SkipGate:
    """SkipGate を保存."""
    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]
    selected_cols = get_selected_features(pipe, X.columns)
    fi = dict(zip(selected_cols, np.abs(model.coef_[0]).tolist()))
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    gate = SkipGate(
        model=model,
        scaler=scaler,
        feature_cols=X.columns.tolist(),
        config=SkipGateConfig(
            mode="as",
            as_threshold=as_threshold,
            threshold_bps=0.0,
        ),
        metadata={
            "n_samples": len(X),
            "as_rate": float(y.mean()),
            "k": k,
            "selected_features": selected_cols,
            "feature_importances": dict(sorted_fi),
            "trained_at": datetime.now().isoformat(),
            "label": label,
        },
        pipeline=pipe,
    )
    p = gate.save(output_path)
    logger.info(f"Saved {label}: {p}")
    logger.info(f"  {len(X)} samples, k={k}, selected={selected_cols}")
    logger.info(f"  Top FI: {sorted_fi[:5]}")
    return gate


def main() -> None:
    import pandas as pd

    logger.info("Loading fill records")
    df = load_fill_records()
    enriched_df = enrich_fill_records(df)

    models_dir = _PROJECT_ROOT / "models/v460"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ===== Primary: A(k=12) — enriched, OB+spread =====
    logger.info("\n=== Primary: A(k=12) — enriched (OB+spread) ===")
    X_primary, y_primary = build_enriched_as_features(enriched_df, require_spread=True)
    pipe_primary = train_model(X_primary, y_primary, k=12)
    gate_primary = save_gate(
        pipe_primary, X_primary, y_primary,
        output_path=models_dir / "skip_gate_as.pkl",
        as_threshold=0.65,
        label="primary_A_k12",
        k=12,
    )

    # ===== Fallback: Curated(k=5) — OB-free =====
    logger.info("\n=== Fallback: Curated(k=5) — OB-free ===")
    X_full, y_full = build_enriched_as_features(enriched_df, require_spread=False)
    curated_cols = [c for c in CURATED_TRADE_FEATURES if c in X_full.columns]
    X_curated = X_full[curated_cols]
    pipe_curated = train_model(X_curated, y_full, k=5)
    gate_curated = save_gate(
        pipe_curated, X_curated, y_full,
        output_path=models_dir / "skip_gate_as_fallback.pkl",
        as_threshold=0.65,
        label="fallback_curated_k5",
        k=5,
    )

    # ===== Verify both models can predict =====
    logger.info("\n=== Verification ===")

    # Primary prediction on first sample
    test_x = X_primary.iloc[:5]
    probs_p = pipe_primary.predict_proba(test_x)[:, 1]
    logger.info(f"Primary predictions (first 5): {probs_p.tolist()}")

    # Fallback prediction on first 5 samples
    test_xc = X_curated.iloc[:5]
    probs_f = pipe_curated.predict_proba(test_xc)[:, 1]
    logger.info(f"Fallback predictions (first 5): {probs_f.tolist()}")

    # Verify loaded models
    gate_loaded = SkipGate.load(models_dir / "skip_gate_as.pkl")
    logger.info(f"Primary loaded: {gate_loaded.metadata.get('label')}")
    logger.info(f"  Feature cols: {len(gate_loaded.feature_cols)} features")
    logger.info(f"  Selected: {gate_loaded.metadata.get('selected_features')}")

    gate_fb_loaded = SkipGate.load(models_dir / "skip_gate_as_fallback.pkl")
    logger.info(f"Fallback loaded: {gate_fb_loaded.metadata.get('label')}")
    logger.info(f"  Feature cols: {len(gate_fb_loaded.feature_cols)} features")
    logger.info(f"  Selected: {gate_fb_loaded.metadata.get('selected_features')}")

    print("\n" + "=" * 60)
    print("Two-Tier model save COMPLETE")
    print(f"  Primary:  {models_dir / 'skip_gate_as.pkl'}")
    print(f"  Fallback: {models_dir / 'skip_gate_as_fallback.pkl'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
