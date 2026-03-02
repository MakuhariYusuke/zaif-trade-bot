"""Compatibility module providing legacy preprocessing classes and helpers.

This is a lightweight wrapper used by older tests that import
`ztb.core.preprocessing.data_preprocessing`. It delegates functionality to
`ztb.data.anomaly_detection` and the local `NoiseFilter` shim.
"""

from __future__ import annotations

import pandas as pd

from ztb.core.preprocessing import AnomalyDetector, NoiseFilter, SyntheticDataGenerator

__all__ = [
    "AnomalyDetector",
    "NoiseFilter",
    "SyntheticDataGenerator",
    "preprocess_data",
]

# Export lightweight aliases for tests
AnomalyDetector = AnomalyDetector
NoiseFilter = NoiseFilter
SyntheticDataGenerator = SyntheticDataGenerator

def preprocess_data(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Minimal preprocess pipeline used in tests.

    Options (from tests):
      - apply_noise_filter: bool
      - apply_anomaly_detection: bool
      - generate_synthetic: bool
      - synthetic_periods: int
    """
    cfg = config or {}
    result_df = df.copy()

    if cfg.get("apply_noise_filter"):
        nf = NoiseFilter(config=cfg)
        result_df = nf.apply_filters(result_df)

    if cfg.get("apply_anomaly_detection"):
        detector = AnomalyDetector(config=cfg)
        # Detect anomalies and drop or mark them
        _, mask = detector.detect_anomalies(
            result_df, method=cfg.get("anomaly_method", "statistical")
        )
        # For simplicity, drop anomaly rows
        if mask.any():
            result_df = result_df.loc[~mask].reset_index(drop=True)

    if cfg.get("generate_synthetic"):
        gen = SyntheticDataGenerator(config=cfg)
        sp = int(cfg.get("synthetic_periods", 0))
        if sp > 0:
            synth_df = gen.generate_time_series(result_df, n_periods=sp)
            result_df = synth_df

    return result_df
