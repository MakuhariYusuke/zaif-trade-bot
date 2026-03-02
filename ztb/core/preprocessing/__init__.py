"""Compatibility wrappers for preprocessing modules.

This module exposes a small API subset expected by legacy tests under
`ztb.core.preprocessing.*`. Implementations delegate to the current
locations under `ztb.data` and `ztb.preprocessing`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ztb.data.anomaly_detection import ComprehensiveAnomalyDetector
from ztb.data.outlier_detection import OutlierDetector

__all__ = [
    "AnomalyDetector",
    "NoiseFilter",
    "SyntheticDataGenerator",
    "preprocess_data",
]

class NoiseFilter:
    """Simple NoiseFilter shim used in older tests.

    This is intentionally basic: it exposes the same method names expected by
    the tests and delegates to the outlier detection utilities.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.zscore_threshold = float(config.get("zscore_threshold", 3.0))
        self.iqr_multiplier = float(config.get("iqr_multiplier", 1.5))

    def filter_zscore(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        detector = OutlierDetector()
        methods = [{"type": "z_score", "threshold": self.zscore_threshold}]
        flags = detector.detect_outliers(df, methods, columns=cols)
        # flags is a DataFrame-like with _is_outlier columns added per column
        for col in cols:
            is_outlier_col = f"{col}_is_outlier"
            if is_outlier_col in flags.columns:
                df.loc[flags[is_outlier_col], col] = df[col].median()
        return df

    def filter_iqr(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        detector = OutlierDetector()
        methods = [{"type": "iqr", "multiplier": self.iqr_multiplier}]
        flags = detector.detect_outliers(df, methods, columns=cols)
        for col in cols:
            is_outlier_col = f"{col}_is_outlier"
            if is_outlier_col in flags.columns:
                df.loc[flags[is_outlier_col], col] = df[col].median()
        return df

    def apply_filters(self, df: pd.DataFrame, cols: list[str] | None = None):
        if cols is None:
            # Try to apply to typical numeric columns
            cols = [
                c for c, dt in df.dtypes.items() if pd.api.types.is_numeric_dtype(dt)
            ]
        df = self.filter_zscore(df, cols)
        df = self.filter_iqr(df, cols)
        return df

class SyntheticDataGenerator:
    """Minimal synthetic data generator for tests.

    Does not require heavy NLP dependencies; it produces a simple DataFrame
    with numeric and text columns for preprocessing tests.
    """

    def __init__(self, seed: int | None = None, config: dict | None = None):
        self.config = config or {}
        # back compatible API supports either explicit seed or config['random_state']
        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.config.get("random_state")
        if self.seed is not None:
            np.random.seed(self.seed)
        # Back-compatible attribute expected in tests
        self.random_state = self.seed

    def generate_market_news(self, n: int = 10) -> list[str]:
        samples = [
            "価格が上昇しました。買いを検討してください。",
            "価格が下落しました。売りシグナルを確認してください。",
            "市場は横ばいです。様子見を推奨します。",
        ]
        return [samples[i % len(samples)] for i in range(n)]

    def generate_df(self, n: int = 100) -> pd.DataFrame:
        timestamps = pd.date_range("2023-01-01", periods=n, freq="1min")
        prices = np.cumsum(np.random.randn(n)) + 100.0
        volumes = np.random.randint(100, 1000, size=n)
        df = pd.DataFrame({"timestamp": timestamps, "price": prices, "volume": volumes})
        return df

    def generate_gaussian_noise(
        self, df: pd.DataFrame, cols: list[str], noise_level: float = 0.1
    ) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                sigma = noise_level * df[col].std()
                noise = np.random.normal(0, sigma, size=len(df))
                df[col] = df[col] + noise
        return df

    def generate_time_series(
        self, df: pd.DataFrame, n_periods: int = 10
    ) -> pd.DataFrame:
        # Simple downsample or repeat to construct a time series of desired length
        if len(df) >= n_periods:
            resampled = df.iloc[:n_periods].copy()
            resampled.index = range(n_periods)
            return resampled
        else:
            # Repeat rows to reach n_periods
            reps = n_periods // len(df) + 1
            new_df = pd.concat([df] * reps, ignore_index=True)
            return new_df.iloc[:n_periods].reset_index(drop=True)

    def generate_smote_like(
        self, df: pd.DataFrame, column: str, n_samples: int = 10
    ) -> pd.DataFrame:
        try:
            import numpy as _np
            from sklearn.neighbors import NearestNeighbors

            numeric = df.select_dtypes(include=["number"])  # relevant numeric features
            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(numeric.values)
            # get k neighbors for each sample
            indices = nn.kneighbors(numeric.values, return_distance=False)
            synthetic_rows = []
            for i in range(n_samples):
                idx = i % len(indices)
                neigh = indices[idx]
                # choose two neighbors and interpolate
                a, b = neigh[0], neigh[1] if len(neigh) > 1 else neigh[0]
                alpha = _np.random.rand()
                new_row = (
                    numeric.iloc[a] * alpha + numeric.iloc[b] * (1 - alpha)
                ).to_dict()
                synthetic_rows.append(new_row)
            synthetic_df = pd.DataFrame(synthetic_rows)
            # Keep original non-numeric columns as copies
            for col in df.columns:
                if col not in synthetic_df.columns:
                    synthetic_df[col] = df[col].iloc[: len(synthetic_df)].values
            return pd.concat(
                [df.reset_index(drop=True), synthetic_df.reset_index(drop=True)],
                ignore_index=True,
            )
        except Exception:
            # fallback: append random existing rows
            idxs = np.random.randint(0, len(df), size=n_samples)
            return pd.concat(
                [df.reset_index(drop=True), df.iloc[idxs].reset_index(drop=True)],
                ignore_index=True,
            )

class AnomalyDetector:
    """Compatibility wrapper exposing simple api expected by legacy tests.

    - Accepts a `config` dict during initialization
    - Exposes a `methods` list showing available detection methods
    - Provides `detect_anomalies(df, method=...)` returning (DataFrame, Series)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.methods = ["statistical", "isolation_forest", "local_outlier_factor"]
        # Underlying modern detector for more advanced usage
        self._comprehensive = ComprehensiveAnomalyDetector()
        self._outlier_detector = OutlierDetector()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        method: str = "statistical",
        columns: list[str] | None = None,
    ):
        if method == "statistical":
            methods = [{"type": "z_score", "threshold": 3.0}]
            flags = self._outlier_detector.detect_outliers(df, methods, columns=columns)
            # Build mask across specified numeric columns
            outlier_cols = [c for c in flags.columns if c.endswith("_is_outlier")]
            if outlier_cols:
                mask = flags[outlier_cols].any(axis=1)
                return df, mask
            return df, pd.Series(False, index=df.index)

        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest

                X = df.select_dtypes(include=["number"]).values
                clf = IsolationForest(
                    contamination=float(self.config.get("contamination", 0.1))
                )
                preds = clf.fit_predict(X)
                mask = pd.Series(preds == -1, index=df.index)
                return df, mask
            except Exception:
                return self.detect_anomalies(df, method="statistical")

        elif method == "local_outlier_factor":
            try:
                from sklearn.neighbors import LocalOutlierFactor

                X = df.select_dtypes(include=["number"]).values
                clf = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=float(self.config.get("contamination", 0.1)),
                )
                preds = clf.fit_predict(X)
                mask = pd.Series(preds == -1, index=df.index)
                return df, mask
            except Exception:
                return self.detect_anomalies(df, method="statistical")

        else:
            return self.detect_anomalies(df, method="statistical")

def preprocess_data(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Helper wrapper used by older tests.

    It performs basic noise filtering and returns the resulting DataFrame.
    """
    filter_obj = NoiseFilter(config=config)
    return filter_obj.apply_filters(df)
