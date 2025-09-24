"""
Base classes and protocols for trading features.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional
import pandas as pd


class Feature(Protocol):
    """Feature protocol"""

    @property
    def name(self) -> str:
        """Feature name"""
        ...

    @property
    def deps(self) -> List[str]:
        """Dependencies (common preprocessing etc)"""
        ...

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Compute feature"""
        ...


class BaseFeature(ABC):
    """Base feature class"""

    def __init__(self, name: str, deps: Optional[List[str]] = None):
        self._name = name
        self._deps = deps or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def deps(self) -> List[str]:
        return self._deps

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Compute feature"""
        pass


class CommonPreprocessor:
    """Common preprocessor class"""

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess basic columns"""
        df = df.copy()

        # Rename price to close if needed
        if 'price' in df.columns and 'close' not in df.columns:
            df = df.rename(columns={'price': 'close'})

        # Check required columns
        required_cols = ['close', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Calculate return
        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change().fillna(0)

        # abs_return
        if 'abs_return' not in df.columns:
            df['abs_return'] = df['return'].abs()

        # rolling mean/std (common)
        windows = [14, 50]
        for w in windows:
            if f'rolling_mean_{w}' not in df.columns:
                df[f'rolling_mean_{w}'] = df['close'].rolling(w).mean().ffill()
            if f'rolling_std_{w}' not in df.columns:
                df[f'rolling_std_{w}'] = df['close'].rolling(w).std().ffill()

        # rolling max/min
        for w in windows:
            if f'rolling_max_{w}' not in df.columns:
                df[f'rolling_max_{w}'] = df['close'].rolling(w).max().ffill()
            if f'rolling_min_{w}' not in df.columns:
                df[f'rolling_min_{w}'] = df['close'].rolling(w).min().ffill()

        # EMA (common spans)
        ema_spans = [5, 10, 12, 14, 20, 26]
        for span in ema_spans:
            if f'ema_{span}' not in df.columns:
                df[f'ema_{span}'] = df['close'].ewm(span=span).mean()

        return df