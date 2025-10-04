"""
Base classes and protocols for trading features.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Set, TypeVar

import pandas as pd

T = TypeVar("T", covariant=True)


class Feature(Protocol[T]):
    """Generic feature protocol"""

    @property
    def name(self) -> str:
        """Feature name"""
        ...

    @property
    def deps(self) -> List[str]:
        """Dependencies (common preprocessing etc)"""
        ...

    def compute(self, df: pd.DataFrame, **params: Any) -> T:
        """Compute feature with generic return type"""
        ...


class ComputableFeature(Protocol):
    """Enhanced feature protocol with strict typing"""

    @property
    def name(self) -> str:
        """Feature name"""
        ...

    @property
    def deps(self) -> List[str]:
        """Dependencies"""
        ...

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature with strict DataFrame return type"""
        ...


class StrictComputableFeature(Protocol):
    """Strict computable feature protocol - no **params allowed"""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame: ...


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

    @property
    def required_calculations(self) -> Set[str]:
        """Required preprocessing calculations for this feature"""
        val = getattr(self, "_required_calculations", None)
        return val if val is not None else set()

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature"""
        pass


class ParameterizedFeature(BaseFeature):
    """Base class for features that accept dynamic parameters"""

    def __init__(
        self,
        name: str,
        deps: Optional[List[str]] = None,
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self._default_params = default_params or {}
        self.default_params = default_params or {}

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        # Merge default params with provided params
        merged_params = {**self._default_params, **params}
        return self._compute_with_params(df, **merged_params)

    @abstractmethod
    def _compute_with_params(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Actual computation with parameters"""
        pass


class MovingAverageFeature(BaseFeature):
    """Base class for moving average based features"""

    def __init__(self, name: str, deps: Optional[List[str]] = None):
        super().__init__(name, deps or ["close"])

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving average based feature"""
        pass


class OscillatorFeature(BaseFeature):
    """Base class for oscillator based features"""

    def __init__(self, name: str, deps: Optional[List[str]] = None):
        super().__init__(name, deps or ["close", "high", "low"])

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute oscillator based feature"""
        pass


class ChannelFeature(BaseFeature):
    """Base class for channel/range based features"""

    def __init__(self, name: str, deps: Optional[List[str]] = None):
        super().__init__(name, deps or ["high", "low", "close"])

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute channel based feature"""
        pass


class CommonPreprocessor:
    """Common preprocessor class"""

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess basic columns"""
        needs_copy = False

        # Rename price to close if needed
        # Only rename if 'price' exists and 'close' does not, to avoid overwriting an existing 'close' column.
        if "price" in df.columns and "close" not in df.columns:
            df = df.rename(columns={"price": "close"})
            needs_copy = True
        # List of required columns for feature computation.
        # This list can be extended in the future if additional columns become necessary for new features.
        # Keeping this as a separate variable improves maintainability and clarity.
        required_cols = ["close", "high", "low", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        # Calculate return
        if "return" not in df.columns:
            if not needs_copy:
                df = df.copy()
                needs_copy = True
            # Ensure 'close' is float and fill missing values
            df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(float)
            df["close"] = df["close"].ffill().bfill()
            df["return"] = df["close"].pct_change().fillna(0)

        # abs_return
        if "abs_return" not in df.columns:
            if not needs_copy:
                df = df.copy()
                needs_copy = True
            df["abs_return"] = df["return"].abs()

        # rolling mean/std (common)
        windows = [14, 50]
        for w in windows:
            if f"rolling_mean_{w}" not in df.columns:
                if not needs_copy:
                    df = df.copy()
                    needs_copy = True
                df[f"rolling_mean_{w}"] = df["close"].rolling(w).mean().ffill()
            if f"rolling_std_{w}" not in df.columns:
                if not needs_copy:
                    df = df.copy()
                    needs_copy = True
                df[f"rolling_std_{w}"] = df["close"].rolling(w).std().ffill()

        # rolling max/min
        for w in windows:
            if f"rolling_max_{w}" not in df.columns:
                if not needs_copy:
                    df = df.copy()
                    needs_copy = True
                df[f"rolling_max_{w}"] = df["close"].rolling(w).max().ffill()
            if f"rolling_min_{w}" not in df.columns:
                if not needs_copy:
                    df = df.copy()
                    needs_copy = True
                df[f"rolling_min_{w}"] = df["close"].rolling(w).min().ffill()

        # EMA (common spans)
        ema_spans = [5, 10, 12, 14, 20, 26]
        for span in ema_spans:
            if f"ema_{span}" not in df.columns:
                if not needs_copy:
                    df = df.copy()
                    needs_copy = True
                df[f"ema_{span}"] = df["close"].ewm(span=span).mean()

        return df
