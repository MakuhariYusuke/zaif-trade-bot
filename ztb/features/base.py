"""
Base classes and protocols for trading features.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, TypeVar
import pandas as pd
from typing import Set

T = TypeVar('T', covariant=True)


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
        return getattr(self, '_required_calculations', set())

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature"""
        pass


class ParameterizedFeature(BaseFeature):
    """Base class for features that accept dynamic parameters"""

    def __init__(self, name: str, deps: Optional[List[str]] = None,
                 default_params: Optional[Dict[str, Any]] = None):
        super().__init__(name, deps)
        self.default_params = default_params or {}

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Compute with parameter override support"""
        # Merge default params with provided params
        merged_params = {**self.default_params, **params}
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