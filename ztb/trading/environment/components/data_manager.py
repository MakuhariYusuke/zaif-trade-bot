"""
Data Manager - Handles data management and streaming logic.

This module separates data-related logic from the main environment class,
including data initialization, streaming, and buffer management.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataManager:
    """
    Manages data operations for trading environment.

    This class handles:
    - Data initialization and validation
    - Feature matrix construction
    - Fast access buffer management
    - Streaming data handling
    - Data availability checks
    """

    def __init__(self):
        """Initialize DataManager."""
        self.df: pd.DataFrame | None = None
        self.features: list[str] = []
        self.n_steps: int = 0
        self.current_step: int = 0

        # Data buffers
        self._feature_matrix: NDArray[np.float32] | None = None
        self._price_array: NDArray[np.float32] | None = None
        self._close_array: NDArray[np.float32] | None = None
        self._atr_array: NDArray[np.float32] | None = None
        self._episode_id_array: NDArray[Any] | None = None

        # Metadata
        self._timestamp_column: str | None = None
        self._episode_id_column: str | None = None
        self._stream_last_timestamp: pd.Timestamp | None = None
        self._stream_rows_appended: int = 0

        # Data quality tracking
        self._nonfinite_rows: set[int] = set()
        self._nonfinite_warned_rows: set[int] = set()

    def initialize_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        timestamp_column: str | None = None,
        episode_id_column: str | None = None,
    ) -> None:
        """
        Initialize data structures.

        Args:
            df: Input dataframe
            features: list of feature names
            timestamp_column: Name of timestamp column
            episode_id_column: Name of episode ID column

        Raises:
            ValueError: If data or features are invalid
            TypeError: If inputs have wrong types
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"df must be pandas DataFrame, got {type(df)}")

            if not features:
                raise ValueError("Features list cannot be empty")

            if not isinstance(features, list):
                raise TypeError(f"features must be list, got {type(features)}")

            self.df = df.copy()
            self.features = features
            self.n_steps = len(df)
            self._timestamp_column = timestamp_column
            self._episode_id_column = episode_id_column

            # Extract episode IDs if available
            if episode_id_column and episode_id_column in df.columns:
                self._episode_id_array = df[episode_id_column].values

            logger.info(
                f"Initialized data with {self.n_steps} steps and {len(features)} features"
            )

        except Exception as e:
            logger.error(f"Failed to initialize data: {e}")
            raise

    def build_fast_access_buffers(self) -> None:
        """Build fast access buffers for commonly used data."""
        if self.df is None:
            raise ValueError("Data not initialized")

        # Build feature matrix — vectorized (no iterrows)
        available_features = [f for f in self.features if f in self.df.columns]
        missing_features = [f for f in self.features if f not in self.df.columns]

        if available_features:
            feature_df = self.df[available_features].astype(np.float64)
            # Detect non-finite rows
            nonfinite_mask = ~np.isfinite(feature_df.values).all(axis=1)
            nonfinite_rows = set(self.df.index[nonfinite_mask].tolist())
        else:
            feature_df = pd.DataFrame(index=self.df.index)
            nonfinite_rows = set()

        # Build full feature array in column order matching self.features
        n_rows = len(self.df)
        feature_data = np.zeros((n_rows, len(self.features)), dtype=np.float32)
        for col_idx, feature in enumerate(self.features):
            if feature in self.df.columns:
                feature_data[:, col_idx] = self.df[feature].values.astype(np.float32)

        self._feature_matrix = feature_data
        self._nonfinite_rows = nonfinite_rows

        # Build price buffers
        if "close" in self.df.columns:
            self._close_array = self.df["close"].values.astype(np.float32)
        if "price" in self.df.columns:
            self._price_array = self.df["price"].values.astype(np.float32)
        if "atr" in self.df.columns:
            self._atr_array = self.df["atr"].values.astype(np.float32)

        logger.info("Built fast access buffers")

    def ensure_data_available(self, step: int) -> None:
        """
        Ensure data is available for the given step.

        Args:
            step: Step to check
        """
        if step >= self.n_steps:
            raise IndexError(
                f"Step {step} exceeds available data (n_steps={self.n_steps})"
            )

    def get_feature_matrix(self) -> NDArray[np.float32] | None:
        """Get the feature matrix."""
        return self._feature_matrix

    def get_price_at_step(self, step: int) -> float:
        """Get price at specific step."""
        if self._price_array is not None and step < len(self._price_array):
            return float(self._price_array[step])
        elif self._close_array is not None and step < len(self._close_array):
            return float(self._close_array[step])
        else:
            raise ValueError(f"No price data available at step {step}")

    def get_atr_at_step(self, step: int) -> float:
        """Get ATR at specific step."""
        if self._atr_array is not None and step < len(self._atr_array):
            return float(self._atr_array[step])
        else:
            return 0.0  # Default ATR value

    def get_episode_id_at_step(self, step: int) -> Any:
        """Get episode ID at specific step."""
        if self._episode_id_array is not None and step < len(self._episode_id_array):
            return self._episode_id_array[step]
        return None

    def update_current_step(self, step: int) -> None:
        """Update current step."""
        self.current_step = step

    def is_episode_boundary(self, current_step: int, next_step: int) -> bool:
        """Check if there's an episode boundary between steps."""
        if self._episode_id_array is None:
            return False

        if current_step < 0 or next_step >= len(self._episode_id_array):
            return False

        current_episode = self._episode_id_array[current_step]
        next_episode = self._episode_id_array[next_step]
        return current_episode != next_episode
