# Observation building utilities for trading environment
# 取引環境の観測値構築ユーティリティ

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ObservationBuilder:
    """Handles observation construction and info generation."""

    def __init__(  # type: ignore[misc]
        self,
        features: List[str],
        feature_matrix: NDArray[np.float32],
        nonfinite_rows: Set[int],
        nonfinite_warned_rows: Set[int],
    ):
        self.features = features
        self._feature_matrix = feature_matrix
        self._nonfinite_rows = nonfinite_rows
        self._nonfinite_warned_rows = nonfinite_warned_rows

    def get_observation(
        self,
        current_step: int,
        n_steps: int,
        df: pd.DataFrame,
    ) -> NDArray[np.float32]:
        """現在の状態を取得"""
        if self._feature_matrix.size:
            max_index = self._feature_matrix.shape[0] - 1
            index = min(max(current_step, 0), max_index)
            obs = self._feature_matrix[index]

            if (
                self._nonfinite_rows
                and index in self._nonfinite_rows
                and index not in self._nonfinite_warned_rows
            ):
                if len(self._nonfinite_warned_rows) < 5 or index % 1000 == 0:
                    print(
                        f"Warning: Step {index} had non-finite feature values. Replaced with zeros for stability."
                    )
                self._nonfinite_warned_rows.add(index)

            return obs

        # Fallback path (should rarely execute) - preserve previous behaviour
        if current_step >= n_steps:
            step_data = df.iloc[-1]
        else:
            step_data = df.iloc[current_step]

        feature_list = list(self.features)
        obs = step_data[feature_list].to_numpy(dtype=np.float32, copy=False)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def get_info(
        self,
        current_step: int,
        n_steps: int,
        position: float,
        total_pnl: float,
        trades_count: int,
        features: List[str],
        config: Any,
    ) -> Dict[str, Any]:
        """追加情報を取得"""
        return {
            "current_step": current_step,
            "total_steps": n_steps,
            "position": position,
            "total_pnl": total_pnl,
            "trades_count": trades_count,
            "features": features,
            "config": config,
            "pnl": total_pnl,
        }

    def update_features(self, features: List[str]) -> None:
        """特徴量リストを更新"""
        self.features = features

    def update_feature_matrix(
        self,
        feature_matrix: NDArray[np.float32],
        nonfinite_rows: Set[int],
    ) -> None:
        """特徴量行列と非有限値情報を更新"""
        self._feature_matrix = feature_matrix
        self._nonfinite_rows = nonfinite_rows
        self._nonfinite_warned_rows.clear()