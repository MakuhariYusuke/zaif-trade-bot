# Observation building utilities for trading environment
# 取引環境の観測値構築ユーティリティ

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.types.common import ConfigDict
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ObservationBuilder:
    """Handles observation construction and info generation."""

    def __init__(
        self,
        features: List[str],
        feature_matrix: NDArray[np.float32],
        nonfinite_rows: Set[int],
        nonfinite_warned_rows: Set[int],
        scaler_mean: Optional[NDArray[np.float32]] = None,
        scaler_std: Optional[NDArray[np.float32]] = None,
        optimizer_tracker: Optional["OptimizerFeatureTracker"] = None,
    ):
        super().__init__()
        self.features = features
        self._feature_matrix = feature_matrix
        self._nonfinite_rows = nonfinite_rows
        self._nonfinite_warned_rows = nonfinite_warned_rows
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.optimizer_tracker = optimizer_tracker

    def get_observation(
        self,
        current_step: int,
        n_steps: int,
        df: pd.DataFrame,
    ) -> NDArray[np.float32]:
        """現在の状態を取得（正規化オプション付き）"""
        if self._feature_matrix.size:
            max_index = self._feature_matrix.shape[0] - 1
            index = min(max(current_step, 0), max_index)
            obs: NDArray[np.float32] = self._feature_matrix[index]

            if (
                self._nonfinite_rows
                and index in self._nonfinite_rows
                and index not in self._nonfinite_warned_rows
            ):
                if len(self._nonfinite_warned_rows) < 5 or index % 1000 == 0:
                    logger.warning(
                        "Step %d had non-finite feature values. Replaced with zeros for stability.",
                        index,
                    )
                self._nonfinite_warned_rows.add(index)

            # スケーラーが設定されていれば正規化を適用
            if self.scaler_mean is not None and self.scaler_std is not None:
                # 標準偏差が0の特徴量はゼロ除算を避けるため正規化しない
                safe_std = np.where(self.scaler_std > 1e-8, self.scaler_std, 1.0)
                obs = ((obs - self.scaler_mean) / safe_std).astype(np.float32)

            # Add optimizer features if tracker is available
            if self.optimizer_tracker is not None:
                optimizer_features = self.optimizer_tracker.get_feature_vector()
                optimizer_values = np.array(
                    list(optimizer_features.values()), dtype=np.float32
                )
                obs = np.concatenate([obs, optimizer_values])

            # Diagnostic: log observation shape and optimizer tracker state (every 100 steps)
            try:
                if hasattr(self, '_step_count'):
                    self._step_count += 1
                else:
                    self._step_count = 1
                if self._step_count % 100 == 0:
                    logger.debug(
                        "ObservationBuilder.get_observation: obs.shape=%s, optimizer_tracker_present=%s",
                        obs.shape,
                        self.optimizer_tracker is not None,
                    )
            except Exception:
                pass

            return obs

        # Fallback path (should rarely execute) - preserve previous behaviour
        if current_step >= n_steps:
            step_data = df.iloc[-1]
        else:
            step_data = df.iloc[current_step]

        feature_list = list(self.features)
        obs = step_data[feature_list].to_numpy(dtype=np.float32, copy=False)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Fallbackパスでも正規化を適用
        if self.scaler_mean is not None and self.scaler_std is not None:
            safe_std = np.where(self.scaler_std > 1e-8, self.scaler_std, 1.0)
            obs = ((obs - self.scaler_mean) / safe_std).astype(np.float32)

        # Add optimizer features if tracker is available
        if self.optimizer_tracker is not None:
            optimizer_features = self.optimizer_tracker.get_feature_vector()
            optimizer_values = np.array(
                list(optimizer_features.values()), dtype=np.float32
            )
            obs = np.concatenate([obs, optimizer_values])

        # Diagnostic: log observation shape and optimizer tracker state for fallback path
        try:
            logger.debug(
                "ObservationBuilder.get_observation (fallback): obs.shape=%s, optimizer_tracker_present=%s",
                obs.shape,
                self.optimizer_tracker is not None,
            )
        except Exception:
            pass

        return obs

    def get_info(
        self,
        current_step: int,
        n_steps: int,
        position: float,
        total_pnl: float,
        trades_count: int,
        features: List[str],
        config: ConfigDict,
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

    def update_scaler(
        self,
        scaler_mean: Optional[NDArray[np.float32]],
        scaler_std: Optional[NDArray[np.float32]],
    ) -> None:
        """スケーラー情報を更新"""
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
