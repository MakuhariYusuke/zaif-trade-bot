"""617# §3.2: 推論時の特徴量標準化ローダー.

retrain 時に出力された norm.json を読み込み、
ライブ推論時の Z-score 変換 + NaN imputation + clipping を行う。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NormLoader:
    """617# §3.2: 推論時の特徴量標準化.

    norm.json を読み込み、特徴量ベクトルの正規化を行う。
    mtime ベースの hot-reload で retrain 後の自動取り込みに対応。
    """

    _EPS: float = 1e-10

    def __init__(self, norm_path: Path) -> None:
        self._path = norm_path
        self._mtime: float = 0.0
        self._feature_names: list[str] = []
        self._means: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._stds: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._mins: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._maxs: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._loaded = False
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def reload_if_changed(self) -> bool:
        """mtime が変わっていたら再読み込み。変更があれば True を返す。"""
        if not self._path.exists():
            return False
        current_mtime = self._path.stat().st_mtime
        if current_mtime != self._mtime:
            self._load()
            return True
        return False

    def normalize(self, raw_features: dict[str, float]) -> NDArray[np.float64]:
        """617# §3.2: NaN→mean 置換、Z-score 変換、min/max clipping.

        Args:
            raw_features: 特徴量名→生値の辞書

        Returns:
            正規化済みベクトル (feature_names 順)
        """
        n = len(self._feature_names)
        if n == 0:
            return np.array([], dtype=np.float64)

        values = np.empty(n, dtype=np.float64)
        for i, name in enumerate(self._feature_names):
            val = raw_features.get(name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                # NaN → mean imputation
                values[i] = self._means[i]
            else:
                values[i] = float(val)

        # Z-score
        z = (values - self._means) / (self._stds + self._EPS)

        # Clipping
        z = np.clip(z, self._mins, self._maxs)

        return z

    def _load(self) -> None:
        """norm.json を読み込んで内部状態を更新。"""
        if not self._path.exists():
            logger.debug(f"NormLoader: {self._path} not found, skipping")
            return

        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"NormLoader: failed to load {self._path}: {e}")
            return

        stats = data.get("feature_stats")
        if not isinstance(stats, dict) or not stats:
            logger.warning(f"NormLoader: empty feature_stats in {self._path}")
            return

        names: list[str] = []
        means: list[float] = []
        stds: list[float] = []
        mins: list[float] = []
        maxs: list[float] = []

        for name, s in stats.items():
            if not isinstance(s, dict):
                continue
            names.append(name)
            means.append(float(s.get("mean", 0.0)))
            stds.append(float(s.get("std", 1.0)))
            mins.append(float(s.get("min", -5.0)))
            maxs.append(float(s.get("max", 5.0)))

        self._feature_names = names
        self._means = np.array(means, dtype=np.float64)
        self._stds = np.array(stds, dtype=np.float64)
        self._mins = np.array(mins, dtype=np.float64)
        self._maxs = np.array(maxs, dtype=np.float64)
        self._mtime = self._path.stat().st_mtime
        self._loaded = True
        logger.info(
            f"NormLoader: loaded {len(names)} features from {self._path}"
        )
