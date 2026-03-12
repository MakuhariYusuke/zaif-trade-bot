"""366# M3: σ-Clustering (Volatility Regime Classification).

ボラティリティの離散クラスタを閾値分類で検出し、
regime 依存の offset 乗数を提供する。

理論
----
σ (Welford std / Parkinson σ / vol_ratio) を入力として、
ボラティリティ環境を 4 段階に分類:

| Cluster   | 代表的 vol_ratio 範囲 | offset 戦略         |
|-----------|----------------------|---------------------|
| LOW       | < low_threshold      | tight → fill 優先   |
| MID       | low ~ high           | balanced (default)  |
| HIGH      | high ~ extreme       | wide → AS 防御      |
| EXTREME   | > extreme_threshold  | halt / 超ワイド     |

既存基盤
--------
- ``WelfordOnlineVar`` (T5): O(1) σ 追跡
- ``RegimeDetector.last_volatility_ratio``: baseline 比の正規化σ
- ``_estimate_sigma()`` (Parkinson): High-Low σ 推定

References
----------
Cont, R. (2001). "Empirical properties of asset returns:
stylized facts and statistical issues."
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

__all__ = [
    "VolatilityCluster",
    "VolatilityClusterConfig",
    "VolatilityRegimeClassifier",
]

logger = logging.getLogger(__name__)


class VolatilityCluster(enum.Enum):
    """ボラティリティ レジーム."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"
    EXTREME = "extreme"

    @property
    def is_defensive(self) -> bool:
        """HIGH/EXTREME = 防御的."""
        return self in (VolatilityCluster.HIGH, VolatilityCluster.EXTREME)


@dataclass(frozen=True)
class VolatilityClusterConfig:
    """σ-clustering 設定.

    閾値は vol_ratio (baseline 比の σ) で指定。
    offset_mult は各レジームでの offset 乗数 (1.0 = 変更なし)。
    """

    # 閾値 (vol_ratio ベース)
    low_threshold: float = 0.6
    high_threshold: float = 1.5
    extreme_threshold: float = 3.0

    # 各レジームの offset 乗数
    low_offset_mult: float = 0.8       # tight → fill 率↑
    mid_offset_mult: float = 1.0       # baseline
    high_offset_mult: float = 1.3      # wide → AS 防御
    extreme_offset_mult: float = 2.0   # 超ワイド / halt 検討

    # ヒステリシス幅 (チャタリング防止)
    hysteresis: float = 0.05

    def __post_init__(self) -> None:
        """H3 fix: 閾値の順序検証."""
        if not (self.low_threshold < self.high_threshold < self.extreme_threshold):
            raise ValueError(
                f"閾値は low < high < extreme である必要があります: "
                f"low={self.low_threshold}, high={self.high_threshold}, "
                f"extreme={self.extreme_threshold}"
            )

    def offset_mult_for(self, cluster: VolatilityCluster) -> float:
        """クラスタ → offset 乗数."""
        return {
            VolatilityCluster.LOW: self.low_offset_mult,
            VolatilityCluster.MID: self.mid_offset_mult,
            VolatilityCluster.HIGH: self.high_offset_mult,
            VolatilityCluster.EXTREME: self.extreme_offset_mult,
        }[cluster]


class VolatilityRegimeClassifier:
    """σ-clustering に基づくボラティリティ レジーム分類器.

    ``classify()`` は vol_ratio (正規化σ) を入力とし、
    ヒステリシス付きで VolatilityCluster を返す。

    Usage
    -----
    >>> cfg = VolatilityClusterConfig()
    >>> classifier = VolatilityRegimeClassifier(cfg)
    >>> cluster = classifier.classify(1.2)
    >>> mult = cfg.offset_mult_for(cluster)
    """

    __slots__ = ("_config", "_current")

    def __init__(
        self,
        config: VolatilityClusterConfig | None = None,
    ) -> None:
        self._config = config or VolatilityClusterConfig()
        self._current = VolatilityCluster.MID

    @property
    def current_cluster(self) -> VolatilityCluster:
        """現在のクラスタ."""
        return self._current

    @property
    def current_offset_mult(self) -> float:
        """現在のクラスタに対応する offset 乗数."""
        return self._config.offset_mult_for(self._current)

    def classify(self, vol_ratio: float) -> VolatilityCluster:
        """vol_ratio からクラスタを判定 (ヒステリシス付き).

        Parameters
        ----------
        vol_ratio:
            baseline 比のボラティリティ (RegimeDetector.last_volatility_ratio)。
            1.0 = baseline と同等。

        Returns
        -------
        VolatilityCluster
        """
        # M2 fix: 負の vol_ratio をガード
        vol_ratio = max(0.0, vol_ratio)
        cfg = self._config
        h = cfg.hysteresis
        prev = self._current

        # 上昇方向のヒステリシス: 閾値 + h で遷移
        # 下降方向のヒステリシス: 閾値 - h で遷移
        if prev == VolatilityCluster.LOW:
            if vol_ratio >= cfg.extreme_threshold + h:
                new = VolatilityCluster.EXTREME
            elif vol_ratio >= cfg.high_threshold + h:
                new = VolatilityCluster.HIGH
            elif vol_ratio >= cfg.low_threshold + h:
                new = VolatilityCluster.MID
            else:
                new = VolatilityCluster.LOW

        elif prev == VolatilityCluster.MID:
            if vol_ratio >= cfg.extreme_threshold + h:
                new = VolatilityCluster.EXTREME
            elif vol_ratio >= cfg.high_threshold + h:
                new = VolatilityCluster.HIGH
            elif vol_ratio < cfg.low_threshold - h:
                new = VolatilityCluster.LOW
            else:
                new = VolatilityCluster.MID

        elif prev == VolatilityCluster.HIGH:
            if vol_ratio >= cfg.extreme_threshold + h:
                new = VolatilityCluster.EXTREME
            elif vol_ratio < cfg.low_threshold - h:
                new = VolatilityCluster.LOW
            elif vol_ratio < cfg.high_threshold - h:
                new = VolatilityCluster.MID
            else:
                new = VolatilityCluster.HIGH

        else:  # EXTREME
            if vol_ratio < cfg.low_threshold - h:
                new = VolatilityCluster.LOW
            elif vol_ratio < cfg.high_threshold - h:
                new = VolatilityCluster.MID
            elif vol_ratio < cfg.extreme_threshold - h:
                new = VolatilityCluster.HIGH
            else:
                new = VolatilityCluster.EXTREME

        if new != prev:
            logger.debug(
                f"[σ-cluster] {prev.value} → {new.value} "
                f"(vol_ratio={vol_ratio:.3f})"
            )
        self._current = new
        return new

    def reset(self) -> None:
        """状態をリセット."""
        self._current = VolatilityCluster.MID
