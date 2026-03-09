"""366# M3: σ-Clustering テスト.

VolatilityRegimeClassifier の単体テスト。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.sigma_clustering import (
    VolatilityCluster,
    VolatilityClusterConfig,
    VolatilityRegimeClassifier,
)


# =====================================================================
# TestVolatilityCluster
# =====================================================================


class TestVolatilityCluster:
    """VolatilityCluster enum のテスト."""

    def test_is_defensive(self) -> None:
        """HIGH/EXTREME は防御的."""
        assert not VolatilityCluster.LOW.is_defensive
        assert not VolatilityCluster.MID.is_defensive
        assert VolatilityCluster.HIGH.is_defensive
        assert VolatilityCluster.EXTREME.is_defensive


# =====================================================================
# TestVolatilityClusterConfig
# =====================================================================


class TestVolatilityClusterConfig:
    """VolatilityClusterConfig のテスト."""

    def test_default_thresholds(self) -> None:
        """デフォルト閾値."""
        cfg = VolatilityClusterConfig()
        assert cfg.low_threshold == 0.6
        assert cfg.high_threshold == 1.5
        assert cfg.extreme_threshold == 3.0

    def test_offset_mult_for_each_cluster(self) -> None:
        """各クラスタの offset 乗数."""
        cfg = VolatilityClusterConfig()
        assert cfg.offset_mult_for(VolatilityCluster.LOW) == 0.8
        assert cfg.offset_mult_for(VolatilityCluster.MID) == 1.0
        assert cfg.offset_mult_for(VolatilityCluster.HIGH) == 1.3
        assert cfg.offset_mult_for(VolatilityCluster.EXTREME) == 2.0


# =====================================================================
# TestVolatilityRegimeClassifier
# =====================================================================


class TestVolatilityRegimeClassifier:
    """VolatilityRegimeClassifier のテスト."""

    def test_initial_state_is_mid(self) -> None:
        """初期状態は MID."""
        clf = VolatilityRegimeClassifier()
        assert clf.current_cluster == VolatilityCluster.MID
        assert clf.current_offset_mult == 1.0

    def test_classify_low(self) -> None:
        """低 vol_ratio → LOW."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)
        result = clf.classify(0.3)
        assert result == VolatilityCluster.LOW

    def test_classify_mid(self) -> None:
        """中 vol_ratio → MID."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)
        result = clf.classify(1.0)
        assert result == VolatilityCluster.MID

    def test_classify_high(self) -> None:
        """高 vol_ratio → HIGH."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)
        result = clf.classify(2.0)
        assert result == VolatilityCluster.HIGH

    def test_classify_extreme(self) -> None:
        """極端 vol_ratio → EXTREME."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)
        result = clf.classify(4.0)
        assert result == VolatilityCluster.EXTREME

    def test_hysteresis_prevents_chattering(self) -> None:
        """ヒステリシスでチャタリング防止."""
        cfg = VolatilityClusterConfig(
            low_threshold=0.6,
            high_threshold=1.5,
            hysteresis=0.1,
        )
        clf = VolatilityRegimeClassifier(cfg)

        # MID → vol_ratio が low_threshold ぎりぎり上で MID 維持
        clf.classify(0.55)  # 0.55 > 0.6 - 0.1 = 0.5 → MID 維持
        assert clf.current_cluster == VolatilityCluster.MID

        # さらに下がると LOW に遷移
        clf.classify(0.45)  # 0.45 < 0.6 - 0.1 = 0.5 → LOW
        assert clf.current_cluster == VolatilityCluster.LOW

        # LOW から戻るにも hysteresis
        clf.classify(0.65)  # 0.65 < 0.6 + 0.1 = 0.7 → LOW 維持
        assert clf.current_cluster == VolatilityCluster.LOW

        # 十分上がると MID に遷移
        clf.classify(0.75)  # 0.75 >= 0.6 + 0.1 = 0.7 → MID
        assert clf.current_cluster == VolatilityCluster.MID

    def test_ascending_regime_transitions(self) -> None:
        """上昇系列: LOW → MID → HIGH → EXTREME."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)

        clf.classify(0.3)
        assert clf.current_cluster == VolatilityCluster.LOW

        clf.classify(1.0)
        assert clf.current_cluster == VolatilityCluster.MID

        clf.classify(2.0)
        assert clf.current_cluster == VolatilityCluster.HIGH

        clf.classify(4.0)
        assert clf.current_cluster == VolatilityCluster.EXTREME

    def test_descending_regime_transitions(self) -> None:
        """下降系列: EXTREME → HIGH → MID → LOW."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)

        # まず EXTREME に
        clf.classify(4.0)
        assert clf.current_cluster == VolatilityCluster.EXTREME

        clf.classify(2.0)
        assert clf.current_cluster == VolatilityCluster.HIGH

        clf.classify(1.0)
        assert clf.current_cluster == VolatilityCluster.MID

        clf.classify(0.3)
        assert clf.current_cluster == VolatilityCluster.LOW

    def test_reset(self) -> None:
        """reset() で MID に戻る."""
        clf = VolatilityRegimeClassifier()
        clf.classify(4.0)
        assert clf.current_cluster == VolatilityCluster.EXTREME
        clf.reset()
        assert clf.current_cluster == VolatilityCluster.MID

    def test_jump_from_low_to_extreme(self) -> None:
        """LOW から一気に EXTREME へジャンプ可能."""
        cfg = VolatilityClusterConfig(hysteresis=0.0)
        clf = VolatilityRegimeClassifier(cfg)
        clf.classify(0.3)
        assert clf.current_cluster == VolatilityCluster.LOW
        clf.classify(5.0)
        assert clf.current_cluster == VolatilityCluster.EXTREME

    def test_offset_mult_reflects_cluster(self) -> None:
        """current_offset_mult がクラスタに連動."""
        cfg = VolatilityClusterConfig(
            hysteresis=0.0,
            low_offset_mult=0.7,
            high_offset_mult=1.5,
        )
        clf = VolatilityRegimeClassifier(cfg)

        clf.classify(0.3)
        assert clf.current_offset_mult == 0.7

        clf.classify(2.0)
        assert clf.current_offset_mult == 1.5


# =====================================================================
# セルフレビュー TG4-TG5: テストギャップ補完
# =====================================================================


class TestReviewGapsSigma:
    """セルフレビューで特定されたテストギャップの補完."""

    def test_tg4_negative_vol_ratio(self) -> None:
        """TG4: 負の vol_ratio が LOW に分類されること (ガード追加済み)."""
        clf = VolatilityRegimeClassifier()
        result = clf.classify(-1.0)
        assert result == VolatilityCluster.LOW

    def test_tg5_threshold_order_validation(self) -> None:
        """TG5: 閾値逆転時に ValueError."""
        with pytest.raises(ValueError, match="low < high < extreme"):
            VolatilityClusterConfig(low_threshold=2.0, high_threshold=1.0, extreme_threshold=3.0)

    def test_tg5_threshold_equal_raises(self) -> None:
        """TG5: 同一閾値でも ValueError."""
        with pytest.raises(ValueError, match="low < high < extreme"):
            VolatilityClusterConfig(low_threshold=1.0, high_threshold=1.0, extreme_threshold=3.0)
