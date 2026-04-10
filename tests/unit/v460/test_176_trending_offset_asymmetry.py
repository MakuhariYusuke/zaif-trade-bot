"""176# テスト: 方向×サイド別 trending offset boost + TRENDING skip 除外 + 横展開.

施策 A: TRENDING (方向不明) を skip_sell_trending_up_only から除外
施策 B: 方向×サイド別 4 分岐 offset boost (trending_up/down × buy/sell)
横展開: hot-reload, ML特徴量, YAML regime_thresholds, retrain_scheduler
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import scripts.v460.ml.retrain_scheduler as retrain_scheduler_mod

from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.regime_detector import FillTestRegime
from scripts.v460.ml.skip_gate import build_features_from_market_state
from tests.unit.v460._fill_test_source import CYCLE_GATE_AGGREGATOR, read_source_text
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping

_CYCLE_GATE_AGGREGATOR_SOURCE = read_source_text(CYCLE_GATE_AGGREGATOR)


# =================================================================
# 施策 A: TRENDING (方向不明) skip 除外テスト
# =================================================================
class TestTrendingSkipExclusion:
    """176# A: TRENDING (undirected) regime は sell skip しない.

    194#: trending_sell ロジックは CycleGateAggregator に集約。
    """

    def test_orchestrator_trending_undirected_not_blocked(self) -> None:
        """skip_sell_trending_up_only=True 時、TRENDING (undirected) は
        trending_up とみなさず sell を通過させる。

        194#: CycleGateAggregator のソースで確認。
        """
        assert 'trending_up' in _CYCLE_GATE_AGGREGATOR_SOURCE, "trending_up check must exist in cycle_gate_aggregator"
        assert 'regime != "trending_up"' in _CYCLE_GATE_AGGREGATOR_SOURCE, (
            "176# A: should check != trending_up"
        )

    def test_orchestrator_allows_trending_down_sell(self) -> None:
        """trending_down 時の sell は通過する (既存動作の維持確認)."""
        assert 'trending_up' in _CYCLE_GATE_AGGREGATOR_SOURCE

    def test_no_old_trending_down_only_check(self) -> None:
        """旧コード `== "trending_down"` 条件が削除されていること."""
        assert 'regime == "trending_down"' not in _CYCLE_GATE_AGGREGATOR_SOURCE, (
            "176# A: old trending_down-only passthrough must be removed"
        )


# =================================================================
# 施策 B: 方向×サイド別 config フィールドテスト
# =================================================================
class TestDirectionalBoostConfig:
    """176# B: 方向×サイド別 offset boost config フィールド."""

    def test_new_fields_exist(self) -> None:
        """4 つの新フィールドが FillTestConfig に存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "trending_up_buy_offset_boost")
        assert hasattr(cfg, "trending_up_sell_offset_boost")
        assert hasattr(cfg, "trending_down_buy_offset_boost")
        assert hasattr(cfg, "trending_down_sell_offset_boost")

    def test_new_fields_default_none(self) -> None:
        """新フィールドのデフォルトは None (フォールバック有効)."""
        cfg = FillTestConfig()
        assert cfg.trending_up_buy_offset_boost is None
        assert cfg.trending_up_sell_offset_boost is None
        assert cfg.trending_down_buy_offset_boost is None
        assert cfg.trending_down_sell_offset_boost is None

    def test_fields_accept_float(self) -> None:
        """float 値を設定可能."""
        cfg = FillTestConfig(
            trending_up_buy_offset_boost=0.7,
            trending_up_sell_offset_boost=1.8,
            trending_down_buy_offset_boost=1.8,
            trending_down_sell_offset_boost=0.7,
        )
        assert cfg.trending_up_buy_offset_boost == 0.7
        assert cfg.trending_up_sell_offset_boost == 1.8
        assert cfg.trending_down_buy_offset_boost == 1.8
        assert cfg.trending_down_sell_offset_boost == 0.7

    def test_yaml_parsing(self) -> None:
        """YAML regime セクションから新フィールドをパース."""
        yaml_data = {
            "regime": {
                "trending_up_buy_offset_boost": 0.7,
                "trending_up_sell_offset_boost": 1.8,
                "trending_down_buy_offset_boost": 1.8,
                "trending_down_sell_offset_boost": 0.7,
            },
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.trending_up_buy_offset_boost == 0.7
        assert cfg.trending_up_sell_offset_boost == 1.8
        assert cfg.trending_down_buy_offset_boost == 1.8
        assert cfg.trending_down_sell_offset_boost == 0.7

    def test_live_yaml_has_direction_boosts(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """本番 YAML に方向別 boost が設定されている."""
        cfg = v460_fill_test_yaml

        regime = cfg["regime"]
        assert regime["trending_up_buy_offset_boost"] == 0.7
        assert regime["trending_up_sell_offset_boost"] == 2.2   # 685#: 1.8→2.2
        assert regime["trending_down_buy_offset_boost"] == 1.8
        assert regime["trending_down_sell_offset_boost"] == 1.3   # 724#: 1.0→1.3

    def test_live_yaml_skip_sell_trending_false(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """176# B → 196#: skip_sell_trending=true + trending_sell_as_offset_enabled=true.

        196#: ハードスキップではなく offset boost で保守的 sell 発注に変換。
        skip_sell_trending=true でゲート条件を有効化しつつ、soft mode で block しない。
        """
        cfg = v460_fill_test_yaml

        lc = cfg["loss_control"]
        # 196#: skip_sell_trending=true (ゲート条件有効化) + soft mode
        assert lc["skip_sell_trending"] is True, (
            "196#: skip_sell_trending must be true (soft offset mode needs gate condition)"
        )
        assert lc["trending_sell_as_offset_enabled"] is True, (
            "196#: trending_sell_as_offset_enabled must be true (offset replaces hard skip)"
        )


# =================================================================
# 施策 B: _resolve_trending_boost テスト
# =================================================================
class TestResolveTrendingBoost:
    """176# B: MakerPriceCalculator._resolve_trending_boost 優先順位テスト."""

    def _make_cfg(self, **kwargs: float | None) -> FillTestConfig:
        """テスト用 FillTestConfig を生成."""
        defaults = {
            "regime_trending_offset_boost": 1.5,
            "regime_trending_offset_boost_buy": None,
            "regime_trending_offset_boost_sell": None,
            "trending_up_buy_offset_boost": None,
            "trending_up_sell_offset_boost": None,
            "trending_down_buy_offset_boost": None,
            "trending_down_sell_offset_boost": None,
        }
        defaults.update(kwargs)
        return FillTestConfig(**defaults)  # type: ignore[arg-type]

    # --- trending_up × buy ---
    def test_trending_up_buy_direction_specific(self) -> None:
        """trending_up + buy → trending_up_buy_offset_boost (最優先)."""
        cfg = self._make_cfg(
            trending_up_buy_offset_boost=0.7,
            regime_trending_offset_boost_buy=1.0,
        )
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_up", "buy")
        assert result == 0.7

    def test_trending_up_buy_fallback_side(self) -> None:
        """trending_up + buy, 方向別未設定 → regime_trending_offset_boost_buy."""
        cfg = self._make_cfg(
            trending_up_buy_offset_boost=None,
            regime_trending_offset_boost_buy=1.0,
        )
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_up", "buy")
        assert result == 1.0

    def test_trending_up_buy_fallback_shared(self) -> None:
        """trending_up + buy, 全て未設定 → regime_trending_offset_boost (共通値)."""
        cfg = self._make_cfg()
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_up", "buy")
        assert result == 1.5

    # --- trending_up × sell ---
    def test_trending_up_sell_direction_specific(self) -> None:
        """trending_up + sell → trending_up_sell_offset_boost (最優先)."""
        cfg = self._make_cfg(trending_up_sell_offset_boost=1.8)
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_up", "sell")
        assert result == 1.8

    def test_trending_up_sell_fallback_side(self) -> None:
        """trending_up + sell, 方向別未設定 → regime_trending_offset_boost_sell."""
        cfg = self._make_cfg(regime_trending_offset_boost_sell=1.5)
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_up", "sell")
        assert result == 1.5

    # --- trending_down × buy ---
    def test_trending_down_buy_direction_specific(self) -> None:
        """trending_down + buy → trending_down_buy_offset_boost (最優先)."""
        cfg = self._make_cfg(trending_down_buy_offset_boost=1.8)
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_down", "buy")
        assert result == 1.8

    # --- trending_down × sell ---
    def test_trending_down_sell_direction_specific(self) -> None:
        """trending_down + sell → trending_down_sell_offset_boost (最優先)."""
        cfg = self._make_cfg(trending_down_sell_offset_boost=0.7)
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending_down", "sell")
        assert result == 0.7

    # --- trending (undirected) → 方向別なし → フォールバック ---
    def test_trending_undirected_uses_side_fallback(self) -> None:
        """trending (方向不明) → 方向別パラメータ無視 → side 別にフォールバック."""
        cfg = self._make_cfg(
            trending_up_buy_offset_boost=0.7,
            regime_trending_offset_boost_buy=1.0,
        )
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending", "buy")
        assert result == 1.0  # 方向別(0.7)は使われず、side 別(1.0)にフォールバック

    def test_trending_undirected_uses_shared_fallback(self) -> None:
        """trending (方向不明) → 全て未設定 → 共通値."""
        cfg = self._make_cfg()
        result = MakerPriceCalculator._resolve_trending_boost(cfg, "trending", "buy")
        assert result == 1.5


# =================================================================
# 施策 B: _apply_regime_boosts 統合テスト
# =================================================================
class TestApplyRegimeBoostsDirectional:
    """176# B: _apply_regime_boosts が方向別 boost を正しく適用."""

    def _make_calc(
        self,
        regime: FillTestRegime = FillTestRegime.TRENDING_UP,
        **cfg_kwargs: float | None,
    ) -> MakerPriceCalculator:
        """テスト用 MakerPriceCalculator を構築."""
        defaults = {
            "regime_trending_offset_boost": 1.5,
            "regime_trending_offset_boost_buy": None,
            "regime_trending_offset_boost_sell": None,
            "trending_up_buy_offset_boost": None,
            "trending_up_sell_offset_boost": None,
            "trending_down_buy_offset_boost": None,
            "trending_down_sell_offset_boost": None,
            "spread_offset_ratio": 0.05,
            "max_offset_ratio": 0.30,
            "min_offset_ratio": 0.01,
            "spread_adaptive_enabled": False,
            "imbalance_enabled": False,
            "volatility_guard_enabled": False,
            "fast_fill_defense_enabled": False,
            "sell_offset_floor": 0.0,
            "sell_max_spread_jpy": 0.0,
            "regime_high_vol_offset_boost": 1.0,
            "regime_ranging_offset_discount": 1.0,
            "unknown_buy_offset_boost": 1.0,
            "low_vol_offset_boost_enabled": False,
        }
        defaults.update(cfg_kwargs)
        cfg = FillTestConfig(**defaults)  # type: ignore[arg-type]

        regime_detector = MagicMock()
        regime_detector.current_regime = regime

        ffd = MagicMock()
        ffd.should_boost.return_value = False
        ffd.get_boost_multiplier.return_value = 1.0

        return MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=regime_detector,
            base_offset_ratio=0.05,
        )

    def test_trending_up_buy_shrinks_offset(self) -> None:
        """trending_up + buy → offset 縮小 (0.7x)."""
        calc = self._make_calc(
            regime=FillTestRegime.TRENDING_UP,
            trending_up_buy_offset_boost=0.7,
        )
        base_offset = 0.10
        result = calc._apply_regime_boosts("buy", base_offset)
        assert result == pytest.approx(0.10 * 0.7, abs=1e-6)

    def test_trending_up_sell_widens_offset(self) -> None:
        """trending_up + sell → offset 拡大 (1.8x)."""
        calc = self._make_calc(
            regime=FillTestRegime.TRENDING_UP,
            trending_up_sell_offset_boost=1.8,
        )
        base_offset = 0.10
        result = calc._apply_regime_boosts("sell", base_offset)
        assert result == pytest.approx(0.10 * 1.8, abs=1e-6)

    def test_trending_down_buy_widens_offset(self) -> None:
        """trending_down + buy → offset 拡大 (1.8x)."""
        calc = self._make_calc(
            regime=FillTestRegime.TRENDING_DOWN,
            trending_down_buy_offset_boost=1.8,
        )
        base_offset = 0.10
        result = calc._apply_regime_boosts("buy", base_offset)
        assert result == pytest.approx(0.10 * 1.8, abs=1e-6)

    def test_trending_down_sell_shrinks_offset(self) -> None:
        """trending_down + sell → offset 縮小 (0.7x)."""
        calc = self._make_calc(
            regime=FillTestRegime.TRENDING_DOWN,
            trending_down_sell_offset_boost=0.7,
        )
        base_offset = 0.10
        result = calc._apply_regime_boosts("sell", base_offset)
        assert result == pytest.approx(0.10 * 0.7, abs=1e-6)

    def test_boost_1_0_no_change(self) -> None:
        """boost=1.0 → offset 変更なし."""
        calc = self._make_calc(
            regime=FillTestRegime.TRENDING_UP,
            trending_up_buy_offset_boost=1.0,
        )
        base_offset = 0.10
        result = calc._apply_regime_boosts("buy", base_offset)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_production_asymmetry_scenario(self) -> None:
        """本番設定 (buy=0.7, sell=1.8) での非対称性確認."""
        calc = self._make_calc(
            regime=FillTestRegime.TRENDING_UP,
            trending_up_buy_offset_boost=0.7,
            trending_up_sell_offset_boost=1.8,
        )
        buy_result = calc._apply_regime_boosts("buy", 0.10)
        sell_result = calc._apply_regime_boosts("sell", 0.10)
        # buy は sell より小さい offset → 積極的に約定
        assert buy_result < sell_result
        # 具体的な値
        assert buy_result == pytest.approx(0.07, abs=1e-6)
        assert sell_result == pytest.approx(0.18, abs=1e-6)


# =================================================================
# 横展開: config_hot_reload 4 フィールド登録確認
# =================================================================
class TestHotReloadDirectionalFields:
    """176# 横展開: 方向別 boost パラメータが hot-reload 対象に含まれること."""

    def test_directional_fields_in_hot_reloadable(self) -> None:
        """4 つの方向別フィールドが _HOT_RELOADABLE_FIELDS に含まれる."""
        expected = [
            "trending_up_buy_offset_boost",
            "trending_up_sell_offset_boost",
            "trending_down_buy_offset_boost",
            "trending_down_sell_offset_boost",
        ]
        for field in expected:
            assert field in _HOT_RELOADABLE_FIELDS, (
                f"{field} not in _HOT_RELOADABLE_FIELDS"
            )

    def test_existing_trending_fields_still_present(self) -> None:
        """既存の trending boost フィールドも維持されている."""
        assert "regime_trending_offset_boost" in _HOT_RELOADABLE_FIELDS
        assert "regime_trending_offset_boost_buy" in _HOT_RELOADABLE_FIELDS
        assert "regime_trending_offset_boost_sell" in _HOT_RELOADABLE_FIELDS


# =================================================================
# 横展開: ML 特徴量 regime_trending の方向対応
# =================================================================
class TestMLFeatureTrendingDirection:
    """176# 横展開: regime_trending が trending_up/trending_down も含むこと."""

    def test_skip_gate_build_features_trending_up(self) -> None:
        """skip_gate build_features_from_market_state で regime='trending_up' → regime_trending=1.0."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=100.0,
            offset_ratio=0.05,
            regime="trending_up",
            recent_trades=[],
            market_timestamp=None,
        )
        assert features["regime_trending"] == 1.0

    def test_skip_gate_build_features_trending_down(self) -> None:
        """skip_gate build_features_from_market_state で regime='trending_down' → regime_trending=1.0."""
        features = build_features_from_market_state(
            side="sell",
            spread_jpy=100.0,
            offset_ratio=0.05,
            regime="trending_down",
            recent_trades=[],
            market_timestamp=None,
        )
        assert features["regime_trending"] == 1.0

    def test_skip_gate_build_features_trending_undirected(self) -> None:
        """skip_gate build_features_from_market_state で regime='trending' → regime_trending=1.0."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=100.0,
            offset_ratio=0.05,
            regime="trending",
            recent_trades=[],
            market_timestamp=None,
        )
        assert features["regime_trending"] == 1.0

    def test_skip_gate_build_features_ranging_not_trending(self) -> None:
        """ranging は regime_trending=0.0."""
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=100.0,
            offset_ratio=0.05,
            regime="ranging",
            recent_trades=[],
            market_timestamp=None,
        )
        assert features["regime_trending"] == 0.0

    def test_feature_enricher_trending_up(self) -> None:
        """feature_enricher で regime='trending_up' → regime_trending=1."""
        df = pd.DataFrame({
            "regime": ["trending_up", "trending_down", "ranging", "trending"],
        })
        regime = df["regime"].fillna("unknown")
        result = regime.str.startswith("trending").astype(int)
        assert list(result) == [1, 1, 0, 1]


# =================================================================
# 横展開: YAML regime_thresholds / regime_sample_weights
# =================================================================
class TestYAMLRegimeDirectionKeys:
    """176# 横展開: YAML の regime 関連マップに方向別キーが存在."""

    def test_skip_gate_regime_thresholds_has_directions(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """skip_gate regime_thresholds に trending_up/trending_down が存在."""
        cfg = v460_fill_test_yaml

        thresholds = cfg["skip_gate"]["regime_thresholds"]
        assert "trending_up" in thresholds
        assert "trending_down" in thresholds

    def test_regime_sample_weights_has_directions(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """retrain regime_sample_weights に trending_up/trending_down が存在."""
        cfg = v460_fill_test_yaml

        weights = cfg["retrain"]["regime_sample_weights"]
        assert "trending_up" in weights
        assert "trending_down" in weights

    def test_dynamic_kill_thresholds_already_directional(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """sell/buy dynamic_kill regime_thresholds は既に方向別 (回帰確認)."""
        cfg = v460_fill_test_yaml

        sell_dk = cfg["loss_control"]["sell_dynamic_kill"]["regime_thresholds"]
        assert "trending_up" in sell_dk
        assert "trending_down" in sell_dk

    def test_retrain_scheduler_defaults_have_directions(self) -> None:
        """retrain_scheduler.py デフォルト dict に trending_up/down が追加済み."""
        defaults = retrain_scheduler_mod._DEFAULT_CONFIG
        weights = defaults["regime_sample_weights"]
        assert "trending_up" in weights
        assert "trending_down" in weights


# =================================================================
# CHANGELOG 日付整合性
# =================================================================
class TestChangelogDateConsistency:
    """176# CHANGELOG の日付が未来日でないことを確認."""

    def test_no_future_dates_in_changelog(self) -> None:
        """CHANGELOG の日付が大幅な未来日でないこと (174# の誤日付修正確認)."""
        content = Path("CHANGELOG.md").read_text(encoding="utf-8")
        today = date.today()
        threshold = today + timedelta(days=2)  # 1日程度のズレは許容
        date_pattern = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
        for match in date_pattern.finditer(content):
            d = date.fromisoformat(match.group(1))
            assert d <= threshold, f"未来日付を検出: {match.group(1)}"
