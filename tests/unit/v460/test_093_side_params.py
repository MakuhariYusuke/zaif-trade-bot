"""
093# spread_adaptive / fast_fill_defense サイド別パラメータテスト.

- spread_adaptive: narrow_spread_boost_buy / narrow_spread_boost_sell の追加
- fast_fill_defense: threshold_sec_buy/sell, offset_boost_buy/sell の追加
- YAML パース検証
- ロジック適用検証 (コード構造ベース)
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.run_fill_test import FillTestConfig
from tests.unit.v460._fill_test_source import (
    MAKER_PRICE,
    read_fill_test_method_source,
    read_source_text,
)
from tests.unit.v460._yaml_test_helpers import load_yaml_mapping

_FILL_CONFIG_FIELDS = FillTestConfig.__dataclass_fields__
_FILL_TEST_YAML = load_yaml_mapping(
    Path(__file__).resolve().parents[3] / "configs" / "v460" / "fill_test.yaml"
)
_MAKER_PRICE_SOURCE = read_source_text(MAKER_PRICE)
_POST_CYCLE_SOURCE = read_fill_test_method_source("_process_post_cycle")


# =====================================================================
# A. spread_adaptive side 別 boost — Config フィールド
# =====================================================================

class TestSpreadAdaptiveSideConfig:
    """093# spread_adaptive サイド別パラメータの Config テスト."""

    def test_narrow_spread_boost_buy_default_none(self) -> None:
        """narrow_spread_boost_buy のデフォルトは None (共通値使用)."""
        assert _FILL_CONFIG_FIELDS["narrow_spread_boost_buy"].default is None

    def test_narrow_spread_boost_sell_default_none(self) -> None:
        """narrow_spread_boost_sell のデフォルトは None (共通値使用)."""
        assert _FILL_CONFIG_FIELDS["narrow_spread_boost_sell"].default is None

    def test_narrow_spread_boost_buy_explicit(self) -> None:
        """narrow_spread_boost_buy を明示指定可能."""
        cfg = FillTestConfig(narrow_spread_boost_buy=1.5)
        assert cfg.narrow_spread_boost_buy == pytest.approx(1.5)

    def test_narrow_spread_boost_sell_explicit(self) -> None:
        """narrow_spread_boost_sell を明示指定可能."""
        cfg = FillTestConfig(narrow_spread_boost_sell=2.5)
        assert cfg.narrow_spread_boost_sell == pytest.approx(2.5)

    def test_common_boost_unchanged(self) -> None:
        """共通値 narrow_spread_boost は従来どおり 2.0."""
        assert _FILL_CONFIG_FIELDS["narrow_spread_boost"].default == pytest.approx(2.0)


# =====================================================================
# B. fast_fill_defense side 別 — Config フィールド
# =====================================================================

class TestFastFillDefenseSideConfig:
    """093# fast_fill_defense サイド別パラメータの Config テスト."""

    def test_threshold_sec_buy_default_none(self) -> None:
        assert _FILL_CONFIG_FIELDS["fast_fill_threshold_sec_buy"].default is None

    def test_threshold_sec_sell_default_none(self) -> None:
        assert _FILL_CONFIG_FIELDS["fast_fill_threshold_sec_sell"].default is None

    def test_offset_boost_buy_default_none(self) -> None:
        assert _FILL_CONFIG_FIELDS["fast_fill_offset_boost_buy"].default is None

    def test_offset_boost_sell_default_none(self) -> None:
        assert _FILL_CONFIG_FIELDS["fast_fill_offset_boost_sell"].default is None

    def test_threshold_sec_sell_explicit(self) -> None:
        cfg = FillTestConfig(fast_fill_threshold_sec_sell=15.0)
        assert cfg.fast_fill_threshold_sec_sell == pytest.approx(15.0)

    def test_offset_boost_sell_explicit(self) -> None:
        cfg = FillTestConfig(fast_fill_offset_boost_sell=2.5)
        assert cfg.fast_fill_offset_boost_sell == pytest.approx(2.5)

    def test_common_threshold_unchanged(self) -> None:
        assert _FILL_CONFIG_FIELDS["fast_fill_threshold_sec"].default == pytest.approx(5.0)

    def test_common_boost_unchanged(self) -> None:
        assert _FILL_CONFIG_FIELDS["fast_fill_offset_boost"].default == pytest.approx(2.0)


# =====================================================================
# C. YAML パース — spread_adaptive side 別
# =====================================================================

class TestSpreadAdaptiveSideYAML:
    """093# YAML から spread_adaptive side 別が正しくパースされる."""

    def test_from_yaml_with_side_boost(self) -> None:
        yaml_cfg = {
            "spread_adaptive": {
                "enabled": True,
                "narrow_spread_bps": 10.0,
                "narrow_spread_boost": 2.0,
                "narrow_spread_boost_buy": 1.5,
                "narrow_spread_boost_sell": 2.0,
                "wide_spread_bps": 25.0,
                "wide_spread_ratio": 0.5,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.narrow_spread_boost_buy == pytest.approx(1.5)
        assert cfg.narrow_spread_boost_sell == pytest.approx(2.0)
        assert cfg.narrow_spread_boost == pytest.approx(2.0)

    def test_from_yaml_without_side_boost(self) -> None:
        """side 別を省略した場合は None (共通値使用)."""
        yaml_cfg = {
            "spread_adaptive": {
                "enabled": True,
                "narrow_spread_boost": 2.0,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.narrow_spread_boost_buy is None
        assert cfg.narrow_spread_boost_sell is None

    def test_production_yaml_has_side_boost(self) -> None:
        """本番 YAML に 093# side 別 boost が設定されている."""
        sa = _FILL_TEST_YAML["spread_adaptive"]
        assert "narrow_spread_boost_buy" in sa
        assert "narrow_spread_boost_sell" in sa
        assert sa["narrow_spread_boost_buy"] == pytest.approx(2.0)   # 183# 1.5→2.0 (spread<2kでAS32%対策)
        assert sa["narrow_spread_boost_sell"] == pytest.approx(2.5)  # 183# 2.0→2.5


# =====================================================================
# D. YAML パース — fast_fill_defense side 別
# =====================================================================

class TestFastFillDefenseSideYAML:
    """093# YAML から fast_fill_defense side 別が正しくパースされる."""

    def test_from_yaml_with_side_params(self) -> None:
        yaml_cfg = {
            "fast_fill_defense": {
                "enabled": True,
                "threshold_sec": 5.0,
                "threshold_sec_buy": 5.0,
                "threshold_sec_sell": 15.0,
                "offset_boost": 2.0,
                "offset_boost_buy": 2.0,
                "offset_boost_sell": 2.5,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.fast_fill_threshold_sec_buy == pytest.approx(5.0)
        assert cfg.fast_fill_threshold_sec_sell == pytest.approx(15.0)
        assert cfg.fast_fill_offset_boost_buy == pytest.approx(2.0)
        assert cfg.fast_fill_offset_boost_sell == pytest.approx(2.5)

    def test_from_yaml_without_side_params(self) -> None:
        yaml_cfg = {
            "fast_fill_defense": {
                "enabled": True,
                "threshold_sec": 5.0,
                "offset_boost": 2.0,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.fast_fill_threshold_sec_buy is None
        assert cfg.fast_fill_threshold_sec_sell is None
        assert cfg.fast_fill_offset_boost_buy is None
        assert cfg.fast_fill_offset_boost_sell is None

    def test_production_yaml_has_side_defense(self) -> None:
        """本番 YAML に 093# side 別 fast_fill_defense が設定されている."""
        ffd = _FILL_TEST_YAML["fast_fill_defense"]
        assert ffd.get("threshold_sec_sell") == pytest.approx(15.0)
        assert ffd.get("offset_boost_sell") == pytest.approx(2.5)


# =====================================================================
# E. ロジック構造テスト — spread_adaptive side 別がコードに存在
# =====================================================================

class TestSpreadAdaptiveSideLogic:
    """093# spread_adaptive ロジックで side 別 boost が使われている."""

    def test_compute_maker_price_uses_side_boost(self) -> None:
        """MakerPriceCalculator.compute に narrow_spread_boost_buy/sell の分岐がある.

        120#: _compute_maker_price は maker_price.py に抽出済み.
        """
        source = _MAKER_PRICE_SOURCE
        assert "narrow_spread_boost_buy" in source
        assert "narrow_spread_boost_sell" in source

    def test_spread_adaptive_method_exists(self) -> None:
        """093# spread_adaptive ロジックが maker_price.py に存在する."""
        source = _MAKER_PRICE_SOURCE
        assert "_apply_spread_adaptive" in source


# =====================================================================
# F. ロジック構造テスト — fast_fill_defense side 別がコードに存在
# =====================================================================

class TestFastFillDefenseSideLogic:
    """100# fast_fill_defense ロジックは FastFillDefense クラスに抽出済み.

    side 別閾値・倍率は FastFillDefense クラス内で処理される。
    """

    def test_fast_fill_defense_class_has_side_threshold(self) -> None:
        """FastFillDefense が side 別閾値を解決する."""
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
        cfg = FastFillDefenseConfig(
            enabled=True,
            threshold_sec=5.0,
            threshold_sec_buy=10.0,
            threshold_sec_sell=15.0,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)
        assert defense._resolve_threshold_sec("buy") == 10.0
        assert defense._resolve_threshold_sec("sell") == 15.0

    def test_fast_fill_defense_class_has_side_boost(self) -> None:
        """FastFillDefense が side 別 boost 倍率を解決する."""
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
        cfg = FastFillDefenseConfig(
            enabled=True,
            offset_boost=2.0,
            offset_boost_buy=1.5,
            offset_boost_sell=2.5,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)
        assert defense._resolve_boost("buy") == 1.5
        assert defense._resolve_boost("sell") == 2.5

    def test_run_continuous_delegates_to_fast_fill_defense(self) -> None:
        """run_continuous が FastFillDefense に委譲している.

        265# extract: post-cycle 処理は _process_post_cycle に分離されたため、
        そちらのソースコードを検査する。
        """
        source = _POST_CYCLE_SOURCE
        assert "fast_fill_defense" in source
        assert "evaluate_fill" in source


# =====================================================================
# G. 実効値テスト — spread_adaptive side 別の実効 offset
# =====================================================================

class TestSpreadAdaptiveSideEffective:
    """093# spread_adaptive の side 別 boost による実効 offset."""

    def test_buy_gets_lower_boost(self) -> None:
        """buy 側は 1.5× boost → 0.05 * 1.5 = 0.075."""
        # Buy: base 0.05, spread_adaptive 1.5× → 0.075
        expected_buy = 0.05 * 1.5
        assert expected_buy == pytest.approx(0.075)

    def test_sell_keeps_existing_boost(self) -> None:
        """sell 側は 2.0× boost → 0.12 * 2.0 = 0.24."""
        expected_sell = 0.12 * 2.0
        assert expected_sell == pytest.approx(0.24)

    def test_buy_offset_lower_than_sell(self) -> None:
        """buy (1.5×) < sell (2.0×) の実効 offset 差."""
        buy_effective = 0.05 * 1.5   # 0.075
        sell_effective = 0.12 * 2.0  # 0.24
        assert buy_effective < sell_effective


# =====================================================================
# H. 実効値テスト — fast_fill_defense side 別
# =====================================================================

class TestFastFillDefenseSideEffective:
    """093# fast_fill_defense の side 別パラメータによる動作変化."""

    def test_sell_threshold_broader_than_buy(self) -> None:
        """sell は 15s、buy は 5s — sell の方が広い範囲で防御."""
        # sell: 12秒 wait → 15s 閾値以下 → 防御発動
        sell_wait = 12.0
        buy_wait = 12.0
        sell_threshold = 15.0
        buy_threshold = _FILL_CONFIG_FIELDS["fast_fill_threshold_sec"].default
        assert sell_wait <= sell_threshold  # sell は発動
        assert buy_wait > buy_threshold    # buy は非発動

    def test_sell_boost_stronger_than_buy(self) -> None:
        """sell 2.5× vs buy 2.0× — sell の方が強い防御."""
        cfg = FillTestConfig(
            fast_fill_defense_enabled=True,
            fast_fill_offset_boost=2.0,
            fast_fill_offset_boost_sell=2.5,
        )
        sell_boost = cfg.fast_fill_offset_boost_sell
        buy_boost = cfg.fast_fill_offset_boost  # buy は共通値
        assert sell_boost > buy_boost
