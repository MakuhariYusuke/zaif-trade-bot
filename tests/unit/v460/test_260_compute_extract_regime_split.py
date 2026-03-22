"""260# テスト — compute() extract method + regime_boosts 5-split.

P2-2: _apply_loss_boost() / _apply_ffd_boost() 抽出
P2-3: _apply_regime_boosts() → 5 sub-method 分割
"""
from __future__ import annotations

import pytest

from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from tests.unit.v460._fill_test_source import (
    MAKER_PRICE,
    MAKER_REGIME_BOOST,
    read_class_method_source,
)


# ======================================================================
# P2-2: compute() extract method
# ======================================================================


class TestComputeExtractMethod:
    """260# P2-2: loss_boost / FFD boost のパイプラインステージ抽出."""

    @staticmethod
    def _maker_price_source(method_name: str) -> str:
        return read_class_method_source(MAKER_PRICE, "MakerPriceCalculator", method_name)

    def test_compute_calls_apply_loss_boost(self) -> None:
        """compute() が loss_boost stage 経由で _apply_loss_boost を使う."""
        src = self._maker_price_source("compute")
        assert "_apply_offset_ratio_stage(" in src
        assert '"loss_boost"' in src
        assert "self._apply_loss_boost" in src

    def test_compute_calls_preflight_helpers(self) -> None:
        """compute() が preflight/cache resolve helper を経由する."""
        src = self._maker_price_source("compute")
        assert "self._resolve_cached_imbalance(" in src
        assert "await self._resolve_market_snapshot(" in src
        assert "self._refresh_market_state(" in src
        assert "self._enforce_spread_guards(" in src

    def test_compute_calls_cross_venue_veto_helper(self) -> None:
        """compute() が veto raise helper を経由する."""
        src = self._maker_price_source("compute")
        assert "self._raise_cross_venue_veto_if_needed()" in src

    def test_compute_calls_apply_ffd_boost(self) -> None:
        """compute() が _apply_ffd_boost() を呼び出す."""
        src = self._maker_price_source("compute")
        assert "_apply_ffd_boost(" in src

    def test_compute_no_inline_exp(self) -> None:
        """compute() 内に直接 math.exp() が存在しない (抽出済み)."""
        src = self._maker_price_source("compute")
        assert "exp(" not in src, "math.exp() should be in _apply_loss_boost(), not compute()"

    def test_compute_no_inline_boost_mult(self) -> None:
        """compute() 内に直接 boost_mult が存在しない (抽出済み)."""
        src = self._maker_price_source("compute")
        assert "boost_mult" not in src, "boost_mult should be in _apply_ffd_boost(), not compute()"

    def test_apply_loss_boost_has_decay(self) -> None:
        """_apply_loss_boost() に指数減衰ロジックが含まれる."""
        src = self._maker_price_source("_apply_loss_boost")
        assert "exp(" in src
        assert "_loss_boost_mult" in src
        assert "_loss_boost_set_time" in src

    def test_apply_ffd_boost_has_clamp(self) -> None:
        """_apply_ffd_boost() に max_offset_ratio クランプが含まれる."""
        src = self._maker_price_source("_apply_ffd_boost")
        assert "_scale_offset_ratio" in src
        assert "max_ratio" in src

    def test_apply_ffd_boost_returns_tuple(self) -> None:
        """_apply_ffd_boost() が (offset_ratio, offset) のタプルを返す."""
        import typing
        hints = typing.get_type_hints(MakerPrice._apply_ffd_boost)
        assert hints.get("return") == tuple[float, float]

    def test_compute_line_count_reduced(self) -> None:
        """compute() が 370 行以下に維持されている (214→180→192, 266# pipeline, 303# C passive MM, 305# OB cache, 306# stage recording + ceiling, 310# sell_hour_boost, 320# C-1 side-specific ceiling, 421# final clamp, 439# cross-venue, 543# OFI-Lite/δ*, 545# OFI boost)."""
        src = self._maker_price_source("compute")
        line_count = len(src.strip().splitlines())
        assert line_count <= 370, (
            f"compute() should be <= 370 lines (was 214, now {line_count})"
        )


# ======================================================================
# P2-3: _apply_regime_boosts() 6-split (397# mid_confidence 追加)
# ======================================================================


class TestRegimeBoostsSplit:
    """260# P2-3: _apply_regime_boosts() → 6 sub-method 分割."""

    @staticmethod
    def _maker_price_source(method_name: str) -> str:
        return read_class_method_source(MAKER_REGIME_BOOST, "RegimeBoostMixin", method_name)

    def test_regime_boosts_is_dispatcher(self) -> None:
        """_apply_regime_boosts() が 6 sub-method を呼び出すディスパッチャー."""
        src = self._maker_price_source("_apply_regime_boosts")
        assert "_regime_boost_trending(" in src
        assert "_regime_boost_high_vol(" in src
        assert "_regime_boost_ranging(" in src
        assert "_regime_boost_low_vol(" in src
        assert "_regime_boost_unknown_buy(" in src
        assert "_regime_boost_mid_confidence(" in src  # 397#

    def test_regime_boosts_line_count(self) -> None:
        """_apply_regime_boosts() が 30 行以下のディスパッチャー."""
        src = self._maker_price_source("_apply_regime_boosts")
        line_count = len(src.strip().splitlines())
        assert line_count <= 30, f"Dispatcher should be <= 30 lines, got {line_count}"

    def test_trending_method_exists(self) -> None:
        """_regime_boost_trending() が存在し is_trending チェックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_trending")
        src = self._maker_price_source("_regime_boost_trending")
        assert "is_trending" in src

    def test_high_vol_method_exists(self) -> None:
        """_regime_boost_high_vol() が存在し HIGH_VOL チェックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_high_vol")
        src = self._maker_price_source("_regime_boost_high_vol")
        assert "HIGH_VOL" in src

    def test_ranging_method_exists(self) -> None:
        """_regime_boost_ranging() が存在し RANGING + OBI ロジックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_ranging")
        src = self._maker_price_source("_regime_boost_ranging")
        assert "RANGING" in src
        assert "obi" in src.lower()

    def test_low_vol_method_exists(self) -> None:
        """_regime_boost_low_vol() が存在し last_volatility_ratio を使用."""
        assert hasattr(MakerPrice, "_regime_boost_low_vol")
        src = self._maker_price_source("_regime_boost_low_vol")
        assert "last_volatility_ratio" in src

    def test_unknown_buy_method_exists(self) -> None:
        """_regime_boost_unknown_buy() が存在し UNKNOWN + buy/sell ガードを含む."""
        assert hasattr(MakerPrice, "_regime_boost_unknown_buy")
        src = self._maker_price_source("_regime_boost_unknown_buy")
        assert "UNKNOWN" in src
        # 440# buy/sell 両対応化
        assert "unknown_buy_offset_boost" in src
        assert "unknown_sell_offset_boost" in src

    def test_mid_confidence_method_exists(self) -> None:
        """397# _regime_boost_mid_confidence() が存在し confidence 帯域チェックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_mid_confidence")
        src = self._maker_price_source("_regime_boost_mid_confidence")
        assert "mid_confidence_offset_boost" in src
        assert "current_confidence" in src

    def test_each_sub_method_under_60_lines(self) -> None:
        """各 sub-method が 60 行以下 (440# ranging side 非対称化で増加)."""
        methods = [
            "_regime_boost_trending",
            "_regime_boost_high_vol",
            "_regime_boost_ranging",
            "_regime_boost_low_vol",
            "_regime_boost_unknown_buy",
            "_regime_boost_mid_confidence",  # 397#
        ]
        for name in methods:
            src = self._maker_price_source(name)
            lines = len(src.strip().splitlines())
            assert lines <= 60, f"{name} should be <= 60 lines, got {lines}"
