"""260# テスト — compute() extract method + regime_boosts 5-split.

P2-2: _apply_loss_boost() / _apply_ffd_boost() 抽出
P2-3: _apply_regime_boosts() → 5 sub-method 分割
"""
from __future__ import annotations

import inspect

import pytest

from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice


# ======================================================================
# P2-2: compute() extract method
# ======================================================================


class TestComputeExtractMethod:
    """260# P2-2: loss_boost / FFD boost のパイプラインステージ抽出."""

    def test_compute_calls_apply_loss_boost(self) -> None:
        """compute() が _apply_loss_boost() を呼び出す."""
        src = inspect.getsource(MakerPrice.compute)
        assert "_apply_loss_boost(" in src

    def test_compute_calls_apply_ffd_boost(self) -> None:
        """compute() が _apply_ffd_boost() を呼び出す."""
        src = inspect.getsource(MakerPrice.compute)
        assert "_apply_ffd_boost(" in src

    def test_compute_no_inline_exp(self) -> None:
        """compute() 内に直接 math.exp() が存在しない (抽出済み)."""
        src = inspect.getsource(MakerPrice.compute)
        assert "exp(" not in src, "math.exp() should be in _apply_loss_boost(), not compute()"

    def test_compute_no_inline_boost_mult(self) -> None:
        """compute() 内に直接 boost_mult が存在しない (抽出済み)."""
        src = inspect.getsource(MakerPrice.compute)
        assert "boost_mult" not in src, "boost_mult should be in _apply_ffd_boost(), not compute()"

    def test_apply_loss_boost_has_decay(self) -> None:
        """_apply_loss_boost() に指数減衰ロジックが含まれる."""
        src = inspect.getsource(MakerPrice._apply_loss_boost)
        assert "exp(" in src
        assert "_loss_boost_mult" in src
        assert "_loss_boost_set_time" in src

    def test_apply_ffd_boost_has_clamp(self) -> None:
        """_apply_ffd_boost() に max_offset_ratio クランプが含まれる."""
        src = inspect.getsource(MakerPrice._apply_ffd_boost)
        assert "_scale_offset_ratio" in src
        assert "max_ratio" in src

    def test_apply_ffd_boost_returns_tuple(self) -> None:
        """_apply_ffd_boost() が (offset_ratio, offset) のタプルを返す."""
        import typing
        hints = typing.get_type_hints(MakerPrice._apply_ffd_boost)
        assert hints.get("return") == tuple[float, float]

    def test_compute_line_count_reduced(self) -> None:
        """compute() が 235 行以下に維持されている (214→180→192, 266# pipeline, 303# C passive MM, 305# OB cache)."""
        src = inspect.getsource(MakerPrice.compute)
        line_count = len(src.strip().splitlines())
        assert line_count <= 235, (
            f"compute() should be <= 235 lines (was 214, now {line_count})"
        )


# ======================================================================
# P2-3: _apply_regime_boosts() 5-split
# ======================================================================


class TestRegimeBoostsSplit:
    """260# P2-3: _apply_regime_boosts() → 5 sub-method 分割."""

    def test_regime_boosts_is_dispatcher(self) -> None:
        """_apply_regime_boosts() が 5 sub-method を呼び出すディスパッチャー."""
        src = inspect.getsource(MakerPrice._apply_regime_boosts)
        assert "_regime_boost_trending(" in src
        assert "_regime_boost_high_vol(" in src
        assert "_regime_boost_ranging(" in src
        assert "_regime_boost_low_vol(" in src
        assert "_regime_boost_unknown_buy(" in src

    def test_regime_boosts_line_count(self) -> None:
        """_apply_regime_boosts() が 30 行以下のディスパッチャー."""
        src = inspect.getsource(MakerPrice._apply_regime_boosts)
        line_count = len(src.strip().splitlines())
        assert line_count <= 30, f"Dispatcher should be <= 30 lines, got {line_count}"

    def test_trending_method_exists(self) -> None:
        """_regime_boost_trending() が存在し is_trending チェックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_trending")
        src = inspect.getsource(MakerPrice._regime_boost_trending)
        assert "is_trending" in src

    def test_high_vol_method_exists(self) -> None:
        """_regime_boost_high_vol() が存在し HIGH_VOL チェックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_high_vol")
        src = inspect.getsource(MakerPrice._regime_boost_high_vol)
        assert "HIGH_VOL" in src

    def test_ranging_method_exists(self) -> None:
        """_regime_boost_ranging() が存在し RANGING + OBI ロジックを含む."""
        assert hasattr(MakerPrice, "_regime_boost_ranging")
        src = inspect.getsource(MakerPrice._regime_boost_ranging)
        assert "RANGING" in src
        assert "obi" in src.lower()

    def test_low_vol_method_exists(self) -> None:
        """_regime_boost_low_vol() が存在し last_volatility_ratio を使用."""
        assert hasattr(MakerPrice, "_regime_boost_low_vol")
        src = inspect.getsource(MakerPrice._regime_boost_low_vol)
        assert "last_volatility_ratio" in src

    def test_unknown_buy_method_exists(self) -> None:
        """_regime_boost_unknown_buy() が存在し UNKNOWN + buy ガードを含む."""
        assert hasattr(MakerPrice, "_regime_boost_unknown_buy")
        src = inspect.getsource(MakerPrice._regime_boost_unknown_buy)
        assert "UNKNOWN" in src
        assert 'side == "buy"' in src

    def test_each_sub_method_under_50_lines(self) -> None:
        """各 sub-method が 50 行以下."""
        methods = [
            "_regime_boost_trending",
            "_regime_boost_high_vol",
            "_regime_boost_ranging",
            "_regime_boost_low_vol",
            "_regime_boost_unknown_buy",
        ]
        for name in methods:
            src = inspect.getsource(getattr(MakerPrice, name))
            lines = len(src.strip().splitlines())
            assert lines <= 50, f"{name} should be <= 50 lines, got {lines}"
