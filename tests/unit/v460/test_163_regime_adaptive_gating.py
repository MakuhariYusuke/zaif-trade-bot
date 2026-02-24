"""163# テスト: TimeFilter regime-adaptive 動的ゲーティング.

107# Phase 3 Step 2 + regime 連動ゲーティングの検証。
- Step 2: BUY=[16], SELL=[8], global=[] (基本遮断)
- high_vol regime 時: BUY += [8,18], SELL += [4,14] (安全ネット)
- FillTestConfig 新フィールドのパース
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.time_filter import TimeFilter


# ── helpers ──────────────────────────────────────────────


def _make_config(**overrides: object) -> FillTestConfig:
    """regime-adaptive 有効な Step 2 設定を生成."""
    defaults: dict[str, object] = dict(
        enable_time_filter=True,
        skip_utc_hours=[],          # Step 2: global block 廃止
        skip_utc_hours_buy=[16],     # UTC16 のみ
        skip_utc_hours_sell=[8],     # UTC08 のみ
        regime_adaptive_enabled=True,
        regime_adaptive_extra_buy=[8, 18],
        regime_adaptive_extra_sell=[4, 14],
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)  # type: ignore[arg-type]


def _mock_hour(hour: int):
    """datetime.now(utc).hour を任意の値に差し替える fixture."""
    mock_dt = datetime(2025, 2, 26, hour, 30, 0, tzinfo=timezone.utc)
    return patch(
        "scripts.v460.lib.time_filter.datetime",
        wraps=datetime,
        **{"now.return_value": mock_dt},  # type: ignore[arg-type]
    )


# ── Step 2 基本テスト ───────────────────────────────────


class TestStep2BasicFiltering:
    """107# Phase 3 Step 2: BUY=[16], SELL=[8], global=[]."""

    def test_buy_blocked_at_utc16(self) -> None:
        """BUY は UTC16 でブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(16):
            assert tf.is_filtered("buy") is True

    def test_buy_not_blocked_at_utc08(self) -> None:
        """BUY は UTC08 では非ブロック (Step 2 で解除)."""
        tf = TimeFilter(_make_config())
        with _mock_hour(8):
            assert tf.is_filtered("buy", regime="ranging") is False

    def test_sell_blocked_at_utc08(self) -> None:
        """SELL は UTC08 でブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(8):
            assert tf.is_filtered("sell") is True

    def test_sell_not_blocked_at_utc16(self) -> None:
        """SELL は UTC16 では非ブロック (Step 2 で解除)."""
        tf = TimeFilter(_make_config())
        with _mock_hour(16):
            assert tf.is_filtered("sell", regime="ranging") is False

    def test_no_global_blocking(self) -> None:
        """global=[] のため side=None では全時間帯パス."""
        tf = TimeFilter(_make_config())
        with _mock_hour(16):
            assert tf.is_filtered() is False
        with _mock_hour(8):
            assert tf.is_filtered() is False


# ── regime-adaptive テスト ───────────────────────────────


class TestRegimeAdaptiveFiltering:
    """163#: high_vol regime 時は追加遮断が有効."""

    def test_buy_extra_blocked_utc08_high_vol(self) -> None:
        """high_vol regime → BUY は UTC08 でもブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(8):
            assert tf.is_filtered("buy", regime="high_vol") is True

    def test_buy_extra_blocked_utc18_high_vol(self) -> None:
        """high_vol regime → BUY は UTC18 でもブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(18):
            assert tf.is_filtered("buy", regime="high_vol") is True

    def test_sell_extra_blocked_utc04_high_vol(self) -> None:
        """high_vol regime → SELL は UTC04 でもブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(4):
            assert tf.is_filtered("sell", regime="high_vol") is True

    def test_sell_extra_blocked_utc14_high_vol(self) -> None:
        """high_vol regime → SELL は UTC14 でもブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(14):
            assert tf.is_filtered("sell", regime="high_vol") is True

    def test_buy_utc08_not_blocked_ranging(self) -> None:
        """ranging regime → BUY UTC08 は非ブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(8):
            assert tf.is_filtered("buy", regime="ranging") is False

    def test_sell_utc04_not_blocked_trending_up(self) -> None:
        """trending_up regime → SELL UTC04 は非ブロック."""
        tf = TimeFilter(_make_config())
        with _mock_hour(4):
            assert tf.is_filtered("sell", regime="trending_up") is False

    def test_no_extra_block_when_regime_none(self) -> None:
        """regime=None → 追加遮断なし."""
        tf = TimeFilter(_make_config())
        with _mock_hour(8):
            assert tf.is_filtered("buy", regime=None) is False
        with _mock_hour(18):
            assert tf.is_filtered("buy", regime=None) is False

    def test_regime_adaptive_disabled(self) -> None:
        """regime_adaptive_enabled=False → high_vol でも追加なし."""
        cfg = _make_config(regime_adaptive_enabled=False)
        tf = TimeFilter(cfg)
        with _mock_hour(8):
            assert tf.is_filtered("buy", regime="high_vol") is False
        with _mock_hour(18):
            assert tf.is_filtered("buy", regime="high_vol") is False


# ── effective hours 網羅テスト ────────────────────────────


class TestEffectiveBlockedHours:
    """全24時間の遮断テーブルを検証."""

    @pytest.fixture
    def tf(self) -> TimeFilter:
        return TimeFilter(_make_config())

    def test_buy_effective_hours_normal(self, tf: TimeFilter) -> None:
        """通常 regime: BUY = {16} のみ."""
        blocked = set()
        for h in range(24):
            with _mock_hour(h):
                if tf.is_filtered("buy", regime="ranging"):
                    blocked.add(h)
        assert blocked == {16}

    def test_buy_effective_hours_high_vol(self, tf: TimeFilter) -> None:
        """high_vol regime: BUY = {8, 16, 18} = Step 1 相当."""
        blocked = set()
        for h in range(24):
            with _mock_hour(h):
                if tf.is_filtered("buy", regime="high_vol"):
                    blocked.add(h)
        assert blocked == {8, 16, 18}

    def test_sell_effective_hours_normal(self, tf: TimeFilter) -> None:
        """通常 regime: SELL = {8} のみ."""
        blocked = set()
        for h in range(24):
            with _mock_hour(h):
                if tf.is_filtered("sell", regime="ranging"):
                    blocked.add(h)
        assert blocked == {8}

    def test_sell_effective_hours_high_vol(self, tf: TimeFilter) -> None:
        """high_vol regime: SELL = {4, 8, 14} = Step 1 相当."""
        blocked = set()
        for h in range(24):
            with _mock_hour(h):
                if tf.is_filtered("sell", regime="high_vol"):
                    blocked.add(h)
        assert blocked == {4, 8, 14}


# ── FillTestConfig YAML パーステスト ─────────────────────


class TestFillTestConfigRegimeAdaptive:
    """163# FillTestConfig が regime_adaptive フィールドを正しくパースする."""

    def test_from_yaml_parses_regime_adaptive(self) -> None:
        """YAML から regime_adaptive_* が正しく読み込まれる."""
        import yaml
        from pathlib import Path

        yaml_path = Path("configs/v460/fill_test.yaml")
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = FillTestConfig.from_yaml(raw)
        assert cfg.regime_adaptive_enabled is True
        assert cfg.regime_adaptive_extra_buy == [8, 18]
        assert cfg.regime_adaptive_extra_sell == [4, 14]

    def test_default_values(self) -> None:
        """デフォルト値: regime_adaptive_enabled=False, 他=None."""
        cfg = FillTestConfig()
        assert cfg.regime_adaptive_enabled is False
        assert cfg.regime_adaptive_extra_buy is None
        assert cfg.regime_adaptive_extra_sell is None

    def test_step2_yaml_values(self) -> None:
        """YAML が Step 2 の値に更新されている."""
        import yaml
        from pathlib import Path

        yaml_path = Path("configs/v460/fill_test.yaml")
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        tf = raw["time_filter"]
        assert tf["skip_utc_hours"] == []
        assert tf["skip_utc_hours_buy"] == [16]
        assert tf["skip_utc_hours_sell"] == [8]
