"""337# テスト: Sell-side 損益悪化対策の検証.

テスト観点:
  - sell_dynamic_kill threshold 緩和 (YAML 反映)
  - sell_dynamic_kill_inv_relaxation 新設 (config/parser/guards)
  - balance_forced_switch PnL フィルタリング (_track_side_pnl)
  - Ho & Stoll 対称性の構造的検証
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_parser import _parse_stopgap_section
from scripts.v460.lib.orchestrator_guards import OrchestratorGuardsMixin
from tests.unit.v460._fill_test_source import ORCHESTRATOR_GUARDS, read_source_text

_GUARDS_SOURCE = read_source_text(ORCHESTRATOR_GUARDS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Config フィールド存在・デフォルト値テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSellInvRelaxationConfig:
    """337# sell_dynamic_kill_inv_relaxation config フィールドの存在確認."""

    def test_config_fields_exist(self) -> None:
        """sell inv_relaxation の 3 フィールドが FillTestConfig に存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "sell_dynamic_kill_inv_relaxation_enabled")
        assert hasattr(cfg, "sell_dynamic_kill_inv_relaxation_scale")
        assert hasattr(cfg, "sell_dynamic_kill_inv_relaxation_max_bps")

    def test_defaults_are_conservative(self) -> None:
        """344# 342#B: inv_bypass 廃止に伴い max_bps を 0.5 に引上げ."""
        cfg = FillTestConfig()
        assert cfg.sell_dynamic_kill_inv_relaxation_enabled is False
        assert cfg.sell_dynamic_kill_inv_relaxation_scale == 0.4
        assert cfg.sell_dynamic_kill_inv_relaxation_max_bps == 0.5

    def test_sell_scale_lt_buy_scale(self) -> None:
        """337# Glosten-Milgrom: sell は buy より保守的."""
        cfg = FillTestConfig()
        assert cfg.sell_dynamic_kill_inv_relaxation_scale < cfg.buy_dynamic_kill_inv_relaxation_scale

    def test_sell_max_bps_compensates_bypass_removal(self) -> None:
        """344# 342#B: sell は inv_bypass 廃止に伴い buy より広い relaxation を持つ."""
        cfg = FillTestConfig()
        assert cfg.sell_dynamic_kill_inv_relaxation_max_bps >= cfg.buy_dynamic_kill_inv_relaxation_max_bps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Parser テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSellInvRelaxationParser:
    """337# YAML parser が sell_dynamic_kill_inv_relaxation を正しく解析."""

    def test_parse_all_fields(self) -> None:
        result = _parse_stopgap_section({
            "止血": {
                "sell_dynamic_kill_inv_relaxation": {
                    "enabled": True,
                    "scale": 0.4,
                    "max_bps": 0.3,
                },
            },
        })
        assert result["sell_dynamic_kill_inv_relaxation_enabled"] is True
        assert result["sell_dynamic_kill_inv_relaxation_scale"] == 0.4
        assert result["sell_dynamic_kill_inv_relaxation_max_bps"] == 0.3

    def test_parse_partial_fields(self) -> None:
        """一部のフィールドだけ指定しても解析できる."""
        result = _parse_stopgap_section({
            "止血": {
                "sell_dynamic_kill_inv_relaxation": {"enabled": True},
            },
        })
        assert result["sell_dynamic_kill_inv_relaxation_enabled"] is True
        assert "sell_dynamic_kill_inv_relaxation_scale" not in result

    def test_parse_empty_section(self) -> None:
        """sell_dynamic_kill_inv_relaxation が無くてもエラーにならない."""
        result = _parse_stopgap_section({"止血": {}})
        assert "sell_dynamic_kill_inv_relaxation_enabled" not in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Guards コード構造テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGuardsStructure:
    """337# orchestrator_guards.py の構造変更検証."""

    def test_sell_inv_relaxation_in_is_side_killed(self) -> None:
        """_is_side_killed に sell inv_relaxation ロジックが存在."""
        assert "sell_dynamic_kill_inv_relaxation_enabled" in _GUARDS_SOURCE

    def test_sell_imbalance_positive_check(self) -> None:
        """sell 側は imbalance > 0 (BTC 過剰) で緩和される."""
        assert "imbalance > 0" in _GUARDS_SOURCE

    def test_buy_imbalance_negative_check(self) -> None:
        """buy 側は imbalance < 0 (BTC 不足) で緩和される (既存)."""
        assert "imbalance < 0" in _GUARDS_SOURCE

    def test_is_side_killed_docstring_mentions_337(self) -> None:
        """_is_side_killed の docstring に 337# 参照がない場合は本体コメント確認."""
        # 337# のロジック追加はコード内コメントで参照される
        src = _GUARDS_SOURCE
        assert "337#" in src, "337# への参照がコード内に存在すること"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Ho-Stoll 対称性の構造テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHoStollSymmetry:
    """337# Ho & Stoll (1981) 対称性: buy/sell config の構造対称性を検証."""

    @pytest.fixture
    def cfg(self) -> FillTestConfig:
        return FillTestConfig()

    def test_both_sides_have_inv_relaxation(self, cfg: FillTestConfig) -> None:
        """buy/sell とも inv_relaxation のフィールドセットが対称的に存在."""
        field_names = {f.name for f in fields(cfg)}
        for prefix in ("buy", "sell"):
            for suffix in ("_enabled", "_scale", "_max_bps"):
                name = f"{prefix}_dynamic_kill_inv_relaxation{suffix}"
                assert name in field_names, f"{name} が FillTestConfig に不在"

    def test_threshold_asymmetry_within_bounds(self, cfg: FillTestConfig) -> None:
        """337# sell/buy threshold 比率が極端に非対称でないこと.

        コードデフォルト値を検証。
        YAML 値は drift prevention test がカバー。
        buy=-1.5, sell=-0.3 (code defaults) → ratio=5.0x。
        341# revert: 340#符号修正でinv_relaxation正常化→YAML sell=-0.3復元。
        コードデフォルトのガードレールは 6x 以下とする。
        """
        buy_t = abs(cfg.buy_dynamic_kill_threshold_bps)
        sell_t = abs(cfg.sell_dynamic_kill_threshold_bps)
        if buy_t > 0 and sell_t > 0:
            ratio = buy_t / sell_t
            assert ratio < 6.0, (
                f"buy/sell threshold 比率={ratio:.1f}x は Ho-Stoll 的に危険域"
            )


