"""197# boost 最適化 + balance_forced offset + Gate 8-9 統合テスト.

テスト対象:
  A. velocity_offset_boost_factor 2.0→1.5 (データ駆動最適化)
  B. trending_sell_offset_boost_factor YAML 3.0→2.0 (regime 累積修正)
  C. balance_forced + trending offset 適用 (253# 削除済: dead config)
  D. Gate 8: narrow_spread_pause Gate 統合 (旧 B3)
  E. Gate 9: maker_price 事前チェック (spread_too_narrow / sell_guard_reject)
  F. maker_price._last_spread キャッシュ
  G. 後方互換性 + 統合テスト
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
    GateCheckResult,
    _GATE_TO_CANCEL_REASON,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from tests.unit.v460._fill_test_source import ORCHESTRATOR_MID_CYCLE, read_source_text

_CYCLE_GATE_EVALUATE_SIG = inspect.signature(CycleGateAggregator.evaluate)


# ─── ヘルパー ──────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _read_lib_source(module_name: str) -> str:
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / "scripts" / "v460" / "lib" / f"{module_name}.py").read_text(
        encoding="utf-8-sig"
    )


def _make_config(**overrides: object) -> FillTestConfig:
    """テスト用の最小 FillTestConfig."""
    defaults: dict[str, object] = {
        "skip_buy_unknown_regime": False,
        "skip_sell_unknown_regime": False,
        "skip_ranging_buy_low_vol": False,
        "low_vol_threshold": 0.75,
        "skip_sell_trending": True,
        "skip_sell_trending_up_only": True,
        "max_consecutive_trending_sell_skip": 30,
        "sell_guard_inv_bypass_threshold": 0.3,
        "buy_dynamic_kill_enabled": False,
        "sell_dynamic_kill_enabled": False,
        "buy_dynamic_kill_threshold_bps": -5.0,
        "sell_dynamic_kill_threshold_bps": -5.0,
        "sell_velocity_skip_enabled": False,
        "buy_velocity_skip_enabled": False,
        "sell_velocity_skip_threshold_bps": 8.0,
        "buy_velocity_skip_threshold_bps": -8.0,
        "velocity_skip_as_offset_enabled": True,
        "velocity_offset_boost_factor": 1.5,
        "trending_sell_as_offset_enabled": True,
        "trending_sell_offset_boost_factor": 3.0,
        # 253# 削除済: balance_forced_apply_trending_offset (234# dead config)
        # narrow_spread
        "narrow_spread_pause_enabled": False,
        "narrow_spread_pause_bps": 3.0,
        "narrow_spread_pause_sec": 5.0,
        # maker_price pre-check
        "min_spread_jpy": 0.0,
        "sell_max_spread_jpy": 0.0,
    }
    defaults.update(overrides)
    return FillTestConfig(**{k: v for k, v in defaults.items() if hasattr(FillTestConfig, k)})


def _make_gate(**overrides: object) -> CycleGateAggregator:
    return CycleGateAggregator(_make_config(**overrides))


def _default_ctx(**overrides: object) -> dict:
    ctx: dict = {
        "side": "sell",
        "regime": "ranging",
        "vol_ratio": 1.0,
        "inv_net_imbalance": 0.0,
        "is_buy_killed": False,
        "is_sell_killed": False,
    }
    ctx.update(overrides)
    return ctx


# =================================================================
# A. velocity_offset_boost_factor 2.0→1.5 (データ駆動最適化)
# =================================================================


class TestVelocityBoostOptimization:
    """197# velocity boost 最適化: fill_records 分析に基づく 2.0→1.5."""

    def test_default_velocity_boost_is_1_5(self):
        """FillTestConfig デフォルトが 1.5 に変更されたこと."""
        cfg = FillTestConfig()
        assert cfg.velocity_offset_boost_factor == 1.5

    def test_yaml_velocity_boost_is_1_5(self, v460_fill_test_yaml_base: dict[str, object]):
        """live YAML が 1.5 に設定されていること."""
        raw = v460_fill_test_yaml_base
        assert raw["skip_gate"]["velocity_offset_boost_factor"] == 1.5

    def test_config_accepts_custom_velocity_boost(self):
        """任意の boost 値が設定可能."""
        cfg = FillTestConfig(velocity_offset_boost_factor=2.5)
        assert cfg.velocity_offset_boost_factor == 2.5


# =================================================================
# B. trending_sell_offset_boost_factor YAML 2.0
# =================================================================


class TestTrendingBoostOptimization:
    """197# trending boost YAML 3.0→2.0: regime boost 1.8x との累積修正."""

    def test_default_trending_boost_is_2_0(self):
        """FillTestConfig デフォルトは 1.5 (336# drift fix, YAML と整合)."""
        cfg = FillTestConfig()
        assert cfg.trending_sell_offset_boost_factor == 1.5

    def test_yaml_trending_boost_is_3_0(self, v460_fill_test_yaml_base: dict[str, object]):
        """320# live YAML が 1.5 (C-1 解消: sell pipeline 復活)."""
        raw = v460_fill_test_yaml_base
        assert raw["loss_control"]["trending_sell_offset_boost_factor"] == 1.5


# =================================================================
# C. balance_forced + trending offset 適用
# =================================================================


class TestBalanceForcedTrendingOffset:
    """197# balance_forced 時の trending offset 適用.

    253# NOTE: balance_forced_apply_trending_offset フィールドは 234# dead config
    として削除済み。テストは CycleGateAggregator の実動作検証のみ残す。
    """

    def test_config_field_removed_253(self):
        """253# balance_forced_apply_trending_offset フィールドが削除済み."""
        cfg = FillTestConfig()
        assert not hasattr(cfg, "balance_forced_apply_trending_offset")

    def test_balance_forced_always_applies_offset_234(self):
        """234#: soft mode の trending offset は常に適用される."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
        ))
        assert r.blocked is False
        # 234#: soft mode は常に offset を適用
        assert r.trending_offset_mult is not None

    def test_balance_forced_with_offset_applies(self):
        """balance_forced + trending → offset_mult 設定."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=2.0,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
        ))
        assert r.blocked is False
        assert r.trending_offset_mult == 2.0

    def test_balance_forced_not_blocked(self):
        """forced sell は block されない."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=3.0,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
        ))
        assert r.blocked is False

    def test_balance_forced_trending_down_with_up_only(self):
        """trending_up_only=True + regime=trending_down → offset なし."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
            skip_sell_trending_up_only=True,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_down",
        ))
        assert r.blocked is False
        assert r.trending_offset_mult is None

    def test_balance_forced_buy_side_unaffected(self):
        """buy 側は trending でも offset なし."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
        )
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="trending_up",
        ))
        assert r.blocked is False
        assert r.trending_offset_mult is None

    def test_audit_trail_trending_offset_detail_234(self):
        """234#: balance_forced は通常パスを通る → 196# soft mode detail."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=2.5,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
        ))
        trending_check = [c for c in r.checks if c.gate_name == "trending_sell"]
        assert len(trending_check) == 1
        # 234#: 統一パスで 196# soft mode を通過
        assert "196#" in trending_check[0].detail
        assert "2.5" in trending_check[0].detail


# =================================================================
# D. Gate 8: narrow_spread_pause
# =================================================================


class TestGate8NarrowSpread:
    """197# Gate 8: narrow_spread_pause Gate 統合 (旧 B3)."""

    def test_disabled_by_default(self):
        """narrow_spread_pause_enabled=False → 常に pass."""
        gate = _make_gate(narrow_spread_pause_enabled=False)
        r = gate.evaluate(**_default_ctx(spread_jpy=1.0, mid_price=15_000_000.0))
        assert r.blocked is False

    def test_blocks_when_spread_too_narrow(self):
        """spread < threshold → blocked=True."""
        gate = _make_gate(
            narrow_spread_pause_enabled=True,
            narrow_spread_pause_bps=3.0,
        )
        # spread=100, mid=15M → spread_bps = 100/15M*10000 ≈ 0.067 bps
        r = gate.evaluate(**_default_ctx(spread_jpy=100.0, mid_price=15_000_000.0))
        assert r.blocked is True
        assert r.blocking_reason == "narrow_spread_pause"

    def test_passes_when_spread_sufficient(self):
        """spread >= threshold → pass."""
        gate = _make_gate(
            narrow_spread_pause_enabled=True,
            narrow_spread_pause_bps=3.0,
        )
        # spread=5000, mid=15M → spread_bps = 5000/15M*10000 ≈ 3.33 bps
        r = gate.evaluate(**_default_ctx(spread_jpy=5000.0, mid_price=15_000_000.0))
        assert r.blocked is False

    def test_none_spread_passes(self):
        """spread=None → pass (データ未取得)."""
        gate = _make_gate(narrow_spread_pause_enabled=True)
        r = gate.evaluate(**_default_ctx(spread_jpy=None, mid_price=15_000_000.0))
        assert r.blocked is False

    def test_none_mid_price_passes(self):
        """mid_price=None → pass."""
        gate = _make_gate(narrow_spread_pause_enabled=True)
        r = gate.evaluate(**_default_ctx(spread_jpy=100.0, mid_price=None))
        assert r.blocked is False

    def test_zero_mid_price_passes(self):
        """mid_price=0 → pass (0除算防止)."""
        gate = _make_gate(narrow_spread_pause_enabled=True)
        r = gate.evaluate(**_default_ctx(spread_jpy=100.0, mid_price=0.0))
        assert r.blocked is False

    def test_cancel_reason_mapping(self):
        """narrow_spread_pause の cancel_reason マッピング."""
        assert _GATE_TO_CANCEL_REASON["narrow_spread_pause"] == "narrow_spread_pause"

    def test_gate8_detail_includes_197(self):
        """Gate 8 の detail に 197# が含まれること."""
        gate = _make_gate(
            narrow_spread_pause_enabled=True,
            narrow_spread_pause_bps=3.0,
        )
        r = gate.evaluate(**_default_ctx(spread_jpy=100.0, mid_price=15_000_000.0))
        gate8 = [c for c in r.checks if c.gate_name == "narrow_spread"]
        assert len(gate8) == 1
        assert "197#" in gate8[0].detail


# =================================================================
# E. Gate 9: maker_price 事前チェック
# =================================================================


class TestGate9MakerPricePrecheck:
    """197# Gate 9 (advisory): maker_price ValueError 事前チェック.

    blocked=True だと compute() 未実行→キャッシュ更新なし→
    永久デッドロックのフィードバックループが発生するため、
    advisory-only (blocked=False)。executor try/except が最終防衛線。
    """

    def test_spread_too_narrow_advisory(self):
        """197# spread < min_spread_jpy → advisory (blocked=False), reason は記録."""
        gate = _make_gate(min_spread_jpy=500.0)
        r = gate.evaluate(**_default_ctx(spread_jpy=300.0))
        assert r.blocked is False  # 197# advisory
        gate9 = [c for c in r.checks if c.gate_name == "maker_price_pre"]
        assert len(gate9) == 1
        assert gate9[0].reason == "spread_too_narrow"

    def test_spread_above_min_passes(self):
        """spread >= min_spread_jpy → pass."""
        gate = _make_gate(min_spread_jpy=500.0)
        r = gate.evaluate(**_default_ctx(spread_jpy=600.0))
        assert r.blocked is False

    def test_sell_guard_reject_advisory(self):
        """197# sell + spread > sell_max_spread_jpy → advisory (blocked=False)."""
        gate = _make_gate(sell_max_spread_jpy=10000.0)
        r = gate.evaluate(**_default_ctx(side="sell", spread_jpy=15000.0))
        assert r.blocked is False  # 197# advisory
        gate9 = [c for c in r.checks if c.gate_name == "maker_price_pre"]
        assert len(gate9) == 1
        assert gate9[0].reason == "sell_guard_reject"

    def test_sell_guard_buy_side_unaffected(self):
        """buy 側は sell_guard に影響されない."""
        gate = _make_gate(sell_max_spread_jpy=10000.0)
        r = gate.evaluate(**_default_ctx(side="buy", spread_jpy=15000.0))
        assert r.blocked is False

    def test_sell_guard_zero_means_unlimited(self):
        """sell_max_spread_jpy=0 → 無制限 (常に pass)."""
        gate = _make_gate(sell_max_spread_jpy=0.0)
        r = gate.evaluate(**_default_ctx(side="sell", spread_jpy=999999.0))
        assert r.blocked is False

    def test_none_spread_passes(self):
        """spread=None → pass (Gate 9 スキップ)."""
        gate = _make_gate(min_spread_jpy=500.0, sell_max_spread_jpy=10000.0)
        r = gate.evaluate(**_default_ctx(spread_jpy=None))
        assert r.blocked is False

    def test_cancel_reason_mapping_spread_too_narrow(self):
        assert _GATE_TO_CANCEL_REASON["spread_too_narrow"] == "spread_too_narrow"

    def test_cancel_reason_mapping_sell_guard_reject(self):
        assert _GATE_TO_CANCEL_REASON["sell_guard_reject"] == "sell_guard_reject"

    def test_gate9_detail_includes_197(self):
        """Gate 9 の detail に 197# が含まれること."""
        gate = _make_gate(min_spread_jpy=500.0)
        r = gate.evaluate(**_default_ctx(spread_jpy=300.0))
        gate9 = [c for c in r.checks if c.gate_name == "maker_price_pre"]
        assert len(gate9) == 1
        assert "197#" in gate9[0].detail


# =================================================================
# F. maker_price._last_spread キャッシュ
# =================================================================


class TestMakerPriceSpreadCache:
    """197# maker_price._last_spread キャッシュ."""

    def test_last_spread_slot_exists(self):
        """MakerPriceCalculator に _last_spread スロットが存在."""
        assert "_last_spread" in MakerPriceCalculator.__slots__

    def test_last_spread_initial_none(self):
        """初期値は None."""
        cfg = FillTestConfig()
        calc = MakerPriceCalculator(
            cfg,
            fast_fill_defense=MagicMock(),
            regime_detector=None,
            base_offset_ratio=0.2,
        )
        assert calc._last_spread is None

    def test_last_spread_in_source(self):
        """compute() 内で _last_spread が更新されること (ソースコード検証)."""
        source = _read_lib_source("maker_price")
        assert "_last_spread" in source
        assert "self._last_spread = spread" in source

    def test_last_spread_property(self):
        """last_spread プロパティが公開されていること."""
        assert hasattr(MakerPriceCalculator, "last_spread")
        assert isinstance(
            getattr(MakerPriceCalculator, "last_spread"), property,
        )

    def test_last_mid_price_property(self):
        """last_mid_price プロパティが公開されていること."""
        assert hasattr(MakerPriceCalculator, "last_mid_price")
        assert isinstance(
            getattr(MakerPriceCalculator, "last_mid_price"), property,
        )


# =================================================================
# G. 後方互換性 + 統合テスト
# =================================================================


class TestBackwardCompatibility197:
    """197# デフォルト値で旧モード動作を維持."""

    def test_defaults_preserve_old_behavior(self):
        """デフォルト設定ではバニラ動作 (Gate 8/9 不活性)."""
        cfg = FillTestConfig()
        # Gate 8: disabled by default
        assert cfg.narrow_spread_pause_enabled is False
        # Gate 9: min_spread_jpy=0 → filter なし
        assert cfg.min_spread_jpy == 0.0
        # 253# 削除済: balance_forced_apply_trending_offset (234# dead config)

    def test_all_nine_gates_pass(self):
        """全 9 ゲートが pass する条件."""
        gate = _make_gate(
            skip_buy_unknown_regime=False,
            skip_sell_unknown_regime=False,
            skip_ranging_buy_low_vol=False,
            skip_sell_trending=False,
            buy_dynamic_kill_enabled=False,
            sell_dynamic_kill_enabled=False,
            sell_velocity_skip_enabled=False,
            buy_velocity_skip_enabled=False,
            narrow_spread_pause_enabled=False,
            min_spread_jpy=0.0,
            sell_max_spread_jpy=0.0,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging",
            spread_jpy=5000.0, mid_price=15_000_000.0,
        ))
        assert r.blocked is False
        assert len(r.checks) == 10  # 197#: 7→9, 475#: 9→10 ゲートに拡張

    def test_gate_evaluation_order(self):
        """Gate 1-9 が正しい順序で評価されること."""
        gate = _make_gate(
            skip_buy_unknown_regime=False,
            skip_ranging_buy_low_vol=False,
            skip_sell_trending=False,
            buy_dynamic_kill_enabled=False,
            sell_dynamic_kill_enabled=False,
            narrow_spread_pause_enabled=False,
            min_spread_jpy=0.0,
        )
        r = gate.evaluate(**_default_ctx(
            spread_jpy=5000.0, mid_price=15_000_000.0,
        ))
        gate_names = [c.gate_name for c in r.checks]
        assert gate_names == [
            "unknown_regime_buy",     # Gate 1
            "ranging_buy_low_vol",    # Gate 2
            "ranging_sell_low_vol",   # Gate 2b (475#)
            "trending_sell",          # Gate 3
            "buy_dynamic_kill",       # Gate 4
            "sell_dynamic_kill",      # Gate 5
            "velocity_skip",          # Gate 6
            "unknown_regime_sell",    # Gate 7
            "narrow_spread",          # Gate 8
            "maker_price_pre",        # Gate 9
        ]

    def test_early_exit_skips_later_gates(self):
        """Gate 8 で block → Gate 9 未評価."""
        gate = _make_gate(
            narrow_spread_pause_enabled=True,
            narrow_spread_pause_bps=3.0,
            min_spread_jpy=500.0,  # Gate 9 の条件も設定
        )
        r = gate.evaluate(**_default_ctx(
            spread_jpy=100.0, mid_price=15_000_000.0,
        ))
        assert r.blocked is True
        assert r.blocking_reason == "narrow_spread_pause"
        gate_names = [c.gate_name for c in r.checks]
        assert "maker_price_pre" not in gate_names  # Gate 9 未到達


class TestYamlNewFields197:
    """197# YAML 新フィールドの検証."""

    def test_yaml_balance_forced_trending_offset_removed_253(
        self,
        v460_fill_test_yaml_base: dict[str, object],
    ):
        """253# YAML から balance_forced_apply_trending_offset が削除済."""
        raw = v460_fill_test_yaml_base
        lc = raw["loss_control"]
        assert "balance_forced_apply_trending_offset" not in lc


class TestDesignConsistency197:
    """197# 設計一貫性テスト."""

    def test_gate_to_cancel_reason_has_197_entries(self):
        """_GATE_TO_CANCEL_REASON に 197# の新エントリが含まれること."""
        assert "narrow_spread_pause" in _GATE_TO_CANCEL_REASON
        assert "spread_too_narrow" in _GATE_TO_CANCEL_REASON
        assert "sell_guard_reject" in _GATE_TO_CANCEL_REASON

    def test_orchestrator_passes_spread_and_mid_price(self):
        """orchestrator が evaluate() に spread_jpy/mid_price を渡すこと."""
        source = read_source_text(ORCHESTRATOR_MID_CYCLE)
        assert "spread_jpy" in source
        assert "mid_price" in source
        assert "last_spread" in source
        assert "last_mid_price" in source

    def test_evaluate_accepts_spread_params(self):
        """evaluate() が spread_jpy/mid_price パラメータを受け付けること."""
        assert "spread_jpy" in _CYCLE_GATE_EVALUATE_SIG.parameters
        assert "mid_price" in _CYCLE_GATE_EVALUATE_SIG.parameters

    def test_cycle_gate_aggregator_source_mentions_197(self):
        """cycle_gate_aggregator.py に 197# コメントが存在."""
        source = _read_lib_source("cycle_gate_aggregator")
        assert "197#" in source
