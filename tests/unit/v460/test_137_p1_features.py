"""137# P1 テスト — 136# §9 レビュー修正 + 134# P1-06/08/11.

- P1-06: reprice sell側上限縮小
- P1-08: narrow spread pause
- P1-11: PnL fee 控除統一
- §9: RetrainTrigger (mtime修正, freshness, rename, YAML外部化)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===== P1-11: PnL Fee Deduction =====


class TestPnlFeeDeduction:
    """PnlMeasurer の fee 控除テスト."""

    def _make_config(self, *, fee_enabled: bool = True, maker_fee_bps: float = 1.5) -> "FillTestConfig":
        """テスト用 config を簡易生成."""
        from scripts.v460.lib.fill_config import FillTestConfig

        return FillTestConfig(
            pnl_fee_deduction_enabled=fee_enabled,
            maker_fee_bps=maker_fee_bps,
            early_exit_enabled=False,
            post_fill_wait_sec=0.01,
        )

    @pytest.mark.asyncio
    async def test_fee_deducted_from_pnl(self) -> None:
        """fee 有効時に PnL から maker fee が控除される."""
        from scripts.v460.lib.pnl_measurer import PnlMeasurer

        cfg = self._make_config(fee_enabled=True, maker_fee_bps=2.0)
        measurer = PnlMeasurer(cfg)

        # mid_at_fill=100, mid_30s_after=100.1 → raw pnl = +10bps (buy)
        call_count = 0

        async def get_mid() -> float:
            nonlocal call_count
            call_count += 1
            # 1回目: mid_at_fill, 2回目: mid_30s_after
            return 100.0 if call_count == 1 else 100.1

        result = await measurer.measure(
            filled=True,
            fill_price=100.0,
            side="buy",
            get_mid_price=get_mid,
        )
        # raw pnl = (100.1 - 100) / 100 * 10000 = 10 bps
        # after fee: 10 - 2.0 = 8.0 bps
        assert result.post_fill_pnl is not None
        assert result.post_fill_pnl == pytest.approx(8.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_no_fee_when_disabled(self) -> None:
        """fee 無効時は PnL がそのまま."""
        from scripts.v460.lib.pnl_measurer import PnlMeasurer

        cfg = self._make_config(fee_enabled=False, maker_fee_bps=2.0)
        measurer = PnlMeasurer(cfg)

        call_count = 0

        async def get_mid() -> float:
            nonlocal call_count
            call_count += 1
            return 100.0 if call_count == 1 else 100.1

        result = await measurer.measure(
            filled=True,
            fill_price=100.0,
            side="buy",
            get_mid_price=get_mid,
        )
        # raw pnl = 10 bps (no fee deduction)
        assert result.post_fill_pnl is not None
        assert result.post_fill_pnl == pytest.approx(10.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_fee_zero_no_change(self) -> None:
        """fee=0 の場合は控除なし（Coincheck maker 現状）."""
        from scripts.v460.lib.pnl_measurer import PnlMeasurer

        cfg = self._make_config(fee_enabled=True, maker_fee_bps=0.0)
        measurer = PnlMeasurer(cfg)

        call_count = 0

        async def get_mid() -> float:
            nonlocal call_count
            call_count += 1
            return 100.0 if call_count == 1 else 100.05

        result = await measurer.measure(
            filled=True,
            fill_price=100.0,
            side="buy",
            get_mid_price=get_mid,
        )
        # raw pnl = 5 bps, fee=0 → no change
        assert result.post_fill_pnl is not None
        assert result.post_fill_pnl == pytest.approx(5.0, abs=0.1)


# ===== P1-08: Narrow Spread Pause =====


class TestNarrowSpreadPause:
    """narrow_spread_pause 設定テスト."""

    def test_config_defaults(self) -> None:
        """デフォルトは無効."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.narrow_spread_pause_enabled is False
        assert cfg.narrow_spread_pause_bps == 3.0
        assert cfg.narrow_spread_pause_sec == 5.0
        assert cfg.narrow_spread_pause_max_consecutive == 3

    def test_yaml_parsing(self, tmp_path: "Path") -> None:
        """YAML から narrow_spread_pause を正しくパース."""
        from scripts.v460.lib.fill_config import FillTestConfig

        yaml_content = {
            "loss_control": {
                "narrow_spread_pause": {
                    "enabled": True,
                    "threshold_bps": 5.0,
                    "pause_sec": 10.0,
                    "max_consecutive": 5,
                },
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_content)
        assert cfg.narrow_spread_pause_enabled is True
        assert cfg.narrow_spread_pause_bps == 5.0
        assert cfg.narrow_spread_pause_sec == 10.0
        assert cfg.narrow_spread_pause_max_consecutive == 5


# ===== P1-11: Fee Config Parsing =====


class TestFeeConfigParsing:
    """fee 設定 YAML パーステスト."""

    def test_defaults(self) -> None:
        """デフォルトは無効."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.pnl_fee_deduction_enabled is False
        assert cfg.maker_fee_bps == 0.0
        assert cfg.taker_fee_bps == 0.0

    def test_yaml_parsing(self, tmp_path: "Path") -> None:
        """YAML から PnL fee 設定をパース."""
        from scripts.v460.lib.fill_config import FillTestConfig

        yaml_content = {
            "loss_control": {
                "pnl_fee_deduction": {
                    "enabled": True,
                    "maker_fee_bps": 1.5,
                    "taker_fee_bps": 3.0,
                },
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_content)
        assert cfg.pnl_fee_deduction_enabled is True
        assert cfg.maker_fee_bps == 1.5
        assert cfg.taker_fee_bps == 3.0


# ===== P1-06: Reprice Sell Max Config =====


class TestRepriceSellMax:
    """reprice sell側上限の YAML テスト."""

    def test_sell_max_reprice_default(self) -> None:
        """デフォルト stale_max_reprice_sell は None (共通値使用)."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.stale_max_reprice_sell is None
        assert cfg.stale_max_reprice == 2  # 共通デフォルト

    def test_yaml_sell_reprice_override(self, tmp_path: "Path") -> None:
        """YAML で sell 側 max_reprice を上書きできる."""
        from scripts.v460.lib.fill_config import FillTestConfig

        yaml_content = {
            "stale_order": {
                "enabled": True,
                "max_reprice_sell": 1,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_content)
        assert cfg.stale_max_reprice_sell == 1


# ===== YAML trigger config externalization (§9 #4) =====


class TestTriggerYamlConfig:
    """§9 #4: RetrainTriggerConfig の YAML 外部化テスト."""

    def test_all_config_fields_have_defaults(self) -> None:
        """全フィールドにデフォルト値が設定されている."""
        from ztb.ml.retrain_trigger import RetrainTriggerConfig

        cfg = RetrainTriggerConfig()
        assert cfg.backoff_multiplier == 2.0
        assert cfg.backoff_max_interval_sec == 14400
        assert cfg.check_feature_freshness is False
        assert cfg.feature_trades_stale_hours == 6.0
        assert cfg.feature_ob_stale_hours == 6.0

    def test_config_override(self) -> None:
        """フィールドをオーバーライドできる."""
        from ztb.ml.retrain_trigger import RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            backoff_multiplier=3.0,
            check_feature_freshness=True,
            feature_trades_stale_hours=12.0,
        )
        assert cfg.backoff_multiplier == 3.0
        assert cfg.check_feature_freshness is True
        assert cfg.feature_trades_stale_hours == 12.0
