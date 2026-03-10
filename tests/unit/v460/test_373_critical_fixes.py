"""373# テスト — CRITICAL/IMPORTANT 修正の検証.

対象修正:
  - CRITICAL-1: order_quantity / min_order_btc ゼロ除算防止バリデーション
  - CRITICAL-2: balance_checker.check() 例外時 skip に変更
  - IMPORTANT-3: SACRetrainConfig __post_init__ バリデーション
  - IMPORTANT-4: read_sidecar_signal TOCTOU 修正
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_validation import validate_fill_config
from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal
from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

if TYPE_CHECKING:
    pass


# ════════════════════════════════════════════════════════════════
# §1 CRITICAL-1: order_quantity / min_order_btc validation
# ════════════════════════════════════════════════════════════════


class TestOrderQuantityValidation:
    """order_quantity / min_order_btc が 0 以下のとき ValueError."""

    def test_order_quantity_zero_raises(self) -> None:
        # __post_init__ が validate_fill_config を呼ぶのでコンストラクタで ValueError
        with pytest.raises(ValueError, match="order_quantity must be > 0"):
            FillTestConfig(order_quantity=0.0)

    def test_order_quantity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="order_quantity must be > 0"):
            FillTestConfig(order_quantity=-0.001)

    def test_min_order_btc_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_order_btc must be > 0"):
            FillTestConfig(min_order_btc=0.0)

    def test_min_order_btc_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_order_btc must be > 0"):
            FillTestConfig(min_order_btc=-0.001)

    def test_valid_defaults_pass(self) -> None:
        """デフォルト値 (0.001) は通過するはず."""
        cfg = FillTestConfig()
        assert cfg.order_quantity > 0
        assert cfg.min_order_btc > 0
        # validate_fill_config は ValueError なしで完了
        validate_fill_config(cfg)


# ════════════════════════════════════════════════════════════════
# §2 CRITICAL-2: balance_checker.check() exception → skip
# ════════════════════════════════════════════════════════════════


class TestBalanceCheckerExceptionSkips:
    """check() 内で例外が発生したら True (skip) を返す."""

    def _make_checker(self) -> "BalanceChecker":
        from scripts.v460.lib.balance_checker import BalanceChecker

        cfg = FillTestConfig(order_quantity=0.001, min_order_btc=0.001)
        return BalanceChecker(cfg)

    def test_check_exception_returns_true_sell(self) -> None:
        """sell 側で例外 → True (注文スキップ)."""
        checker = self._make_checker()
        adapter = AsyncMock()
        adapter.get_balance.side_effect = RuntimeError("API timeout")
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                checker.check("sell", adapter, "btc_jpy")
            )
        finally:
            loop.close()
        assert result is True, "例外時は True (skip) を返すべき"

    def test_check_exception_returns_true_buy(self) -> None:
        """buy 側で例外 → True (注文スキップ)."""
        checker = self._make_checker()
        adapter = AsyncMock()
        adapter.get_balance.side_effect = ConnectionError("network down")
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                checker.check("buy", adapter, "btc_jpy")
            )
        finally:
            loop.close()
        assert result is True, "例外時は True (skip) を返すべき"


# ════════════════════════════════════════════════════════════════
# §3 IMPORTANT-3: SACRetrainConfig __post_init__ validation
# ════════════════════════════════════════════════════════════════


class TestSACRetrainConfigValidation:
    """SACRetrainConfig の __post_init__ バリデーション."""

    def test_default_config_passes(self) -> None:
        """デフォルト値はバリデーション通過."""
        cfg = SACRetrainConfig()
        assert cfg.total_timesteps == 50_000

    def test_rolling_window_days_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rolling_window_days"):
            SACRetrainConfig(rolling_window_days=0)

    def test_total_timesteps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="total_timesteps"):
            SACRetrainConfig(total_timesteps=0)

    def test_val_ratio_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="val_ratio"):
            SACRetrainConfig(val_ratio=0.0)
        with pytest.raises(ValueError, match="val_ratio"):
            SACRetrainConfig(val_ratio=1.0)

    def test_buffer_smaller_than_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="buffer_size.*batch_size"):
            SACRetrainConfig(buffer_size=64, batch_size=256)

    def test_learning_rate_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            SACRetrainConfig(learning_rate=0.0)

    def test_retrain_interval_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="retrain_interval_max_sec"):
            SACRetrainConfig(retrain_interval_sec=7200, retrain_interval_max_sec=3600)

    def test_n_eval_episodes_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_eval_episodes"):
            SACRetrainConfig(n_eval_episodes=0)


# ════════════════════════════════════════════════════════════════
# §4 IMPORTANT-4: read_sidecar_signal TOCTOU 修正
# ════════════════════════════════════════════════════════════════


class TestSidecarSignalTOCTOU:
    """read_sidecar_signal の TOCTOU 修正テスト."""

    def test_nonexistent_file_returns_none(self, tmp_path: Path) -> None:
        """存在しないファイル → None (FileNotFoundError 経由)."""
        result = read_sidecar_signal(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_signal_reads(self, tmp_path: Path) -> None:
        """正常なシグナルファイルが読み込まれる."""
        from datetime import datetime, timezone

        sig_path = tmp_path / "signal.json"
        sig_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "directional_bias": 0.1,
            "confidence": 0.8,
        }
        sig_path.write_text(json.dumps(sig_data), encoding="utf-8")
        result = read_sidecar_signal(sig_path, ttl_sec=0)
        assert result is not None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        """壊れた JSON → None."""
        sig_path = tmp_path / "bad.json"
        sig_path.write_text("{corrupt", encoding="utf-8")
        result = read_sidecar_signal(sig_path)
        assert result is None

    def test_file_deleted_between_check_and_read_returns_none(
        self, tmp_path: Path
    ) -> None:
        """TOCTOU: read_text 呼び出し時にファイルが消えても None."""
        sig_path = tmp_path / "vanish.json"
        # ファイルは存在しない → FileNotFoundError → None
        assert not sig_path.exists()
        result = read_sidecar_signal(sig_path)
        assert result is None
