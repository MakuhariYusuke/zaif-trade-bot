"""145# §13 境界値ガードテスト (144# §10 / 145# §11 指摘対応).

対象:
  - §11-#1 HIGH: empty valid_index で _compute_regime_sample_weights crash 防止
  - §11-#2 MEDIUM: regime_current_lookback=0 での IndexError 防止
  - §11-#3 MEDIUM: get_effective_interval busy-loop (interval=0) 防止
  - §10.2-#2 / §11-#7: bitflyer adapter 重複 docstring 除去
  - §10.1-#1 / §11-#6: CoincheckAdapter 継承非対称 (検証のみ)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# §11-#1: empty valid_index crash guard
# ======================================================================


class TestEmptyValidIndexNocrash:
    """_compute_regime_sample_weights が空 valid_index で落ちないこと."""

    def test_empty_valid_index_returns_uniform(self) -> None:
        """len(valid_index)==0 → 空配列 + reason=empty_valid_index."""
        from scripts.v460.ml.retrain_scheduler import _compute_regime_sample_weights

        df = pd.DataFrame({"regime": ["trending", "high_vol", "ranging"]})
        empty_idx = pd.Index([], dtype="int64")
        cfg = {
            "regime_sample_weights": {"trending": 1.5, "high_vol": 0.8},
            "regime_current_boost": 2.0,
            "regime_current_lookback": 3,
            "regime_weight_floor": 0.1,
        }

        weights, stats = _compute_regime_sample_weights(df, empty_idx, cfg)
        assert len(weights) == 0
        assert stats["regime_weighting"] == "uniform"
        assert stats["reason"] == "empty_valid_index"

    def test_empty_valid_index_no_numpy_error(self) -> None:
        """空配列で np.min/np.max/np.mean が呼ばれない."""
        from scripts.v460.ml.retrain_scheduler import _compute_regime_sample_weights

        df = pd.DataFrame({"regime": []})
        empty_idx = pd.Index([], dtype="int64")
        cfg: dict = {}
        # ValueError が出ないことを確認
        weights, stats = _compute_regime_sample_weights(df, empty_idx, cfg)
        assert isinstance(weights, np.ndarray)
        assert stats["reason"] == "empty_valid_index"

    def test_single_sample_valid_index(self) -> None:
        """1 件だけの valid_index でも正常動作."""
        from scripts.v460.ml.retrain_scheduler import _compute_regime_sample_weights

        df = pd.DataFrame({"regime": ["trending"]}, index=[0])
        idx = pd.Index([0])
        cfg = {
            "regime_sample_weights": {"trending": 2.0},
            "regime_current_boost": 1.5,
            "regime_current_lookback": 1,
            "regime_weight_floor": 0.1,
        }
        weights, stats = _compute_regime_sample_weights(df, idx, cfg)
        assert len(weights) == 1
        assert stats["regime_weighting"] == "applied"


# ======================================================================
# §11-#2: lookback=0 IndexError guard
# ======================================================================


class TestLookbackZeroGuard:
    """regime_current_lookback=0 設定時の IndexError 防止."""

    def test_lookback_zero_in_compute_weights(self) -> None:
        """lookback=0 → max(1,...) により lookback=1 として動作."""
        from scripts.v460.ml.retrain_scheduler import _compute_regime_sample_weights

        df = pd.DataFrame({"regime": ["trending", "ranging", "high_vol"]})
        idx = df.index
        cfg = {
            "regime_current_lookback": 0,
            "regime_current_boost": 1.5,
            "regime_weight_floor": 0.1,
        }
        weights, stats = _compute_regime_sample_weights(df, idx, cfg)
        assert len(weights) == 3
        assert stats["regime_weighting"] == "applied"

    def test_lookback_zero_does_not_index_error(self) -> None:
        """lookback=0 + empty X_valid でも IndexError にならない."""
        from scripts.v460.ml.retrain_scheduler import _compute_regime_sample_weights

        df = pd.DataFrame({"regime": ["x"]}, index=[0])
        # lookback=0 → max(1, 0) = 1, len(regimes)=1 >= 1 → regime detection 実行
        cfg = {"regime_current_lookback": 0, "regime_current_boost": 1.0}
        weights, stats = _compute_regime_sample_weights(df, pd.Index([0]), cfg)
        assert stats["current_regime"] == "x"

    def test_lookback_negative_clamped(self) -> None:
        """負の lookback もmax(1,...) でクランプ."""
        from scripts.v460.ml.retrain_scheduler import _compute_regime_sample_weights

        df = pd.DataFrame({"regime": ["trending", "ranging"]})
        idx = df.index
        cfg = {"regime_current_lookback": -5, "regime_current_boost": 1.0}
        weights, stats = _compute_regime_sample_weights(df, idx, cfg)
        assert len(weights) == 2

    def test_retrain_model_lookback_zero_no_crash(self) -> None:
        """retrain_model 内のレジーム検出も lookback=0 で安全."""
        # retrain_model のレジーム検出コード部分をソース検証
        src = Path("scripts/v460/ml/retrain_scheduler.py").read_text(encoding="utf-8")
        assert "max(1, safe_to_int(cfg.get(\"regime_current_lookback\"" in src


# ======================================================================
# §11-#3: busy-loop interval guard
# ======================================================================


class TestBusyLoopIntervalGuard:
    """get_effective_interval が 0 を返さないこと."""

    def _make_trigger(self, **kwargs):  # type: ignore[no-untyped-def]
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig
        cfg = RetrainTriggerConfig(
            check_trades_health=False,
            **kwargs,
        )
        return RetrainTrigger(results_dir=Path("/tmp/test"), config=cfg)

    def test_small_base_with_half_multiplier(self) -> None:
        """base=1, regime_mul=0.5 → interval=max(1, int(0.5))=max(1,0)=1."""
        trigger = self._make_trigger(
            base_interval_sec=1,
            regime_interval_multipliers={"high_vol": 0.5, "unknown": 0.5},
        )
        trigger.update_regime("unknown")
        interval = trigger.get_effective_interval()
        assert interval >= 1

    def test_very_small_multiplier(self) -> None:
        """regime_mul=0.01 + base=1 → interval >= 1."""
        trigger = self._make_trigger(
            base_interval_sec=1,
            regime_interval_multipliers={"test": 0.01, "unknown": 0.01},
        )
        trigger.update_regime("test")
        interval = trigger.get_effective_interval()
        assert interval >= 1

    def test_normal_case_unchanged(self) -> None:
        """通常ケース (base=1000, mul=0.5) → 500."""
        trigger = self._make_trigger(
            base_interval_sec=1000,
            regime_interval_multipliers={"high_vol": 0.5, "unknown": 1.0},
        )
        trigger.update_regime("high_vol")
        assert trigger.get_effective_interval() == 500

    def test_config_rejects_zero_multiplier(self) -> None:
        """regime_interval_multipliers に 0 を指定すると ValueError."""
        from ztb.ml.retrain_trigger import RetrainTriggerConfig
        with pytest.raises(ValueError, match="must be > 0"):
            RetrainTriggerConfig(
                regime_interval_multipliers={"high_vol": 0.0},
            )

    def test_config_rejects_negative_multiplier(self) -> None:
        """regime_interval_multipliers に負値を指定すると ValueError."""
        from ztb.ml.retrain_trigger import RetrainTriggerConfig
        with pytest.raises(ValueError, match="must be > 0"):
            RetrainTriggerConfig(
                regime_interval_multipliers={"high_vol": -1.0},
            )

    def test_config_rejects_zero_base_interval(self) -> None:
        """base_interval_sec=0 は ValueError."""
        from ztb.ml.retrain_trigger import RetrainTriggerConfig
        with pytest.raises(ValueError, match="base_interval_sec must be >= 1"):
            RetrainTriggerConfig(base_interval_sec=0)

    def test_source_has_max_guard(self) -> None:
        """get_effective_interval 内に max(1, ...) ガードが存在."""
        src = Path("ztb/ml/retrain_trigger.py").read_text(encoding="utf-8")
        assert "max(1," in src


# ======================================================================
# §10.2-#2 / §11-#7: bitflyer duplicate docstring
# ======================================================================


class TestBitflyerDocstringCleanup:
    """bitflyer adapter の _make_request に重複 docstring がないこと."""

    def test_no_duplicate_docstring(self) -> None:
        """_make_request のソースに docstring が 1 つだけ."""
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        src = inspect.getsource(BitFlyerAdapter._make_request)
        # triple-quote の出現回数 (開始+終了で 2 回が正常)
        triple_count = src.count('"""')
        assert triple_count == 2, f"Expected 2 triple-quotes (one docstring), got {triple_count}"


# ======================================================================
# §10.1-#1 / §11-#6: CoincheckAdapter 継承検証 (移行タスク記録)
# ======================================================================


class TestCoincheckAdapterInheritance:
    """CoincheckAdapter の継承構造を検証 (145# §14 移行完了)."""

    def test_implements_ibroker(self) -> None:
        """CoincheckAdapter が IBroker を実装していること."""
        from ztb.trading.live.exchanges.base.broker_interfaces import IBroker
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
        assert issubclass(CoincheckAdapter, IBroker)

    def test_coincheck_uses_base_adapter(self) -> None:
        """CoincheckAdapter が BaseExchangeAdapter 継承であること (145# §14)."""
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
        assert issubclass(CoincheckAdapter, BaseExchangeAdapter)

    def test_bitflyer_uses_base_adapter(self) -> None:
        """BitFlyerAdapter が BaseExchangeAdapter 継承であること."""
        from ztb.trading.live.exchanges.base.adapter import BaseExchangeAdapter
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter
        assert issubclass(BitFlyerAdapter, BaseExchangeAdapter)

    def test_shared_interface_methods(self) -> None:
        """両 adapter が IBroker の主要メソッドを実装していること."""
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
        from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter

        required_methods = [
            "place_order", "cancel_order", "get_order_status",
            "get_open_orders", "get_balance", "get_current_price",
        ]
        for method in required_methods:
            assert hasattr(CoincheckAdapter, method), f"CoincheckAdapter missing {method}"
            assert hasattr(BitFlyerAdapter, method), f"BitFlyerAdapter missing {method}"
