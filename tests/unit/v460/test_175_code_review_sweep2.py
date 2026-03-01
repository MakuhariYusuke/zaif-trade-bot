"""175# Code Review Sweep #2 — 修正検証テスト.

対象修正:
  - HIGH: FFD boost max_offset_ratio clamp (maker_price.py)
  - HIGH: buy_dynamic_kill_window validation (fill_config.py)
  - MED:  _inject_calibrator 二重呼び出し除去 (skip_gate_evaluator.py)
  - MED:  heartbeat_task cleanup (fill_loop_orchestrator.py)
  - MED:  sell_offset_floor_inv_discount YAML binding (fill_config.py)
  - MED:  stale side hot-reload fields (config_hot_reload.py)
  - LOW:  dead code n==0 除去 (maker_price.py)
  - LOW:  skip counter conflation 修正 (fill_record_helpers.py)
  - LOW:  FFD boost TTL decay (fast_fill_defense.py)
  - MED:  order_monitor object→Protocol 型化
"""

from __future__ import annotations

import pytest


# ======================================================================
# 1. FFD boost max_offset_ratio clamp
# ======================================================================

class TestFFDBoostClamp:
    """FFD boost 後に effective_offset_ratio が max_offset_ratio を超えない."""

    def test_boost_clamped_to_max(self) -> None:
        """VG boost + FFD boost が重なっても max_offset_ratio でクランプ."""
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        import inspect
        source = inspect.getsource(MakerPriceCalculator.compute)
        # FFD boost セクションが helper 経由で max_offset_ratio clamp を使うことを確認
        assert "boost_mult" in source
        assert "_scale_offset_ratio" in source
        boost_idx = source.find("boost_mult")
        helper_after_boost = source.find("_scale_offset_ratio", boost_idx)
        max_after_helper = source.find("max_ratio=cfg.max_offset_ratio", helper_after_boost)
        assert max_after_helper > 0, "FFD boost 後に helper clamp が必要"


# ======================================================================
# 2. buy_dynamic_kill_window validation
# ======================================================================

class TestBuyDynamicKillWindowValidation:
    """buy_dynamic_kill_window のバリデーション."""

    def test_zero_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="buy_dynamic_kill_window"):
            FillTestConfig(buy_dynamic_kill_window=0)

    def test_negative_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="buy_dynamic_kill_window"):
            FillTestConfig(buy_dynamic_kill_window=-1)

    def test_valid_default(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.buy_dynamic_kill_window >= 1


# ======================================================================
# 3. calibrator 二重呼び出し除去
# ======================================================================

class TestCalibratorNotDoubled:
    """_check_and_reload_model で _inject_calibrator が二重呼び出しされない."""

    def test_no_double_inject(self) -> None:
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
        import inspect
        source = inspect.getsource(SkipGateEvaluator._check_and_reload_model)
        # コメント以外の行で _inject_calibrator が呼ばれていないこと
        code_lines = [
            line.strip()
            for line in source.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        inject_calls = [
            line for line in code_lines
            if "_inject_calibrator" in line
            and not line.startswith("#")
            and not line.startswith("\"")
        ]
        assert len(inject_calls) == 0, f"_inject_calibrator は _load_gate_from_path 内で呼ばれるので二重呼び出し不要: {inject_calls}"


# ======================================================================
# 4. sell_offset_floor_inv_discount YAML binding
# ======================================================================

class TestYAMLInvDiscountBinding:
    """sell_offset_floor_inv_discount が YAML からパース可能."""

    def test_parse_from_yaml(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = {
            "sell_guard": {
                "offset_floor_inv_discount": 0.3,
            },
        }
        kwargs = FillTestConfig._parse_stale_vg_section(yaml_cfg)
        assert kwargs.get("sell_offset_floor_inv_discount") == 0.3

    def test_parse_none_skipped(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = {"sell_guard": {}}
        kwargs = FillTestConfig._parse_stale_vg_section(yaml_cfg)
        assert "sell_offset_floor_inv_discount" not in kwargs


# ======================================================================
# 5. stale side hot-reload fields
# ======================================================================

class TestStaleSideHotReload:
    """stale side 別フィールドが hot-reload 対象."""

    @pytest.mark.parametrize("field", [
        "stale_check_after_sec_buy",
        "stale_check_after_sec_sell",
        "stale_drift_bps_buy",
        "stale_drift_bps_sell",
        "stale_max_reprice_buy",
        "stale_max_reprice_sell",
    ])
    def test_stale_side_field_reloadable(self, field: str) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert field in _HOT_RELOADABLE_FIELDS


# ======================================================================
# 6. dead code n==0 除去
# ======================================================================

class TestDeadCodeRemoved:
    """update_inventory の n==0 dead code が除去されている."""

    def test_no_n_equals_zero_check(self) -> None:
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        import inspect
        source = inspect.getsource(MakerPriceCalculator.update_inventory)
        assert "n == 0" not in source


# ======================================================================
# 7. skip counter conflation 修正
# ======================================================================

class TestSkipCounterSeparation:
    """trending_sell_skip と balance_forced_skip のカウンタが独立."""

    def test_interleaved_records_count_correctly(self) -> None:
        """交互に来た場合、各カウンタは末尾の連続数のみ."""
        from unittest.mock import MagicMock, patch
        from scripts.v460.lib.fill_record_helpers import FillRecordHelpersMixin

        class FakeRunner(FillRecordHelpersMixin):
            def __init__(self) -> None:
                self._cycle_count = 0
                self._trending_sell_skip_count = 0
                self._balance_forced_skip_count = 0
                self._results_dir = "/tmp/test"
                self._side_selector = MagicMock()
                self._side_selector.last_side = "buy"
                self._maker_price = MagicMock()
                self._maker_price._last_imbalance = 0.0

        runner = FakeRunner()

        # 交互の skip records をシミュレート
        mock_records = []
        for cr in ["trending_sell_skip", "balance_forced_skip",
                    "trending_sell_skip", "balance_forced_skip"]:
            rec = MagicMock()
            rec.cancel_reason = cr
            rec.side = "sell"
            mock_records.append(rec)

        with patch("ztb.metrics.fill_quality.iter_fill_records_glob",
                    return_value=iter(mock_records)):
            runner.resume_from_existing()

        # 末尾は balance_forced_skip 1件のみ連続
        assert runner._balance_forced_skip_count == 1
        # trending_sell_skip は末尾0 (最後の rec が bfs なので tss は 0)
        assert runner._trending_sell_skip_count == 0

    def test_consecutive_same_type(self) -> None:
        """同一タイプの連続カウントは正しい."""
        from unittest.mock import MagicMock, patch
        from scripts.v460.lib.fill_record_helpers import FillRecordHelpersMixin

        class FakeRunner(FillRecordHelpersMixin):
            def __init__(self) -> None:
                self._cycle_count = 0
                self._trending_sell_skip_count = 0
                self._balance_forced_skip_count = 0
                self._results_dir = "/tmp/test"
                self._side_selector = MagicMock()
                self._side_selector.last_side = "buy"
                self._maker_price = MagicMock()
                self._maker_price._last_imbalance = 0.0

        runner = FakeRunner()

        mock_records = []
        for cr in ["filled", "trending_sell_skip", "trending_sell_skip",
                    "trending_sell_skip"]:
            rec = MagicMock()
            rec.cancel_reason = cr
            rec.side = "sell"
            mock_records.append(rec)

        with patch("ztb.metrics.fill_quality.iter_fill_records_glob",
                    return_value=iter(mock_records)):
            runner.resume_from_existing()

        assert runner._trending_sell_skip_count == 3
        assert runner._balance_forced_skip_count == 0


# ======================================================================
# 8. FFD boost TTL decay
# ======================================================================

class TestFFDBoostTTL:
    """FFD boost の TTL decay 機能."""

    def test_boost_ttl_config_default(self) -> None:
        from scripts.v460.lib.fast_fill_defense import FastFillDefenseConfig
        cfg = FastFillDefenseConfig()
        assert cfg.boost_ttl_sec == 600.0

    def test_boost_ttl_field_in_config(self) -> None:
        from scripts.v460.lib.fast_fill_defense import FastFillDefenseConfig
        cfg = FastFillDefenseConfig(boost_ttl_sec=300.0)
        assert cfg.boost_ttl_sec == 300.0

    def test_boost_activated_at_tracked(self) -> None:
        """boost 活性化時に activated_at が記録される."""
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
        cfg = FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0)
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)
        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        state = defense._state_buy
        assert state.boost_active
        assert state.boost_activated_at > 0


# ======================================================================
# 9. order_monitor Protocol 型化
# ======================================================================

class TestOrderMonitorProtocols:
    """order_monitor の引数型が Protocol 化されている."""

    def test_monitor_signature_typed(self) -> None:
        from scripts.v460.lib.order_monitor import OrderMonitor
        import inspect
        sig = inspect.signature(OrderMonitor.monitor)
        params = sig.parameters
        # object ではなく型付きであること
        shutdown_ann = str(params["shutdown_check"].annotation)
        assert "object" not in shutdown_ann
        assert "KillSwitch" in shutdown_ann

    def test_pending_order_setter_callable(self) -> None:
        from scripts.v460.lib.order_monitor import OrderMonitor
        import inspect
        sig = inspect.signature(OrderMonitor.monitor)
        ann = str(sig.parameters["pending_order_setter"].annotation)
        assert "Callable" in ann

    def test_get_mid_price_awaitable(self) -> None:
        from scripts.v460.lib.order_monitor import OrderMonitor
        import inspect
        sig = inspect.signature(OrderMonitor.monitor)
        ann = str(sig.parameters["get_mid_price"].annotation)
        assert "Callable" in ann or "Awaitable" in ann


# ======================================================================
# 10. heartbeat cleanup method exists
# ======================================================================

class TestHeartbeatCleanup:
    """cleanup_heartbeat メソッドが存在すること."""

    def test_cleanup_method_exists(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "cleanup_heartbeat")
        import inspect
        assert inspect.iscoroutinefunction(FillLoopOrchestratorMixin.cleanup_heartbeat)
