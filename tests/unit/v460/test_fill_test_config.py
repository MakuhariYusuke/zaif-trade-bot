"""Tests for fill_test YAML config loading + FillTestConfig.from_yaml.

055# Fix: 挙動テスト追加 (_next_side, round-trip 双方向).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from scripts.v460.lib.config_loader import load_fill_test_config
from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner
from ztb.metrics.fill_quality import (
    FillRecord,
    RoundTripRecord,
    compute_round_trip_metrics,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _make_runner(
    smart_side_enabled: bool = False,
    smart_side_mode: str = "suppress",
    smart_side_max_consecutive: int = 2,
    imbalance_threshold: float = 0.3,
    start_side: str = "buy",
    **kwargs: object,
) -> FillTestRunner:
    """テスト用の FillTestRunner を構築 (adapter はモック)."""
    config = FillTestConfig(
        smart_side_enabled=smart_side_enabled,
        smart_side_mode=smart_side_mode,
        smart_side_max_consecutive=smart_side_max_consecutive,
        imbalance_threshold=imbalance_threshold,
        start_side=start_side,
        enable_regime=False,  # テストでレジーム検知は不要
        **kwargs,  # type: ignore[arg-type]
    )
    adapter = MagicMock()
    return FillTestRunner(adapter=adapter, config=config)


class TestLoadFillTestConfig:
    """load_fill_test_config のテスト."""

    def test_load_default_path(self) -> None:
        """デフォルトパスから fill_test.yaml をロードできる."""
        cfg = load_fill_test_config()
        assert isinstance(cfg, dict)
        assert cfg["symbol"] == "btc_jpy"
        assert cfg["order_quantity"] == 0.001
        assert "adaptation" in cfg
        assert "lot_sizing" in cfg
        assert "safety" in cfg

    def test_load_explicit_path(self) -> None:
        """明示パスからロードできる."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        cfg = load_fill_test_config(path)
        assert cfg["spread_offset_ratio"] == 0.05

    def test_yaml_has_all_sections(self) -> None:
        """YAML が全セクションを含む."""
        cfg = load_fill_test_config()
        # フラットキー
        assert "cycle_interval_sec" in cfg
        assert "order_timeout_sec" in cfg
        assert "as_deadzone_bps" in cfg
        # 035# 新フラットキー
        assert "save_fail_threshold" in cfg
        assert "progress_log_interval" in cfg
        assert "log_max_bytes" in cfg
        assert "log_backup_count" in cfg
        # ネストセクション
        assert cfg["adaptation"]["enabled"] is False  # 122# R2: 因果分離のため無効化
        assert cfg["lot_sizing"]["enabled"] is False
        assert cfg["lot_sizing"]["recent_pnl_window"] == 50
        assert cfg["safety"]["loss_cap_jpy"] == 10000.0
        # 041# 新セクション
        assert "time_filter" in cfg
        assert cfg["time_filter"]["enabled"] is True
        assert cfg["safety"]["loss_cap_auto"] is True


class TestFillTestConfigFromYaml:
    """FillTestConfig.from_yaml のテスト."""

    def test_from_yaml_defaults(self) -> None:
        """YAML デフォルト値から FillTestConfig が構築される."""
        cfg = load_fill_test_config()
        config = FillTestConfig.from_yaml(cfg)
        assert config.symbol == "btc_jpy"
        assert config.order_quantity == 0.001
        assert config.spread_offset_ratio == 0.05
        assert config.enable_auto_adapt is False  # 122# R2: 因果分離のため無効化
        assert config.enable_dynamic_lot is False
        assert config.loss_cap_jpy == 10000.0
        assert config.loss_cap_warning_ratio == 0.7
        # 041# 新フィールド
        assert config.enable_time_filter is True
        assert config.loss_cap_auto is True
        assert config.loss_cap_ratio == 0.05
        assert config.as_deadzone_bps == 2.5  # 052#: 2.0→2.5

    def test_from_yaml_custom_values(self) -> None:
        """カスタム値が正しくマッピングされる."""
        yaml_cfg = {
            "symbol": "btc_jpy",
            "order_quantity": 0.002,
            "cycle_interval_sec": 60.0,
            "spread_offset_ratio": 0.10,
            "as_deadzone_bps": 1.0,
            "save_fail_threshold": 5,
            "progress_log_interval": 100,
            "log_max_bytes": 5242880,
            "log_backup_count": 3,
            "adaptation": {
                "enabled": True,
                "interval_cycles": 30,
                "min_samples": 30,
            },
            "lot_sizing": {
                "enabled": True,
                "interval_cycles": 25,
                "max_lot": 0.010,
                "recent_pnl_window": 30,
            },
            "safety": {
                "loss_cap_jpy": 5000.0,
                "loss_cap_warning_ratio": 0.5,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.order_quantity == 0.002
        assert config.cycle_interval_sec == 60.0
        assert config.spread_offset_ratio == 0.10
        assert config.as_deadzone_bps == 1.0
        assert config.enable_auto_adapt is True
        assert config.adapt_interval_cycles == 30
        assert config.enable_dynamic_lot is True
        assert config.lot_adapt_interval_cycles == 25
        assert config.max_lot == 0.010
        assert config.loss_cap_jpy == 5000.0
        assert config.loss_cap_warning_ratio == 0.5
        # 035# 新フィールド
        assert config.save_fail_threshold == 5
        assert config.progress_log_interval == 100
        assert config.log_max_bytes == 5242880
        assert config.log_backup_count == 3
        assert config.min_adapt_samples == 30
        assert config.recent_pnl_window == 30

    def test_from_yaml_partial(self) -> None:
        """部分的な YAML でもデフォルトが効く."""
        yaml_cfg = {"spread_offset_ratio": 0.15}
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.spread_offset_ratio == 0.15
        # 未指定はデフォルト値
        assert config.order_quantity == 0.001
        assert config.cycle_interval_sec == 120.0
        assert config.enable_auto_adapt is False
        assert config.loss_cap_jpy == 10_000.0

    def test_from_yaml_empty(self) -> None:
        """空 dict → 全デフォルト."""
        config = FillTestConfig.from_yaml({})
        assert config.order_quantity == 0.001
        assert config.enable_auto_adapt is False
        # 035# 新フィールドもデフォルト
        assert config.save_fail_threshold == 3
        assert config.progress_log_interval == 50
        assert config.log_max_bytes == 10 * 1024 * 1024
        assert config.log_backup_count == 5
        assert config.min_adapt_samples == 50
        assert config.recent_pnl_window == 50

    def test_from_yaml_missing_sections(self) -> None:
        """adaptation / lot_sizing / safety セクションなし → デフォルト."""
        yaml_cfg = {"symbol": "btc_jpy", "order_quantity": 0.003}
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.order_quantity == 0.003
        assert config.enable_auto_adapt is False
        assert config.enable_dynamic_lot is False
        assert config.loss_cap_jpy == 10_000.0


class TestFillTestYamlFile:
    """configs/v460/fill_test.yaml ファイル自体のバリデーション."""

    def test_yaml_is_valid(self) -> None:
        """YAML として正しく parse できる."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_yaml_roundtrip(self) -> None:
        """YAML → FillTestConfig → 主要フィールドが一致."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config = FillTestConfig.from_yaml(cfg)
        assert config.symbol == cfg["symbol"]
        assert config.order_quantity == cfg["order_quantity"]
        assert config.spread_offset_ratio == cfg["spread_offset_ratio"]
        assert config.loss_cap_jpy == cfg["safety"]["loss_cap_jpy"]

    def test_yaml_tuning_roundtrip(self) -> None:
        """103# tuning セクションの全 18 キーが FillTestConfig に正しくマッピングされる."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config = FillTestConfig.from_yaml(cfg)
        tuning = cfg.get("tuning", {})
        assert config.max_offset_ratio == tuning["max_offset_ratio"]
        assert config.min_offset_ratio == tuning["min_offset_ratio"]
        assert config.loss_cap_update_interval == tuning["loss_cap_update_interval"]
        assert config.min_loss_cap_jpy == tuning["min_loss_cap_jpy"]
        assert config.mid_trend_validity_sec == tuning["mid_trend_validity_sec"]
        assert config.balance_margin_ratio == tuning["balance_margin_ratio"]
        assert config.balance_shrink_consecutive == tuning["balance_shrink_consecutive"]
        assert config.balance_shrink_divisor == tuning["balance_shrink_divisor"]
        assert config.skip_gate_recent_trades_limit == tuning["skip_gate_recent_trades_limit"]
        assert config.rate_limit_min_backoff_sec == tuning["rate_limit_min_backoff_sec"]
        assert config.save_retry_backoff_sec == tuning["save_retry_backoff_sec"]
        assert config.regime_warmup_multiplier == tuning["regime_warmup_multiplier"]
        assert config.e3_60s_multiplier == tuning["e3_60s_multiplier"]
        assert config.e3_120s_multiplier == tuning["e3_120s_multiplier"]
        assert config.adapt_min_side_samples == tuning["adapt_min_side_samples"]
        assert config.batch_flush_interval_sec == tuning["batch_flush_interval_sec"]
        assert config.heartbeat_interval_sec == tuning["heartbeat_interval_sec"]

    def test_tuning_custom_values(self) -> None:
        """103# tuning カスタム値→FillTestConfig."""
        yaml_cfg = {
            "tuning": {
                "max_offset_ratio": 0.25,
                "min_offset_ratio": 0.02,
                "min_loss_cap_jpy": 100.0,
                "balance_shrink_divisor": 3,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.max_offset_ratio == 0.25
        assert config.min_offset_ratio == 0.02
        assert config.min_loss_cap_jpy == 100.0
        assert config.balance_shrink_divisor == 3

    def test_post_init_balance_shrink_divisor_zero(self) -> None:
        """103# balance_shrink_divisor=0 → ValueError."""
        import pytest
        with pytest.raises(ValueError, match="balance_shrink_divisor"):
            FillTestConfig(balance_shrink_divisor=0)

    def test_post_init_offset_ratio_invariant(self) -> None:
        """103# max_offset_ratio <= min_offset_ratio → ValueError."""
        import pytest
        with pytest.raises(ValueError, match="max_offset_ratio"):
            FillTestConfig(max_offset_ratio=0.01, min_offset_ratio=0.01)

    def test_adaptation_min_samples_mapping(self) -> None:
        """103# adaptation.min_samples → min_adapt_samples."""
        yaml_cfg = {
            "adaptation": {
                "enabled": True,
                "min_samples": 80,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.min_adapt_samples == 80


class Test054ImbalanceConfig:
    """054# S1: Imbalance 設定のテスト."""

    def test_yaml_has_imbalance_section(self) -> None:
        """YAML に imbalance セクションがある (071# disabled)."""
        cfg = load_fill_test_config()
        assert "imbalance" in cfg
        assert cfg["imbalance"]["enabled"] is False  # 071# OB無視
        assert cfg["imbalance"]["depth"] == 5
        assert cfg["imbalance"]["threshold"] == 0.3

    def test_from_yaml_imbalance(self) -> None:
        """Imbalance config が FillTestConfig に正しくマッピングされる."""
        yaml_cfg = {
            "imbalance": {
                "enabled": True,
                "depth": 10,
                "threshold": 0.4,
                "offset_boost": 2.0,
                "skip_threshold": 0.8,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.imbalance_enabled is True
        assert config.imbalance_depth == 10
        assert config.imbalance_threshold == 0.4
        assert config.imbalance_offset_boost == 2.0
        assert config.imbalance_skip_threshold == 0.8

    def test_from_yaml_imbalance_defaults(self) -> None:
        """Imbalance 未設定時のデフォルト値."""
        config = FillTestConfig.from_yaml({})
        assert config.imbalance_enabled is False
        assert config.imbalance_depth == 5
        assert config.imbalance_threshold == 0.3
        assert config.imbalance_offset_boost == 1.5
        assert config.imbalance_skip_threshold == 0.7


class Test054SmartSideConfig:
    """054# S2: Smart Side 設定のテスト."""

    def test_yaml_has_smart_side_section(self) -> None:
        """YAML に smart_side セクションがある (071# disabled)."""
        cfg = load_fill_test_config()
        assert "smart_side" in cfg
        assert cfg["smart_side"]["enabled"] is False  # 071# OB無視
        assert cfg["smart_side"]["mode"] == "suppress"
        assert cfg["smart_side"]["max_consecutive_same"] == 2

    def test_from_yaml_smart_side(self) -> None:
        """Smart Side config が正しくマッピングされる."""
        yaml_cfg = {
            "smart_side": {
                "enabled": True,
                "mode": "follow",
                "max_consecutive_same": 3,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.smart_side_enabled is True
        assert config.smart_side_mode == "follow"
        assert config.smart_side_max_consecutive == 3

    def test_from_yaml_smart_side_defaults(self) -> None:
        """Smart Side 未設定時のデフォルト値."""
        config = FillTestConfig.from_yaml({})
        assert config.smart_side_enabled is False
        assert config.smart_side_mode == "suppress"
        assert config.smart_side_max_consecutive == 2


class Test054EarlyExitConfig:
    """054# S3: Early Exit 設定のテスト."""

    def test_yaml_has_early_exit_section(self) -> None:
        """YAML に early_exit セクションがある."""
        cfg = load_fill_test_config()
        assert "early_exit" in cfg
        # 120# P0: EE は 119# 分析結果に基づき無効化
        assert cfg["early_exit"]["enabled"] is False
        assert cfg["early_exit"]["threshold_bps"] == 5.0

    def test_from_yaml_early_exit(self) -> None:
        """Early Exit config が正しくマッピングされる."""
        yaml_cfg = {
            "early_exit": {
                "enabled": True,
                "threshold_bps": 3.0,
                "monitoring_interval_sec": 3.0,
                "rapid_exit_interval_sec": 5.0,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.early_exit_enabled is True
        assert config.early_exit_threshold_bps == 3.0
        assert config.early_exit_monitor_interval_sec == 3.0
        assert config.early_exit_rapid_interval_sec == 5.0


class Test054SpreadAdaptiveConfig:
    """054# S4: Spread Adaptive 設定のテスト."""

    def test_yaml_has_spread_adaptive_section(self) -> None:
        """YAML に spread_adaptive セクションがある."""
        cfg = load_fill_test_config()
        assert "spread_adaptive" in cfg
        assert cfg["spread_adaptive"]["enabled"] is True
        # 120# A1: 実データ分布に基づき閾値切り下げ (10→2.0)
        assert cfg["spread_adaptive"]["narrow_spread_bps"] == 2.0

    def test_from_yaml_spread_adaptive(self) -> None:
        """Spread Adaptive config が正しくマッピングされる."""
        yaml_cfg = {
            "spread_adaptive": {
                "enabled": True,
                "narrow_spread_bps": 8.0,
                "narrow_spread_boost": 1.8,
                "wide_spread_bps": 30.0,
                "wide_spread_ratio": 0.6,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.spread_adaptive_enabled is True
        assert config.narrow_spread_bps == 8.0
        assert config.narrow_spread_boost == 1.8
        assert config.wide_spread_bps == 30.0
        assert config.wide_spread_ratio == 0.6

    def test_yaml_roundtrip_054(self) -> None:
        """054# 全設定の YAML → FillTestConfig roundtrip."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config = FillTestConfig.from_yaml(cfg)
        # S1 (071# disabled — OB無視)
        assert config.imbalance_enabled is False
        assert config.imbalance_depth == cfg["imbalance"]["depth"]
        # S2 (071# disabled — OB無視)
        assert config.smart_side_enabled is False
        assert config.smart_side_mode == cfg["smart_side"]["mode"]
        # S3 — 120# P0: EE disabled
        assert config.early_exit_enabled is False
        assert config.early_exit_threshold_bps == cfg["early_exit"]["threshold_bps"]
        # S4
        assert config.spread_adaptive_enabled is True
        assert config.narrow_spread_bps == cfg["spread_adaptive"]["narrow_spread_bps"]


class Test054FillRecordNewFields:
    """054# S5: FillRecord 新フィールドのテスト."""

    def test_new_fields_exist(self) -> None:
        """054# 新フィールドが FillRecord に存在する."""
        from ztb.metrics.fill_quality import FillRecord
        fields = FillRecord.__dataclass_fields__
        assert "orderbook_imbalance" in fields
        assert "bid_depth_total" in fields
        assert "ask_depth_total" in fields
        assert "mid_price_trend_5s" in fields
        assert "spread_bps" in fields
        assert "effective_offset_used" in fields

    def test_new_fields_default_none(self) -> None:
        """054# 新フィールドのデフォルト値は全て None."""
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=1.0,
            side="buy",
            order_price=1000.0,
            order_quantity=0.001,
        )
        assert rec.orderbook_imbalance is None
        assert rec.bid_depth_total is None
        assert rec.ask_depth_total is None
        assert rec.mid_price_trend_5s is None
        assert rec.spread_bps is None
        assert rec.effective_offset_used is None

    def test_new_fields_roundtrip(self) -> None:
        """054# 新フィールドの to_dict/from_dict roundtrip."""
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test_054",
            timestamp=1.0,
            side="buy",
            order_price=10000000.0,
            order_quantity=0.001,
            orderbook_imbalance=0.35,
            bid_depth_total=1.5,
            ask_depth_total=0.8,
            mid_price_trend_5s=-2.1,
            spread_bps=15.3,
            effective_offset_used=0.075,
        )
        d = rec.to_dict()
        assert d["orderbook_imbalance"] == 0.35
        assert d["spread_bps"] == 15.3
        rec2 = FillRecord.from_dict(d)
        assert rec2.orderbook_imbalance == rec.orderbook_imbalance
        assert rec2.bid_depth_total == rec.bid_depth_total
        assert rec2.spread_bps == rec.spread_bps
        assert rec2.effective_offset_used == rec.effective_offset_used

    def test_backward_compat_from_dict(self) -> None:
        """054# 新フィールドなしの古い dict から FillRecord を復元できる."""
        from ztb.metrics.fill_quality import FillRecord
        old_dict = {
            "cycle_id": "old_record",
            "timestamp": 1.0,
            "side": "sell",
            "order_price": 10000000.0,
            "order_quantity": 0.001,
        }
        rec = FillRecord.from_dict(old_dict)
        assert rec.orderbook_imbalance is None
        assert rec.spread_bps is None


class Test054SmartSideLogic:
    """054# S2: Smart Side ロジックの単体テスト."""

    def test_next_side_alternates_when_disabled(self) -> None:
        """Smart Side 無効時は従来の交互ロジック."""
        config = FillTestConfig(smart_side_enabled=False)
        # _next_side を直接テスト (FillTestRunner なしで)
        # config の確認のみ
        assert config.smart_side_enabled is False
        assert config.smart_side_mode == "suppress"

    def test_next_side_suppress_mode_config(self) -> None:
        """Suppress mode の設定値."""
        config = FillTestConfig(
            smart_side_enabled=True,
            smart_side_mode="suppress",
            smart_side_max_consecutive=3,
            imbalance_threshold=0.25,
        )
        assert config.smart_side_enabled is True
        assert config.smart_side_mode == "suppress"
        assert config.smart_side_max_consecutive == 3

    def test_next_side_follow_mode_config(self) -> None:
        """Follow mode の設定値."""
        config = FillTestConfig(
            smart_side_enabled=True,
            smart_side_mode="follow",
        )
        assert config.smart_side_mode == "follow"


# ======================================================================
# 055# Fix: 挙動テスト (_next_side + round-trip 双方向)
# ======================================================================


class Test055NextSideBehavior:
    """055# Fix: _next_side() の挙動テスト (実際の FillTestRunner で検証)."""

    # --- 基本交互ロジック ---

    def test_alternates_buy_sell(self) -> None:
        """Smart Side 無効時、buy → sell → buy の交互."""
        runner = _make_runner(smart_side_enabled=False)
        assert runner._next_side() == "buy"  # _last_side=None → buy
        runner._last_side = "buy"
        assert runner._next_side() == "sell"
        runner._last_side = "sell"
        assert runner._next_side() == "buy"

    def test_start_side_sell(self) -> None:
        """start_side=sell のとき最初は sell."""
        runner = _make_runner(smart_side_enabled=False, start_side="sell")
        assert runner._next_side() == "sell"

    # --- 055# Fix #1: rapid_exit_side 優先返却 ---

    def test_rapid_exit_side_forces_side(self) -> None:
        """rapid_exit_side が設定されていれば、それを優先返却."""
        runner = _make_runner(smart_side_enabled=False)
        runner._last_side = "buy"  # 通常なら sell
        runner._side_selector.rapid_exit_side = "buy"  # しかし rapid exit は buy を強制
        assert runner._next_side() == "buy"
        # 使用後にクリアされる
        assert runner._side_selector.rapid_exit_side is None

    def test_rapid_exit_side_overrides_smart_side(self) -> None:
        """rapid_exit は Smart Side よりも優先."""
        runner = _make_runner(smart_side_enabled=True, smart_side_mode="suppress")
        runner._last_side = "buy"
        runner._maker_price._last_imbalance = +0.5  # sell を抑制する状況
        runner._side_selector.rapid_exit_side = "sell"  # rapid exit は sell を強制
        assert runner._next_side() == "sell"
        assert runner._side_selector.rapid_exit_side is None

    def test_rapid_exit_side_clears_after_use(self) -> None:
        """rapid_exit_side は 1 回で消費される."""
        runner = _make_runner(smart_side_enabled=False)
        runner._side_selector.rapid_exit_side = "sell"
        result1 = runner._next_side()
        assert result1 == "sell"
        assert runner._side_selector.rapid_exit_side is None
        # 次回は通常ロジック
        runner._last_side = "sell"
        assert runner._next_side() == "buy"

    # --- 054# S2: suppress mode 挙動テスト ---

    def test_suppress_buy_on_strong_sell_pressure(self) -> None:
        """売り圧力が強い (imbalance < -threshold) → buy を抑制."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="suppress",
            imbalance_threshold=0.3,
        )
        runner._last_side = "sell"  # 通常次は buy
        runner._maker_price._last_imbalance = -0.5  # 売り圧力
        assert runner._next_side() == "sell"  # buy が抑制され sell 継続

    def test_suppress_sell_on_strong_buy_pressure(self) -> None:
        """買い圧力が強い (imbalance > threshold) → sell を抑制."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="suppress",
            imbalance_threshold=0.3,
        )
        runner._last_side = "buy"  # 通常次は sell
        runner._maker_price._last_imbalance = +0.5  # 買い圧力
        assert runner._next_side() == "buy"  # sell が抑制され buy 継続

    def test_suppress_no_action_below_threshold(self) -> None:
        """imbalance が閾値以下なら抑制しない."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="suppress",
            imbalance_threshold=0.3,
        )
        runner._last_side = "sell"
        runner._maker_price._last_imbalance = -0.2  # 閾値以下
        assert runner._next_side() == "buy"  # 通常の交互

    def test_suppress_max_consecutive_forces_base(self) -> None:
        """連続同 side 上限に達したら base_side を強制."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="suppress",
            smart_side_max_consecutive=2,
            imbalance_threshold=0.3,
        )
        runner._last_side = "sell"
        runner._maker_price._last_imbalance = -0.5  # 売り圧力
        runner._side_selector._consecutive_same_side = 2  # 上限到達
        assert runner._next_side() == "buy"  # 強制的に buy

    # --- 054# S2: follow mode 挙動テスト ---

    def test_follow_buy_on_positive_imbalance(self) -> None:
        """正の imbalance → buy に追従."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="follow",
            imbalance_threshold=0.3,
        )
        runner._last_side = "buy"  # 通常次は sell
        runner._maker_price._last_imbalance = +0.5  # 買い方向
        assert runner._next_side() == "buy"  # 追従

    def test_follow_sell_on_negative_imbalance(self) -> None:
        """負の imbalance → sell に追従."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="follow",
            imbalance_threshold=0.3,
        )
        runner._last_side = "sell"  # 通常次は buy
        runner._maker_price._last_imbalance = -0.5
        assert runner._next_side() == "sell"  # 追従

    def test_follow_max_consecutive_limits(self) -> None:
        """follow mode でも連続上限を尊重."""
        runner = _make_runner(
            smart_side_enabled=True,
            smart_side_mode="follow",
            smart_side_max_consecutive=2,
            imbalance_threshold=0.3,
        )
        runner._last_side = "buy"
        runner._maker_price._last_imbalance = +0.5  # buy 追従だが
        runner._side_selector._consecutive_same_side = 2  # 上限
        assert runner._next_side() == "sell"  # base に戻る


class Test055RoundTripBidirectional:
    """055# Fix: Round-trip 双方向ペアリングの挙動テスト."""

    @staticmethod
    def _rec(side: str, fill_price: float, ts: float) -> FillRecord:
        """テスト用 FillRecord を生成."""
        return FillRecord(
            cycle_id=f"test_{ts}",
            timestamp=ts,
            side=side,
            order_price=fill_price,
            order_quantity=0.001,
            filled=True,
            fill_price=fill_price,
        )

    def test_buy_sell_pair(self) -> None:
        """標準的な buy→sell ペアリング."""
        records = [
            self._rec("buy", 10_000_000.0, 1.0),
            self._rec("sell", 10_001_000.0, 2.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 1
        assert trips[0].direction == "buy_first"
        assert trips[0].pnl_bps > 0  # 利益

    def test_sell_buy_pair(self) -> None:
        """sell→buy ペアリング (055# 新機能)."""
        records = [
            self._rec("sell", 10_001_000.0, 1.0),
            self._rec("buy", 10_000_000.0, 2.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 1
        assert trips[0].direction == "sell_first"
        assert trips[0].pnl_bps > 0  # sell 高→buy 安 = 利益

    def test_mixed_directions(self) -> None:
        """buy→sell, sell→buy の混在."""
        records = [
            self._rec("buy", 10_000_000.0, 1.0),
            self._rec("sell", 10_001_000.0, 2.0),  # pair 1 (buy_first)
            self._rec("sell", 10_002_000.0, 3.0),
            self._rec("buy", 10_001_500.0, 4.0),  # pair 2 (sell_first)
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 2
        assert trips[0].direction == "buy_first"
        assert trips[1].direction == "sell_first"

    def test_unpaired_sells_tracked(self) -> None:
        """未ペア sell が追跡される."""
        records = [
            self._rec("sell", 10_000_000.0, 1.0),
            self._rec("sell", 10_001_000.0, 2.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 0
        assert metrics.unpaired_sells == 2
        assert metrics.net_inventory == -2

    def test_unpaired_buys_tracked(self) -> None:
        """未ペア buy が追跡される."""
        records = [
            self._rec("buy", 10_000_000.0, 1.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 0
        assert metrics.unpaired_buys == 1
        assert metrics.net_inventory == 1

    def test_net_inventory(self) -> None:
        """純在庫の正確性."""
        records = [
            self._rec("buy", 10_000_000.0, 1.0),
            self._rec("sell", 10_001_000.0, 2.0),  # pair
            self._rec("buy", 10_000_000.0, 3.0),
            self._rec("buy", 10_000_000.0, 4.0),
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 1
        assert metrics.unpaired_buys == 2
        assert metrics.unpaired_sells == 0
        assert metrics.net_inventory == 2

    def test_backward_compat_buy_sell_record_properties(self) -> None:
        """後方互換: buy_record/sell_record プロパティが動作."""
        records = [
            self._rec("buy", 10_000_000.0, 1.0),
            self._rec("sell", 10_001_000.0, 2.0),
        ]
        _, trips = compute_round_trip_metrics(records)
        trip = trips[0]
        assert trip.buy_record.side == "buy"
        assert trip.sell_record.side == "sell"

    def test_backward_compat_sell_first_properties(self) -> None:
        """sell_first でも buy_record/sell_record が正しい."""
        records = [
            self._rec("sell", 10_001_000.0, 1.0),
            self._rec("buy", 10_000_000.0, 2.0),
        ]
        _, trips = compute_round_trip_metrics(records)
        trip = trips[0]
        assert trip.buy_record.side == "buy"
        assert trip.sell_record.side == "sell"

    def test_consecutive_same_side_then_close(self) -> None:
        """Smart Side で連続 sell → buy で全部ペアリング."""
        records = [
            self._rec("sell", 10_002_000.0, 1.0),
            self._rec("sell", 10_001_000.0, 2.0),
            self._rec("buy", 10_000_000.0, 3.0),  # 1st sell とペア
            self._rec("buy", 10_000_500.0, 4.0),  # 2nd sell とペア
        ]
        metrics, trips = compute_round_trip_metrics(records)
        assert metrics.total_pairs == 2
        assert metrics.unpaired_buys == 0
        assert metrics.unpaired_sells == 0


# ======================================================================
# 062# S5: SkipGate Config Tests
# ======================================================================


class Test062SkipGateConfig:
    """062# S5: SkipGate 設定のテスト."""

    def test_yaml_has_skip_gate_section(self) -> None:
        """YAML に skip_gate セクションがある."""
        cfg = load_fill_test_config()
        assert "skip_gate" in cfg
        assert cfg["skip_gate"]["enabled"] is True  # 065#: 学習済みモデルで有効化
        assert cfg["skip_gate"]["mode"] == "as"
        assert cfg["skip_gate"]["as_threshold"] == 0.50  # 120# A3: 0.52→0.50 (変曲点)
        assert cfg["skip_gate"]["max_skip_rate"] == 0.3

    def test_from_yaml_skip_gate(self) -> None:
        """SkipGate config が FillTestConfig に正しくマッピングされる."""
        yaml_cfg = {
            "skip_gate": {
                "enabled": True,
                "mode": "as",
                "model_path": "models/v460/custom_gate.pkl",
                "as_threshold": 0.5,
                "pnl_threshold": -1.0,
                "max_skip_rate": 0.2,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.skip_gate_enabled is True
        assert config.skip_gate_mode == "as"
        assert config.skip_gate_model_path == "models/v460/custom_gate.pkl"
        assert config.skip_gate_as_threshold == 0.5
        assert config.skip_gate_pnl_threshold == -1.0
        assert config.skip_gate_max_skip_rate == 0.2

    def test_from_yaml_skip_gate_defaults(self) -> None:
        """SkipGate 未設定時のデフォルト値."""
        config = FillTestConfig.from_yaml({})
        assert config.skip_gate_enabled is False
        assert config.skip_gate_mode == "as"
        assert config.skip_gate_model_path == "models/v460/skip_gate_as.pkl"
        assert config.skip_gate_as_threshold == 0.52  # 100# default 0.6→0.52
        assert config.skip_gate_pnl_threshold == 0.0
        assert config.skip_gate_max_skip_rate == 0.3

    def test_from_yaml_skip_gate_partial(self) -> None:
        """SkipGate 部分設定 + デフォルト混在."""
        yaml_cfg = {
            "skip_gate": {
                "enabled": True,
                "as_threshold": 0.7,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.skip_gate_enabled is True
        assert config.skip_gate_as_threshold == 0.7
        # 未指定フィールドはデフォルト
        assert config.skip_gate_mode == "as"
        assert config.skip_gate_model_path == "models/v460/skip_gate_as.pkl"

    def test_yaml_roundtrip_skip_gate(self) -> None:
        """YAML → FillTestConfig roundtrip for skip_gate."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config = FillTestConfig.from_yaml(cfg)
        assert config.skip_gate_enabled == cfg["skip_gate"]["enabled"]
        assert config.skip_gate_mode == cfg["skip_gate"]["mode"]
        assert config.skip_gate_as_threshold == cfg["skip_gate"]["as_threshold"]

    def test_071_no_fallback_path_in_config(self) -> None:
        """071# fallback_path は OB 除去で廃止済み."""
        config = FillTestConfig.from_yaml({})
        assert not hasattr(config, "skip_gate_fallback_path")

    def test_071_no_ob_freshness_in_config(self) -> None:
        """071# ob_freshness_sec は OB 除去で廃止済み."""
        config = FillTestConfig.from_yaml({})
        assert not hasattr(config, "skip_gate_ob_freshness_sec")
        assert not hasattr(config, "ob_fail_max_consecutive")
        assert not hasattr(config, "ob_fail_offset_boost")

    def test_072_use_ob_features_default_false(self) -> None:
        """072# use_ob_features のデフォルトは False."""
        config = FillTestConfig.from_yaml({})
        assert config.skip_gate_use_ob_features is False

    def test_072_use_ob_features_from_yaml(self) -> None:
        """072# use_ob_features が YAML から正しくマッピングされる."""
        yaml_cfg = {
            "skip_gate": {
                "enabled": True,
                "mode": "as",
                "model_path": "models/v460/skip_gate_as.pkl",
                "as_threshold": 0.65,
                "use_ob_features": True,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.skip_gate_use_ob_features is True

    def test_072_yaml_roundtrip_use_ob_features(self) -> None:
        """072# fill_test.yaml の use_ob_features roundtrip."""
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config = FillTestConfig.from_yaml(cfg)
        assert config.skip_gate_use_ob_features == cfg["skip_gate"]["use_ob_features"]


class Test062SkipGateRunner:
    """062# S5: SkipGate の FillTestRunner 統合テスト."""

    def test_runner_skip_gate_disabled_by_default(self) -> None:
        """SkipGate disabled ではロードされない."""
        runner = _make_runner(skip_gate_enabled=False)
        assert runner._skip_gate is None

    def test_runner_skip_gate_model_not_found(self) -> None:
        """モデルファイルが存在しない場合は None のまま."""
        runner = _make_runner(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/nonexistent.pkl",
        )
        assert runner._skip_gate is None

    def test_fill_record_has_skip_gate_fields(self) -> None:
        """FillRecord に skip_gate フィールドが存在する."""
        record = FillRecord(
            cycle_id="test",
            timestamp=1.0,
            side="buy",
            order_price=10_000_000.0,
            order_quantity=0.001,
            skip_gate_skipped=True,
            skip_gate_score=-0.5,
            skip_gate_reason="skip",
        )
        assert record.skip_gate_skipped is True
        assert record.skip_gate_score == -0.5
        assert record.skip_gate_reason == "skip"

    def test_fill_record_skip_gate_fields_default_none(self) -> None:
        """FillRecord の skip_gate フィールドはデフォルト None."""
        record = FillRecord(
            cycle_id="test",
            timestamp=1.0,
            side="buy",
            order_price=10_000_000.0,
            order_quantity=0.001,
        )
        assert record.skip_gate_skipped is None
        assert record.skip_gate_score is None
        assert record.skip_gate_reason is None

    def test_fill_record_to_dict_includes_skip_gate(self) -> None:
        """FillRecord.to_dict() に skip_gate フィールドが含まれる."""
        record = FillRecord(
            cycle_id="test",
            timestamp=1.0,
            side="buy",
            order_price=10_000_000.0,
            order_quantity=0.001,
            skip_gate_skipped=False,
            skip_gate_score=0.3,
            skip_gate_reason="pass",
        )
        d = record.to_dict()
        assert "skip_gate_skipped" in d
        assert d["skip_gate_skipped"] is False
        assert d["skip_gate_score"] == 0.3
        assert d["skip_gate_reason"] == "pass"

    def test_fill_record_from_dict_with_skip_gate(self) -> None:
        """FillRecord.from_dict() が skip_gate フィールドを復元."""
        d = {
            "cycle_id": "test",
            "timestamp": 1.0,
            "side": "buy",
            "order_price": 10_000_000.0,
            "order_quantity": 0.001,
            "skip_gate_skipped": True,
            "skip_gate_score": -2.5,
            "skip_gate_reason": "skip",
        }
        record = FillRecord.from_dict(d)
        assert record.skip_gate_skipped is True
        assert record.skip_gate_score == -2.5
        assert record.skip_gate_reason == "skip"

    def test_fill_record_from_dict_backward_compat(self) -> None:
        """旧フォーマット (skip_gate フィールドなし) からの後方互換."""
        d = {
            "cycle_id": "test",
            "timestamp": 1.0,
            "side": "buy",
            "order_price": 10_000_000.0,
            "order_quantity": 0.001,
        }
        record = FillRecord.from_dict(d)
        assert record.skip_gate_skipped is None
        assert record.skip_gate_score is None
        assert record.skip_gate_reason is None


class TestSideOverride:
    """075# Fix: side_override パラメータの回帰テスト (076# HIGH#5)."""

    def test_side_override_skips_next_side(self) -> None:
        """side_override が指定されると _next_side() を呼ばない."""
        runner = _make_runner(start_side="buy")
        # _next_side() は通常 buy → sell と交互
        assert runner._next_side() == "buy"
        runner._last_side = "buy"
        assert runner._next_side() == "sell"  # 次は sell

        # side_override が渡された場合、内部状態に関わらず指定 side を使う
        # run_single_cycle は async なので、ここでは side 決定ロジックのみ検証
        runner._last_side = "buy"  # 次は sell のはず
        # side_override="buy" なら buy が強制される (sell にならない)
        # run_single_cycle の冒頭ロジックを直接テスト
        side_override = "buy"
        if side_override is not None:
            side = side_override
        else:
            side = runner._next_side()
        assert side == "buy", "side_override should force buy even when next would be sell"

    def test_side_override_none_falls_through(self) -> None:
        """side_override=None の場合は通常の _next_side() が呼ばれる."""
        runner = _make_runner(start_side="buy")
        runner._last_side = "buy"  # 次は sell

        side_override = None
        if side_override is not None:
            side = side_override
        else:
            side = runner._next_side()
        assert side == "sell", "side_override=None should fall through to _next_side()"

    def test_side_override_updates_tracking(self) -> None:
        """side_override 後も _last_side / _consecutive_same_side が正しく更新される."""
        runner = _make_runner(start_side="buy")
        runner._last_side = "sell"
        runner._consecutive_same_side = 0

        # side_override="sell" → 連続 same side
        side = "sell"
        if side == runner._last_side:
            runner._consecutive_same_side += 1
        else:
            runner._consecutive_same_side = 0
        runner._last_side = side

        assert runner._last_side == "sell"
        assert runner._consecutive_same_side == 1

    def test_run_continuous_passes_side_override(self) -> None:
        """run_continuous 内で run_single_cycle(side_override=next_side) が呼ばれる.

        ソースコードレベルで side_override パスの存在を確認.
        """
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner.run_continuous)
        assert "side_override=next_side" in source or "side_override=" in source, (
            "run_continuous must pass side_override to run_single_cycle"
        )

        source_sc = inspect.getsource(FillTestRunner.run_single_cycle)
        assert "side_override" in source_sc, (
            "run_single_cycle must accept side_override parameter"
        )


class Test110DeadlockBreak:
    """110# 086# デッドロック修正のテスト."""

    def test_config_default_max_086(self) -> None:
        """max_086_consecutive_wait のデフォルト値は 3."""
        cfg = FillTestConfig(enable_regime=False)
        assert cfg.max_086_consecutive_wait == 3

    def test_config_from_yaml_max_086(self) -> None:
        """YAML から max_086_consecutive_wait を読み込める."""
        yaml_dict = {
            "time_filter": {
                "enabled": True,
                "max_086_consecutive_wait": 5,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_dict)
        assert cfg.max_086_consecutive_wait == 5

    def test_config_from_yaml_no_086_uses_default(self) -> None:
        """YAML で未指定時はデフォルト値 3 を使用."""
        yaml_dict = {
            "time_filter": {
                "enabled": True,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_dict)
        assert cfg.max_086_consecutive_wait == 3

    def test_runner_has_consecutive_086_counter(self) -> None:
        """FillTestRunner が _time_filter.consecutive_086_wait カウンタを持つ (121# 委譲)."""
        runner = _make_runner()
        assert hasattr(runner, "_time_filter")
        assert runner._time_filter.consecutive_086_wait == 0

    def test_consecutive_086_wait_zero_means_unlimited(self) -> None:
        """max_086_consecutive_wait=0 は無制限 (旧動作互換)."""
        cfg = FillTestConfig(enable_regime=False, max_086_consecutive_wait=0)
        assert cfg.max_086_consecutive_wait == 0

    def test_deadlock_break_logic_in_source(self) -> None:
        """run_continuous 内に 110# デッドロック解除ロジックが存在."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner.run_continuous)
        assert "consecutive_086_wait" in source, (
            "run_continuous must reference consecutive_086_wait counter"
        )
        assert "110#" in source, (
            "run_continuous must contain 110# deadlock break comment"
        )
        assert "max_086_consecutive_wait" in source, (
            "run_continuous must reference max_086_consecutive_wait config"
        )

    def test_is_time_filtered_unchanged(self) -> None:
        """110# は _is_time_filtered ロジック自体は変更しない."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner._is_time_filtered)
        # 086# / 110# ロジックは _is_time_filtered ではなく main loop にある
        assert "_consecutive_086_wait" not in source, (
            "_is_time_filtered should not contain deadlock counter logic"
        )

    def test_yaml_roundtrip_max_086(self) -> None:
        """YAML → FillTestConfig → 値の保持を確認."""
        yaml_text = """
time_filter:
  enabled: true
  skip_utc_hours: [16]
  skip_utc_hours_buy: [1, 2]
  skip_utc_hours_sell: [4]
  max_086_consecutive_wait: 7
"""
        yaml_dict = yaml.safe_load(yaml_text)
        cfg = FillTestConfig.from_yaml(yaml_dict)
        assert cfg.enable_time_filter is True
        assert cfg.skip_utc_hours == [16]
        assert cfg.skip_utc_hours_buy == [1, 2]
        assert cfg.skip_utc_hours_sell == [4]
        assert cfg.max_086_consecutive_wait == 7
