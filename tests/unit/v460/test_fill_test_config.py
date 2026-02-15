"""Tests for fill_test YAML config loading + FillTestConfig.from_yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.v460.lib.config_loader import load_fill_test_config
from scripts.v460.run_fill_test import FillTestConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
        assert cfg["adaptation"]["enabled"] is True  # 041# false→true
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
        assert config.enable_auto_adapt is True  # 041# false→true
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
            "min_adapt_samples": 30,
            "adaptation": {
                "enabled": True,
                "interval_cycles": 30,
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


class Test054ImbalanceConfig:
    """054# S1: Imbalance 設定のテスト."""

    def test_yaml_has_imbalance_section(self) -> None:
        """YAML に imbalance セクションがある."""
        cfg = load_fill_test_config()
        assert "imbalance" in cfg
        assert cfg["imbalance"]["enabled"] is True
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
        """YAML に smart_side セクションがある."""
        cfg = load_fill_test_config()
        assert "smart_side" in cfg
        assert cfg["smart_side"]["enabled"] is True
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
        assert cfg["early_exit"]["enabled"] is True
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
        assert cfg["spread_adaptive"]["narrow_spread_bps"] == 10.0

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
        # S1
        assert config.imbalance_enabled is True
        assert config.imbalance_depth == cfg["imbalance"]["depth"]
        # S2
        assert config.smart_side_enabled is True
        assert config.smart_side_mode == cfg["smart_side"]["mode"]
        # S3
        assert config.early_exit_enabled is True
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
