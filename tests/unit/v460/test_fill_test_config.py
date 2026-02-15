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
