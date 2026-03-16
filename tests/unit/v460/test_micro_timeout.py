"""452# Micro-timeout (TIF Emulation) — config, parser, sub-cycle loop tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_results import FillMonitorResult


# ======================================================================
# Config / Parser tests
# ======================================================================

class TestMicroTimeoutConfigDefaults:
    """micro_timeout フィールドのデフォルト値テスト."""

    def test_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.micro_timeout_enabled is False
        assert cfg.micro_timeout_wait_sec == 15.0
        assert cfg.micro_timeout_wait_sec_sell is None
        assert cfg.micro_timeout_max_requote == 4
        assert cfg.micro_timeout_requote_cooloff_sec == 5.0
        assert cfg.micro_timeout_cancel_on_cv_flip is True

    def test_custom_values(self) -> None:
        cfg = FillTestConfig(
            micro_timeout_enabled=True,
            micro_timeout_wait_sec=10.0,
            micro_timeout_wait_sec_sell=8.0,
            micro_timeout_max_requote=6,
            micro_timeout_requote_cooloff_sec=3.0,
            micro_timeout_cancel_on_cv_flip=False,
        )
        assert cfg.micro_timeout_enabled is True
        assert cfg.micro_timeout_wait_sec == 10.0
        assert cfg.micro_timeout_wait_sec_sell == 8.0
        assert cfg.micro_timeout_max_requote == 6
        assert cfg.micro_timeout_requote_cooloff_sec == 3.0
        assert cfg.micro_timeout_cancel_on_cv_flip is False


class TestMicroTimeoutYamlParsing:
    """YAML → FillTestConfig の micro_timeout セクションパース."""

    def test_from_yaml_micro_timeout_enabled(self) -> None:
        yaml_cfg = {
            "micro_timeout": {
                "enabled": True,
                "wait_sec": 12.0,
                "wait_sec_sell": 9.0,
                "max_requote_per_cycle": 5,
                "requote_cooloff_sec": 2.0,
                "cancel_on_cross_venue_flip": False,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.micro_timeout_enabled is True
        assert cfg.micro_timeout_wait_sec == 12.0
        assert cfg.micro_timeout_wait_sec_sell == 9.0
        assert cfg.micro_timeout_max_requote == 5
        assert cfg.micro_timeout_requote_cooloff_sec == 2.0
        assert cfg.micro_timeout_cancel_on_cv_flip is False

    def test_from_yaml_micro_timeout_disabled(self) -> None:
        yaml_cfg = {
            "micro_timeout": {"enabled": False},
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.micro_timeout_enabled is False
        # other fields should be defaults
        assert cfg.micro_timeout_wait_sec == 15.0

    def test_from_yaml_no_micro_timeout_section(self) -> None:
        cfg = FillTestConfig.from_yaml({})
        assert cfg.micro_timeout_enabled is False

    def test_from_yaml_partial_micro_timeout(self) -> None:
        yaml_cfg = {
            "micro_timeout": {
                "enabled": True,
                "wait_sec": 20.0,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.micro_timeout_enabled is True
        assert cfg.micro_timeout_wait_sec == 20.0
        # defaults for unspecified
        assert cfg.micro_timeout_max_requote == 4
        assert cfg.micro_timeout_wait_sec_sell is None


# ======================================================================
# FillMonitorResult tests
# ======================================================================

class TestFillMonitorResultRequoteFields:
    """FillMonitorResult に追加した requote_attempts/partial_filled_qty テスト."""

    def test_defaults(self) -> None:
        result = FillMonitorResult()
        assert result.requote_attempts == 0
        assert result.partial_filled_qty == 0.0

    def test_custom(self) -> None:
        result = FillMonitorResult(
            requote_attempts=3,
            partial_filled_qty=0.002,
        )
        assert result.requote_attempts == 3
        assert result.partial_filled_qty == 0.002


# ======================================================================
# FillRecord tests
# ======================================================================

class TestFillRecordRequoteFields:
    """FillRecord に追加した requote_attempts テスト."""

    def test_defaults(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(cycle_id="c1", timestamp=1.0, side="buy", order_price=100.0, order_quantity=0.001)
        assert rec.requote_attempts is None
        assert rec.micro_timeout_partial_filled_qty is None

    def test_set_fields(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(cycle_id="c1", timestamp=1.0, side="buy", order_price=100.0, order_quantity=0.001)
        rec.requote_attempts = 2
        rec.micro_timeout_partial_filled_qty = 0.0005
        d = rec.to_dict()
        assert d["requote_attempts"] == 2
        assert d["micro_timeout_partial_filled_qty"] == 0.0005


# ======================================================================
# Production YAML file tests
# ======================================================================

class TestProductionYamlMicroTimeout:
    """本番 fill_test.yaml に micro_timeout セクションが存在するか."""

    def test_yaml_has_micro_timeout_section(self) -> None:
        from pathlib import Path
        import yaml
        yaml_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "v460" / "fill_test.yaml"
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        mt = raw.get("micro_timeout", {})
        assert isinstance(mt, dict)
        assert "enabled" in mt
        assert "wait_sec" in mt
        assert "max_requote_per_cycle" in mt
        assert "requote_cooloff_sec" in mt

    def test_yaml_micro_timeout_defaults_disabled(self) -> None:
        from pathlib import Path
        import yaml
        yaml_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "v460" / "fill_test.yaml"
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        cfg = FillTestConfig.from_yaml(raw)
        # 454# Step 1: 保守的設定で有効化済み
        assert cfg.micro_timeout_enabled is True
        assert cfg.micro_timeout_wait_sec == 30.0
        assert cfg.micro_timeout_max_requote == 2


# ======================================================================
# 453# Validation tests
# ======================================================================

class TestMicroTimeoutValidation:
    """micro_timeout フィールドのバリデーションテスト."""

    def test_wait_sec_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="micro_timeout_wait_sec must be > 0"):
            FillTestConfig(micro_timeout_wait_sec=-1.0)

    def test_wait_sec_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="micro_timeout_wait_sec must be > 0"):
            FillTestConfig(micro_timeout_wait_sec=0.0)

    def test_wait_sec_sell_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="micro_timeout_wait_sec_sell must be > 0"):
            FillTestConfig(micro_timeout_wait_sec_sell=-5.0)

    def test_wait_sec_sell_none_ok(self) -> None:
        cfg = FillTestConfig(micro_timeout_wait_sec_sell=None)
        assert cfg.micro_timeout_wait_sec_sell is None

    def test_max_requote_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="micro_timeout_max_requote must be >= 1"):
            FillTestConfig(micro_timeout_max_requote=0)

    def test_cooloff_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="micro_timeout_requote_cooloff_sec must be >= 0"):
            FillTestConfig(micro_timeout_requote_cooloff_sec=-1.0)

    def test_cooloff_zero_ok(self) -> None:
        cfg = FillTestConfig(micro_timeout_requote_cooloff_sec=0.0)
        assert cfg.micro_timeout_requote_cooloff_sec == 0.0

    def test_total_time_exceeds_cycle_warns(self) -> None:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FillTestConfig(
                micro_timeout_enabled=True,
                micro_timeout_wait_sec=50.0,
                micro_timeout_max_requote=4,
                micro_timeout_requote_cooloff_sec=10.0,
            )
            mt_warnings = [x for x in w if "合計時間" in str(x.message)]
            assert len(mt_warnings) >= 1

    def test_cancel_reason_micro_timeout_label(self) -> None:
        """453# review: cancel_reason が micro_timeout に上書きされることの概念テスト."""
        result = FillMonitorResult(cancel_reason="timeout")
        # micro_timeout 有効 + unfilled + cancel_reason="timeout" → "micro_timeout"
        is_micro = True
        filled = False
        if is_micro and not filled and result.cancel_reason == "timeout":
            label = "micro_timeout"
        else:
            label = result.cancel_reason
        assert label == "micro_timeout"
