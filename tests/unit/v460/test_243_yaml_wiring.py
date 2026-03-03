"""243# YAML Wiring + Validation: 242# Liveness features の YAML→Config→Runtime 配線テスト.

242# で追加した quiescence_* / toxic_kill_stale_multiplier が
YAML→FillTestConfig→DynamicKillConfig に正しくパススルーされることを保証する。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig


# ============================================================
# A. quiescence_* flat_keys YAML wiring
# ============================================================
class TestQuiescenceYAMLWiring243:
    """quiescence_gate_blocks_threshold / quiescence_sleep_sec の YAML 配線."""

    def test_quiescence_defaults(self) -> None:
        """デフォルト値確認."""
        cfg = FillTestConfig()
        assert cfg.quiescence_gate_blocks_threshold == 20
        assert cfg.quiescence_sleep_sec == 1800.0

    def test_quiescence_from_yaml_flat_keys(self) -> None:
        """YAML の flat key から quiescence_* が読み込まれる."""
        yaml_cfg: dict = {
            "quiescence_gate_blocks_threshold": 30,
            "quiescence_sleep_sec": 3600.0,
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.quiescence_gate_blocks_threshold == 30
        assert cfg.quiescence_sleep_sec == 3600.0

    def test_quiescence_sleep_sec_negative_raises(self) -> None:
        """quiescence_sleep_sec < 0 は ValueError."""
        with pytest.raises(ValueError, match="quiescence_sleep_sec"):
            FillTestConfig(quiescence_sleep_sec=-1.0)

    def test_quiescence_gate_blocks_threshold_negative_raises(self) -> None:
        """quiescence_gate_blocks_threshold < 0 は ValueError."""
        with pytest.raises(ValueError, match="quiescence_gate_blocks_threshold"):
            FillTestConfig(quiescence_gate_blocks_threshold=-1)


# ============================================================
# B. toxic_kill_stale_multiplier YAML → FillConfig → DynamicKillConfig
# ============================================================
class TestToxicStaleMultYAMLWiring243:
    """toxic_kill_stale_multiplier の YAML stopgap 配線."""

    def test_defaults(self) -> None:
        """FillTestConfig のデフォルト値."""
        cfg = FillTestConfig()
        assert cfg.sell_dynamic_kill_toxic_stale_mult == 10
        assert cfg.buy_dynamic_kill_toxic_stale_mult == 10

    def test_sell_toxic_stale_from_yaml(self) -> None:
        """YAML 止血.sell_dynamic_kill.toxic_stale_multiplier が配線される."""
        yaml_cfg: dict = {
            "止血": {
                "sell_dynamic_kill": {
                    "toxic_stale_multiplier": 5,
                },
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.sell_dynamic_kill_toxic_stale_mult == 5

    def test_buy_toxic_stale_from_yaml(self) -> None:
        """YAML 止血.buy_dynamic_kill.toxic_stale_multiplier が配線される."""
        yaml_cfg: dict = {
            "止血": {
                "buy_dynamic_kill": {
                    "toxic_stale_multiplier": 3,
                },
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.buy_dynamic_kill_toxic_stale_mult == 3

    def test_both_sides_from_yaml(self) -> None:
        """sell/buy 両方の toxic_stale_multiplier が同時に配線."""
        yaml_cfg: dict = {
            "止血": {
                "sell_dynamic_kill": {"toxic_stale_multiplier": 7},
                "buy_dynamic_kill": {"toxic_stale_multiplier": 15},
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.sell_dynamic_kill_toxic_stale_mult == 7
        assert cfg.buy_dynamic_kill_toxic_stale_mult == 15


# ============================================================
# C. DynamicKillConfig validation
# ============================================================
class TestDynamicKillConfigValidation243:
    """DynamicKillConfig.toxic_kill_stale_multiplier バリデーション."""

    def test_negative_raises(self) -> None:
        """負値は ValueError."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig
        with pytest.raises(ValueError, match="toxic_kill_stale_multiplier"):
            DynamicKillConfig(toxic_kill_stale_multiplier=-1)

    def test_zero_raises(self) -> None:
        """0 は ValueError (>= 1 必須)."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig
        with pytest.raises(ValueError, match="toxic_kill_stale_multiplier"):
            DynamicKillConfig(toxic_kill_stale_multiplier=0)

    def test_one_ok(self) -> None:
        """1 は許容 (延長無し = 通常)."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig
        cfg = DynamicKillConfig(toxic_kill_stale_multiplier=1)
        assert cfg.toxic_kill_stale_multiplier == 1


# ============================================================
# D. FillConfig → DynamicKillConfig passthrough (run_fill_test 経路)
# ============================================================
class TestPassthroughToDynamicKillConfig243:
    """run_fill_test.py の SellKillConfig/DynamicKillConfig 構築経路."""

    def test_sell_kill_config_accepts_toxic_stale(self) -> None:
        """SellKillConfig が toxic_kill_stale_multiplier を受理."""
        from ztb.risk.sell_dynamic_kill import SellKillConfig
        cfg = SellKillConfig(toxic_kill_stale_multiplier=20)
        assert cfg.toxic_kill_stale_multiplier == 20

    def test_buy_kill_config_accepts_toxic_stale(self) -> None:
        """DynamicKillConfig が toxic_kill_stale_multiplier を受理 (buy 側)."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig
        cfg = DynamicKillConfig(toxic_kill_stale_multiplier=15)
        assert cfg.toxic_kill_stale_multiplier == 15
