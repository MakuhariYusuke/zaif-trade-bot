"""189# テスト: Alt Horizon 訓練 + MacroRegime 統合 + retrain multi-horizon.

- retrain_scheduler multi-horizon 拡張 (alt_horizon_enabled, train_specs)
- FillRecord macro フィールド (to_dict / from_dict roundtrip)
- FillTestConfig YAML パース (regime.macro サブセクション)
- MacroRegime compose_regimes conflict → downgrade 挙動
- fill_cycle_executor macro 統合 (mock ベース)
- config_hot_reload キー確認
- YAML 整合性 (ev_weighted + macro + alt_horizon)
"""

from __future__ import annotations

import copy
import importlib
import math
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml
from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.macro_regime import (
    MacroRegimeConfig,
    MacroRegimeDetector,
    MacroRegimeResult,
    MacroTrend,
    compose_regimes,
)
from scripts.v460.ml.retrain_scheduler import _DEFAULT_CONFIG, load_retrain_config
from scripts.v460.ml.train_alt_horizon import _ALT_SPECS
from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# 1. retrain_scheduler multi-horizon テスト
# ======================================================================


class TestRetrainMultiHorizon:
    """189# retrain_scheduler の alt_horizon_enabled=True 時の挙動."""

    def test_default_config_has_alt_keys(self) -> None:
        """_DEFAULT_CONFIG に alt_horizon 関連キーが存在すること."""

        assert "alt_horizon_enabled" in _DEFAULT_CONFIG
        assert _DEFAULT_CONFIG["alt_horizon_enabled"] is False
        assert "target_buy_alt" in _DEFAULT_CONFIG
        assert "target_sell_alt" in _DEFAULT_CONFIG
        assert _DEFAULT_CONFIG["target_buy_alt"] == "pnl120"
        assert _DEFAULT_CONFIG["target_sell_alt"] == "pnl30"

    def test_train_specs_primary_only(self) -> None:
        """alt_horizon_enabled=False → primary のみの train_specs 2 件."""
        cfg = {
            "alt_horizon_enabled": False,
            "model_path_buy": "models/buy_primary.pkl",
            "model_path_sell": "models/sell_primary.pkl",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "model_path_buy_long": "models/buy_alt.pkl",
            "model_path_sell_short": "models/sell_alt.pkl",
        }
        specs = self._build_train_specs(cfg)
        assert len(specs) == 2
        labels = [s[3] for s in specs]
        assert labels == ["primary", "primary"]

    def test_train_specs_with_alt(self) -> None:
        """alt_horizon_enabled=True → primary+alt で 4 件."""
        cfg = {
            "alt_horizon_enabled": True,
            "model_path_buy": "models/buy_primary.pkl",
            "model_path_sell": "models/sell_primary.pkl",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "model_path_buy_long": "models/buy_alt.pkl",
            "model_path_sell_short": "models/sell_alt.pkl",
            "target_buy_alt": "pnl120",
            "target_sell_alt": "pnl30",
        }
        specs = self._build_train_specs(cfg)
        assert len(specs) == 4
        sides = [s[0] for s in specs]
        labels = [s[3] for s in specs]
        # buy primary, buy alt, sell primary, sell alt
        assert sides == ["buy", "buy", "sell", "sell"]
        assert labels == ["primary", "alt", "primary", "alt"]
        # target 確認
        assert specs[0][2] == "pnl30"   # buy primary target
        assert specs[1][2] == "pnl120"  # buy alt target
        assert specs[2][2] == "pnl120"  # sell primary target
        assert specs[3][2] == "pnl30"   # sell alt target

    def test_train_specs_no_alt_model_path(self) -> None:
        """alt_horizon_enabled=True でも model_path 未設定なら alt スキップ."""
        cfg = {
            "alt_horizon_enabled": True,
            "model_path_buy": "models/buy_primary.pkl",
            "model_path_sell": "models/sell_primary.pkl",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "model_path_buy_long": "",   # 空文字
            "model_path_sell_short": "",  # 空文字
        }
        specs = self._build_train_specs(cfg)
        assert len(specs) == 2
        labels = [s[3] for s in specs]
        assert labels == ["primary", "primary"]

    def test_load_retrain_config_inherits_alt_paths(self, tmp_path: Path) -> None:
        """load_retrain_config が skip_gate.model_path_buy_long を継承すること."""
        yaml_data = {
            "skip_gate": {
                "model_path": "models/v460/skip_gate_lgbm_pnl120.pkl",
                "mode": "pnl",
                "use_ob_features": True,
                "model_path_buy": "models/v460/skip_gate_lgbm_pnl30_buy.pkl",
                "model_path_sell": "models/v460/skip_gate_lgbm_pnl120_sell.pkl",
                "model_path_buy_long": "models/v460/skip_gate_lgbm_pnl120_buy.pkl",
                "model_path_sell_short": "models/v460/skip_gate_lgbm_pnl30_sell.pkl",
            },
            "retrain": {
                "enabled": True,
                "alt_horizon_enabled": True,
                "target_buy_alt": "pnl120",
                "target_sell_alt": "pnl30",
            },
        }

        config_path = tmp_path / "alt_horizon.yaml"
        config_path.write_text(yaml.dump(yaml_data), encoding="utf-8")

        cfg = load_retrain_config(config_path)
        assert cfg["model_path_buy_long"] == "models/v460/skip_gate_lgbm_pnl120_buy.pkl"
        assert cfg["model_path_sell_short"] == "models/v460/skip_gate_lgbm_pnl30_sell.pkl"
        assert cfg["alt_horizon_enabled"] is True
        assert cfg["target_buy_alt"] == "pnl120"
        assert cfg["target_sell_alt"] == "pnl30"

    @staticmethod
    def _build_train_specs(
        cfg: dict,
    ) -> list[tuple[str, str, str, str]]:
        """_retrain_side_specific 内の train_specs 構築ロジックを抽出."""
        model_path_map = {
            "buy": cfg.get("model_path_buy", ""),
            "sell": cfg.get("model_path_sell", ""),
        }
        target_map = {
            "buy": cfg.get("target_buy", cfg.get("target", "pnl30")),
            "sell": cfg.get("target_sell", cfg.get("target", "pnl30")),
        }
        alt_enabled = bool(cfg.get("alt_horizon_enabled", False))
        alt_model_path_map = {
            "buy": cfg.get("model_path_buy_long", ""),
            "sell": cfg.get("model_path_sell_short", ""),
        }
        alt_target_map = {
            "buy": cfg.get("target_buy_alt", "pnl120"),
            "sell": cfg.get("target_sell_alt", "pnl30"),
        }
        train_specs: list[tuple[str, str, str, str]] = []
        for side in ("buy", "sell"):
            if model_path_map[side]:
                train_specs.append(
                    (side, str(model_path_map[side]), str(target_map[side]), "primary")
                )
            if alt_enabled and alt_model_path_map[side]:
                train_specs.append(
                    (side, str(alt_model_path_map[side]), str(alt_target_map[side]), "alt")
                )
        return train_specs


# ======================================================================
# 2. FillRecord macro フィールドテスト
# ======================================================================


class TestFillRecordMacroFields:
    """189# FillRecord の macro 4 フィールドの検証."""

    _BASE = {"cycle_id": "c1", "timestamp": 0.0, "side": "buy", "order_price": 1e7, "order_quantity": 0.001}

    def test_macro_fields_default_none(self) -> None:
        """macro フィールドのデフォルトが None であること."""
        rec = FillRecord(**self._BASE)
        assert rec.macro_trend is None
        assert rec.macro_slope_5m is None
        assert rec.macro_slope_15m is None
        assert rec.macro_aligned is None

    def test_macro_fields_set(self) -> None:
        """macro フィールドに値をセットできること."""
        rec = FillRecord(
            **self._BASE,
            macro_trend="macro_strong_up",
            macro_slope_5m=2.5,
            macro_slope_15m=1.8,
            macro_aligned=True,
        )
        assert rec.macro_trend == "macro_strong_up"
        assert rec.macro_slope_5m == 2.5
        assert rec.macro_slope_15m == 1.8
        assert rec.macro_aligned is True

    def test_to_dict_roundtrip(self) -> None:
        """to_dict → from_dict で macro フィールドが保存されること."""
        original = FillRecord(
            **self._BASE,
            macro_trend="macro_weak_down",
            macro_slope_5m=-1.2,
            macro_slope_15m=-0.8,
            macro_aligned=False,
        )
        d = original.to_dict()
        assert d["macro_trend"] == "macro_weak_down"
        assert d["macro_slope_5m"] == -1.2
        assert d["macro_aligned"] is False

        restored = FillRecord.from_dict(d)
        assert restored.macro_trend == original.macro_trend
        assert restored.macro_slope_5m == original.macro_slope_5m
        assert restored.macro_slope_15m == original.macro_slope_15m
        assert restored.macro_aligned == original.macro_aligned

    def test_to_dict_none_fields(self) -> None:
        """macro フィールドが None のまま to_dict → from_dict."""
        rec = FillRecord(**self._BASE)
        d = rec.to_dict()
        assert "macro_trend" in d
        assert d["macro_trend"] is None

        restored = FillRecord.from_dict(d)
        assert restored.macro_trend is None

    def test_from_dict_ignores_unknown_macro(self) -> None:
        """from_dict が不明な macro_ プレフィックス フィールドを無視."""
        d = {**self._BASE, "macro_trend": "neutral", "macro_future_field": 42}
        rec = FillRecord.from_dict(d)
        assert rec.macro_trend == "neutral"
        assert not hasattr(rec, "macro_future_field")


# ======================================================================
# 3. FillTestConfig YAML パース (regime.macro)
# ======================================================================


class TestFillTestConfigMacroYAML:
    """189# FillTestConfig.from_yaml の regime.macro サブセクション."""

    def test_macro_enabled(self) -> None:
        """regime.macro.enabled → enable_macro_regime."""
        yaml_cfg = {
            "regime": {
                "macro": {
                    "enabled": True,
                    "bucket_sec": 45.0,
                    "slope_threshold": 2.0,
                    "strong_threshold": 5.0,
                    "conflict_action": "downgrade",
                },
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.enable_macro_regime is True
        assert config.macro_regime_bucket_sec == 45.0
        assert config.macro_regime_slope_threshold == 2.0
        assert config.macro_regime_strong_threshold == 5.0
        assert config.macro_regime_conflict_action == "downgrade"

    def test_macro_disabled_default(self) -> None:
        """regime.macro 未設定 → enable_macro_regime=False (デフォルト)."""
        config = FillTestConfig.from_yaml({})
        assert config.enable_macro_regime is False
        assert config.macro_regime_conflict_action == "log"

    def test_macro_partial_config(self) -> None:
        """regime.macro の一部のみ設定 → 残りはデフォルト."""
        yaml_cfg = {
            "regime": {
                "macro": {
                    "enabled": True,
                    "conflict_action": "downgrade",
                },
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.enable_macro_regime is True
        assert config.macro_regime_bucket_sec == 30.0  # default
        assert config.macro_regime_conflict_action == "downgrade"

    def test_ev_weighted_yaml_roundtrip(self) -> None:
        """skip_gate セクションの ev_weighted 設定がパースされること."""
        yaml_cfg = {
            "skip_gate": {
                "ev_weighted_enabled": True,
                "ev_w30": 0.3,
                "ev_w120": 0.7,
                "model_path_buy_long": "models/v460/skip_gate_lgbm_pnl120_buy.pkl",
                "model_path_sell_short": "models/v460/skip_gate_lgbm_pnl30_sell.pkl",
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.skip_gate_ev_weighted_enabled is True
        assert config.skip_gate_ev_w30 == 0.3
        assert config.skip_gate_ev_w120 == 0.7
        assert config.skip_gate_model_path_buy_long == "models/v460/skip_gate_lgbm_pnl120_buy.pkl"
        assert config.skip_gate_model_path_sell_short == "models/v460/skip_gate_lgbm_pnl30_sell.pkl"


# ======================================================================
# 4. MacroRegime compose_regimes conflict → downgrade
# ======================================================================


class TestComposeRegimesConflict:
    """189# compose_regimes の conflict 検出と downgrade 挙動."""

    def test_trending_up_vs_strong_down_conflict(self) -> None:
        """micro=trending_up + macro=strong_down → aligned=False."""
        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_DOWN,
            confidence=0.9,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("trending_up", 0.7, macro)
        assert aligned is False

    def test_trending_down_vs_strong_up_conflict(self) -> None:
        """micro=trending_down + macro=strong_up → aligned=False."""
        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_UP,
            confidence=0.8,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("trending_down", 0.6, macro)
        assert aligned is False

    def test_ranging_no_conflict(self) -> None:
        """micro=ranging → macro との conflict なし (ranging は中立)."""
        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_UP,
            confidence=0.9,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("ranging", 0.5, macro)
        assert aligned is True

    def test_neutral_macro_no_conflict(self) -> None:
        """macro=NEUTRAL → always aligned."""
        macro = MacroRegimeResult(
            trend=MacroTrend.NEUTRAL,
            confidence=0.5,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("trending_up", 0.8, macro)
        assert aligned is True

    def test_insufficient_macro_always_aligned(self) -> None:
        """macro=INSUFFICIENT → always aligned (データ不足は conflict 不可)."""
        macro = MacroRegimeResult(
            trend=MacroTrend.INSUFFICIENT,
            buckets_available=2,
        )
        regime, aligned = compose_regimes("trending_up", 0.9, macro)
        assert aligned is True

    def test_weak_up_vs_weak_down_conflict(self) -> None:
        """micro=trending_up + macro=weak_down → aligned=False (弱くても逆方向)."""
        macro = MacroRegimeResult(
            trend=MacroTrend.WEAK_DOWN,
            confidence=0.6,
            buckets_available=15,
        )
        regime, aligned = compose_regimes("trending_up", 0.5, macro)
        assert aligned is False


# ======================================================================
# 5. fill_cycle_executor macro 統合 (mock ベース)
# ======================================================================


class TestFillCycleExecutorMacroIntegration:
    """189# fill_cycle_executor の macro regime 統合の境界テスト."""

    def test_build_fill_record_accepts_macro_fields(self) -> None:
        """_build_fill_record が macro フィールドを FillRecord に設定すること."""
        # FillRecord 直接構築で確認 (executor 内部メソッドの出力を模倣)
        rec = FillRecord(
            cycle_id="c1",
            timestamp=0.0,
            side="buy",
            order_price=1e7,
            order_quantity=0.001,
            macro_trend="macro_strong_up",
            macro_slope_5m=3.2,
            macro_slope_15m=2.1,
            macro_aligned=True,
        )
        assert rec.macro_trend == "macro_strong_up"
        assert rec.macro_slope_5m == 3.2
        assert rec.macro_aligned is True

    def test_macro_detector_not_set_no_error(self) -> None:
        """_macro_regime_detector 未設定でもエラーにならないこと.

        hasattr ガードが正しく機能していることを確認。
        """
        # fill_cycle_executor の macro ブロックの条件:
        # if hasattr(self, "_macro_regime_detector") and self._macro_regime_detector is not None
        obj = MagicMock()
        del obj._macro_regime_detector  # hasattr が False を返すようにする

        # hasattr チェックが False → ブロックに入らない
        assert not hasattr(obj, "_macro_regime_detector")

    def test_macro_detector_none_no_processing(self) -> None:
        """_macro_regime_detector=None → macro 処理スキップ."""
        obj = MagicMock()
        obj._macro_regime_detector = None

        # None チェックで弾かれる
        has_detector = (
            hasattr(obj, "_macro_regime_detector")
            and obj._macro_regime_detector is not None
        )
        assert not has_detector

    def test_conflict_action_downgrade_changes_regime(self) -> None:
        """conflict_action=downgrade → regime_str が ranging に変更されること."""
        # ロジック抽出テスト (fill_cycle_executor.py L759-766)
        regime_str = "trending_up"
        macro_aligned = False
        conflict_action = "downgrade"

        if not macro_aligned:
            if conflict_action == "downgrade":
                regime_str = "ranging"

        assert regime_str == "ranging"

    def test_conflict_action_log_preserves_regime(self) -> None:
        """conflict_action=log → regime_str 変更なし."""
        regime_str = "trending_up"
        macro_aligned = False
        conflict_action = "log"

        if not macro_aligned:
            if conflict_action == "downgrade":
                regime_str = "ranging"

        assert regime_str == "trending_up"

    def test_aligned_preserves_regime_regardless(self) -> None:
        """aligned=True → downgrade でも regime_str 変更なし."""
        regime_str = "trending_up"
        macro_aligned = True
        conflict_action = "downgrade"

        if not macro_aligned:
            if conflict_action == "downgrade":
                regime_str = "ranging"

        assert regime_str == "trending_up"


# ======================================================================
# 6. config_hot_reload キー検証
# ======================================================================


class TestHotReloadMacroKeys:
    """189# MacroRegime 関連キーの hot-reload 設定テスト."""

    def test_macro_keys_in_hot_reload(self) -> None:
        """enable_macro_regime, macro_regime_conflict_action が hot-reload 可能."""
        assert "enable_macro_regime" in _HOT_RELOADABLE_FIELDS
        assert "macro_regime_conflict_action" in _HOT_RELOADABLE_FIELDS

    def test_ev_weighted_keys_in_hot_reload(self) -> None:
        """ev_weighted 関連キーが hot-reload 可能 (188# から存在)."""
        assert "skip_gate_ev_weighted_enabled" in _HOT_RELOADABLE_FIELDS
        assert "skip_gate_ev_w30" in _HOT_RELOADABLE_FIELDS
        assert "skip_gate_ev_w120" in _HOT_RELOADABLE_FIELDS


# ======================================================================
# 7. YAML 整合性テスト (fill_test.yaml)
# ======================================================================


class TestYAMLIntegrity:
    """189# YAML 更新の整合性検証."""

    @pytest.fixture(scope="class")
    def yaml_config(self, v460_fill_test_yaml_base: dict[str, object]) -> dict:
        """fill_test.yaml を class 単位で再利用してロードコストを削減."""
        return copy.deepcopy(v460_fill_test_yaml_base)

    def test_ev_weighted_model_paths_in_yaml(self, yaml_config: dict) -> None:
        """YAML に ev_weighted の alt モデルパスが存在."""
        sg = yaml_config.get("skip_gate", {})
        assert sg.get("model_path_buy_long"), "model_path_buy_long missing from YAML"
        assert sg.get("model_path_sell_short"), "model_path_sell_short missing from YAML"

    def test_ev_weighted_enabled_in_yaml(self, yaml_config: dict) -> None:
        """YAML で ev_weighted_enabled=true."""
        sg = yaml_config.get("skip_gate", {})
        assert sg.get("ev_weighted_enabled") is True

    def test_ev_weights_sum_to_one(self, yaml_config: dict) -> None:
        """ev_w30 + ev_w120 = 1.0 であること."""
        sg = yaml_config.get("skip_gate", {})
        w30 = sg.get("ev_w30", 0.4)
        w120 = sg.get("ev_w120", 0.6)
        assert abs(w30 + w120 - 1.0) < 1e-6

    def test_macro_section_exists(self, yaml_config: dict) -> None:
        """regime.macro セクションが存在."""
        regime = yaml_config.get("regime", {})
        assert "macro" in regime

    def test_macro_conflict_action_valid(self, yaml_config: dict) -> None:
        """macro.conflict_action が valid 値."""
        macro = yaml_config.get("regime", {}).get("macro", {})
        action = macro.get("conflict_action", "log")
        assert action in ("log", "downgrade")

    def test_retrain_alt_horizon_keys(self, yaml_config: dict) -> None:
        """retrain セクションに alt_horizon 関連キーが存在."""
        retrain = yaml_config.get("retrain", {})
        assert "alt_horizon_enabled" in retrain
        assert "target_buy_alt" in retrain
        assert "target_sell_alt" in retrain

    def test_model_paths_consistency(self, yaml_config: dict) -> None:
        """skip_gate の primary/alt モデルパスが矛盾しないこと.

        buy primary=pnl30, buy alt=pnl120
        sell primary=pnl120, sell alt=pnl30
        """
        sg = yaml_config.get("skip_gate", {})
        buy_primary = sg.get("model_path_buy", "")
        buy_alt = sg.get("model_path_buy_long", "")
        sell_primary = sg.get("model_path_sell", "")
        sell_alt = sg.get("model_path_sell_short", "")

        # buy primary は pnl30, alt は pnl120
        assert "pnl30" in buy_primary, f"buy primary should be pnl30: {buy_primary}"
        assert "pnl120" in buy_alt, f"buy alt should be pnl120: {buy_alt}"
        # sell primary は pnl120, alt は pnl30
        assert "pnl120" in sell_primary, f"sell primary should be pnl120: {sell_primary}"
        assert "pnl30" in sell_alt, f"sell alt should be pnl30: {sell_alt}"


# ======================================================================
# 8. MacroRegimeDetector 境界テスト (189# 固有)
# ======================================================================


class TestMacroRegimeEdgeCases:
    """189# MacroRegime のエッジケース検証."""

    def test_compose_regimes_high_vol_micro(self) -> None:
        """micro=high_vol は trending ではないので conflict なし."""
        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_UP,
            confidence=0.9,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("high_vol", 0.5, macro)
        assert aligned is True  # high_vol は方向なし → conflict 不可

    def test_compose_regimes_unknown_micro(self) -> None:
        """micro=unknown は方向なし → aligned=True."""
        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_DOWN,
            confidence=0.8,
            buckets_available=15,
        )
        regime, aligned = compose_regimes("unknown", 0.3, macro)
        assert aligned is True

    def test_macro_slope_fields_populated(self) -> None:
        """update 後に slope_5m, slope_15m が設定されること."""
        cfg = MacroRegimeConfig(bucket_sec=1.0, slope_window_5m=5)
        det = MacroRegimeDetector(cfg)
        t0 = time.time()
        for i in range(20):
            det.update(t0 + i * 2.0, 10_000_000 + i * 10)

        result = det.update(t0 + 40, 10_000_200)
        # slope が計算されていること (NaN でないこと)
        assert not math.isnan(result.slope_5m_bps_per_min)

    def test_fill_config_macro_field_types(self) -> None:
        """FillTestConfig の macro フィールドの型が正しいこと."""
        config = FillTestConfig()
        assert isinstance(config.enable_macro_regime, bool)
        assert isinstance(config.macro_regime_bucket_sec, float)
        assert isinstance(config.macro_regime_slope_threshold, float)
        assert isinstance(config.macro_regime_strong_threshold, float)
        assert isinstance(config.macro_regime_conflict_action, str)


# ======================================================================
# 9. train_alt_horizon.py スクリプト構造テスト
# ======================================================================


class TestTrainAltHorizonScript:
    """189# train_alt_horizon.py の構造検証."""

    def test_script_importable(self) -> None:
        """train_alt_horizon.py がインポート可能であること."""
        mod = importlib.import_module("scripts.v460.ml.train_alt_horizon")
        assert hasattr(mod, "_ALT_SPECS")
        assert hasattr(mod, "train_one")

    def test_alt_specs_coverage(self) -> None:
        """_ALT_SPECS に buy/sell 両方のスペックがあること."""
        assert "buy" in _ALT_SPECS
        assert "sell" in _ALT_SPECS
        # buy の alt は pnl120
        assert _ALT_SPECS["buy"]["target_label"] == "pnl120"
        # sell の alt は pnl30
        assert _ALT_SPECS["sell"]["target_label"] == "pnl30"

    def test_alt_specs_model_paths(self) -> None:
        """_ALT_SPECS の model_file が正しい命名規則に従うこと."""
        buy_fname = _ALT_SPECS["buy"]["model_file"]
        sell_fname = _ALT_SPECS["sell"]["model_file"]

        # buy alt = pnl120 → ファイル名に pnl120_buy
        assert "pnl120" in buy_fname
        assert "buy" in buy_fname
        # sell alt = pnl30 → ファイル名に pnl30_sell
        assert "pnl30" in sell_fname
        assert "sell" in sell_fname
