"""157# §19: 未活用レジーム機能の実装テスト.

§18.5 Phase E 候補の実装:
  1. BuyDynamicKillManager — buy 側 rolling PnL 自動停止 (sell との対称性)
  2. trending offset boost buy/sell 非対称化
  3. AdvancedRegimeDetector archived 移動
  4. stale_check_after_sec_sell 30→20s
  + retrain 機能検証
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import ztb.analysis.regime as regime_module
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
import scripts.v460.ml.retrain_scheduler as rs
from scripts.v460.ml.retrain_scheduler import _DEFAULT_CONFIG, load_retrain_config
from tests.unit.v460._fill_test_source import (
    FILL_TEST_CLI,
    MAKER_REGIME_BOOST,
    SKIP_GATE_EVALUATOR,
    read_class_method_source,
    read_source_text,
)
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from ztb.analysis.regime import MarketRegime, MarketRegimeDetector
from ztb.trading.common import cancel_reasons as CR
from ztb.trading.signal.regime.regime_detector import FillTestRegime
from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig
from ztb.risk.sell_dynamic_kill import (
    BuyDynamicKillManager,
    DynamicKillConfig,
    DynamicKillManager,
    SellDynamicKillManager,
    SellKillConfig,
)

_V460_FILL_TEST_YAML_PATH = Path(__file__).resolve().parents[3] / "configs" / "v460" / "fill_test.yaml"


@lru_cache(maxsize=1)
def _load_cached_retrain_config() -> dict[str, object]:
    return load_retrain_config(_V460_FILL_TEST_YAML_PATH)


# =====================================================================
# 1. DynamicKillManager — DRY 化 + BuyDynamicKillManager
# =====================================================================


class TestDynamicKillManagerDRY:
    """DynamicKillManager の DRY 化テスト."""

    def test_sell_kill_manager_backward_compat(self) -> None:
        """SellDynamicKillManager / SellKillConfig エイリアスが存続."""
        assert SellDynamicKillManager is DynamicKillManager
        assert SellKillConfig is DynamicKillConfig

    def test_sell_kill_manager_default_side(self) -> None:
        """SellDynamicKillManager() のデフォルト side は 'sell'."""
        mgr = SellDynamicKillManager()
        assert mgr.side == "sell"

    def test_buy_kill_manager_side(self) -> None:
        """BuyDynamicKillManager() の side は 'buy'."""
        mgr = BuyDynamicKillManager()
        assert mgr.side == "buy"

    def test_dynamic_kill_manager_custom_side(self) -> None:
        """DynamicKillManager(side='custom') のカスタム side."""
        mgr = DynamicKillManager(side="custom")
        assert mgr.side == "custom"

    def test_telemetry_includes_side(self) -> None:
        """テレメトリに side フィールドが含まれる."""
        mgr = BuyDynamicKillManager()
        _, telemetry = mgr.check_kill()
        assert telemetry.side == "buy"

    def test_sell_telemetry_side(self) -> None:
        """sell side テレメトリ確認."""
        mgr = SellDynamicKillManager()
        _, telemetry = mgr.check_kill()
        assert telemetry.side == "sell"


class TestBuyDynamicKillManager:
    """BuyDynamicKillManager 固有テスト."""

    def test_buy_kill_activation(self) -> None:
        """rolling PnL が閾値以下で buy kill 発動."""
        cfg = DynamicKillConfig(window=5, threshold_bps=-1.0, resume_window=3)
        mgr = BuyDynamicKillManager(cfg)
        for _ in range(5):
            mgr.track(-2.0)
        killed, tel = mgr.check_kill()
        assert killed is True
        assert tel.cooldown_remaining == 3
        assert tel.total_kills == 1
        assert tel.side == "buy"

    def test_buy_kill_not_activated_above_threshold(self) -> None:
        """rolling PnL が閾値以上では kill しない."""
        cfg = DynamicKillConfig(window=5, threshold_bps=-1.0, resume_window=3)
        mgr = BuyDynamicKillManager(cfg)
        for _ in range(5):
            mgr.track(0.5)
        killed, tel = mgr.check_kill()
        assert killed is False

    def test_buy_kill_regime_threshold_override(self) -> None:
        """レジーム別閾値が基本閾値をオーバーライド."""
        cfg = DynamicKillConfig(
            window=5,
            threshold_bps=-1.0,
            resume_window=3,
            regime_thresholds={"trending_down": -0.3},
        )
        mgr = BuyDynamicKillManager(cfg)
        for _ in range(5):
            mgr.track(-0.5)  # -0.5 < -0.3 (trending_down threshold)
        killed, tel = mgr.check_kill(regime="trending_down")
        assert killed is True
        assert tel.threshold_used == -0.3

    def test_buy_kill_cooldown(self) -> None:
        """cooldown 期間中は kill 維持."""
        cfg = DynamicKillConfig(window=3, threshold_bps=-0.5, resume_window=2)
        mgr = BuyDynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-1.0)
        killed1, _ = mgr.check_kill()
        assert killed1 is True
        # cooldown 中
        killed2, tel2 = mgr.check_kill()
        assert killed2 is True
        assert tel2.cooldown_remaining == 1
        # cooldown 消化
        killed3, tel3 = mgr.check_kill()
        assert killed3 is False or tel3.cooldown_remaining == 0

    def test_buy_kill_disabled(self) -> None:
        """enabled=False では kill しない."""
        cfg = DynamicKillConfig(enabled=False, window=3, threshold_bps=-0.5)
        mgr = BuyDynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill()
        assert killed is False


# =====================================================================
# 2. FillTestConfig — buy_dynamic_kill フィールド
# =====================================================================


class TestFillConfigBuyDynamicKill:
    """FillTestConfig に buy_dynamic_kill 設定が存在."""

    def test_default_buy_dynamic_kill_disabled(self) -> None:
        """デフォルトでは buy_dynamic_kill は無効."""
        cfg = FillTestConfig()
        assert cfg.buy_dynamic_kill_enabled is False
        assert cfg.buy_dynamic_kill_window == 50
        assert cfg.buy_dynamic_kill_threshold_bps == -0.8  # 341# revert
        assert cfg.buy_dynamic_kill_resume_window == 10

    def test_yaml_buy_dynamic_kill_parsing(self) -> None:
        """YAML から buy_dynamic_kill 設定を読み込む."""
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(_BUY_DYNAMIC_KILL_YAML))
        assert cfg.buy_dynamic_kill_enabled is True
        assert cfg.buy_dynamic_kill_window == 30
        assert cfg.buy_dynamic_kill_threshold_bps == -0.6
        assert cfg.buy_dynamic_kill_resume_window == 5
        assert cfg.buy_dynamic_kill_regime_thresholds == {"trending_down": -0.3}


# =====================================================================
# 3. Trending offset boost buy/sell 非対称化
# =====================================================================


class TestTrendingOffsetAsymmetry:
    """trending offset boost の buy/sell 非対称テスト."""

    def test_config_default_none(self) -> None:
        """デフォルトでは side-specific boost は None (共通値にフォールバック)."""
        cfg = FillTestConfig()
        assert cfg.regime_trending_offset_boost_buy is None
        assert cfg.regime_trending_offset_boost_sell is None
        assert cfg.regime_trending_offset_boost == 1.5

    def test_yaml_side_specific_boost(self) -> None:
        """YAML から side-specific boost を読み込む."""
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(_TRENDING_OFFSET_BOOST_YAML))
        assert cfg.regime_trending_offset_boost == 1.5
        assert cfg.regime_trending_offset_boost_buy == 1.0
        assert cfg.regime_trending_offset_boost_sell == 1.8

    def test_maker_price_buy_uses_buy_boost(self) -> None:
        """maker_price が buy 時に buy-specific boost を使用."""
        cfg = FillTestConfig(
            regime_trending_offset_boost=1.5,
            regime_trending_offset_boost_buy=1.0,  # buy は boost なし
            regime_trending_offset_boost_sell=1.5,
            spread_offset_ratio=0.05,
            max_offset_ratio=0.30,
            min_offset_ratio=0.01,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
            volatility_guard_enabled=False,
            fast_fill_defense_enabled=False,
            sell_offset_floor=0.0,
            sell_max_spread_jpy=0.0,
        )
        regime_detector = MagicMock()
        regime_detector.current_regime = FillTestRegime.TRENDING_UP
        regime_detector.last_volatility_ratio = 1.0  # 648# σ refresh 対応

        ffd = MagicMock()
        ffd.should_boost.return_value = False
        ffd.get_boost_multiplier.return_value = 1.0

        calc = MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=regime_detector,
            base_offset_ratio=0.05,
        )

        # mock adapter
        adapter = MagicMock()
        ob = MagicMock()
        ob.bids = [(15_000_000, 0.1)]
        ob.asks = [(15_001_000, 0.1)]
        adapter.get_orderbook = AsyncMock(return_value=ob)

        # buy 側: boost=1.0 → offset は boost されない
        result_buy = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        # sell 側: boost=1.5 → offset は 1.5x
        result_sell = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))

        # buy の offset は sell より小さい (trending_up で buy は有利方向 → boost 不要)
        assert result_buy.effective_offset_ratio < result_sell.effective_offset_ratio

    def test_maker_price_source_has_side_specific_boost(self) -> None:
        """maker_price のソースに side 別 boost ロジックが存在."""
        # 322# God Object 分割: regime boost は RegimeBoostMixin に移管
        source = read_class_method_source(
            MAKER_REGIME_BOOST,
            "RegimeBoostMixin",
            "_resolve_trending_boost",
        )
        assert "regime_trending_offset_boost_buy" in source
        assert "regime_trending_offset_boost_sell" in source


# =====================================================================
# 4. AdvancedRegimeDetector archived 移動
# =====================================================================


class TestAdvancedRegimeDetectorArchived:
    """AdvancedRegimeDetector が __init__.py から削除されている."""

    def test_not_exported_from_regime_init(self) -> None:
        """regime パッケージから AdvancedRegimeDetector が export されていない."""
        assert "AdvancedRegimeDetector" not in regime_module.__all__

    def test_regime_init_imports_work(self) -> None:
        """regime パッケージの残存 import が正常."""
        assert MarketRegime is not None
        assert MarketRegimeDetector is not None


# =====================================================================
# 5. stale_check_after_sec_sell YAML 値
# =====================================================================


class TestStaleCheckSellReduction:
    """stale_check_after_sec_sell の YAML 値が 20s に変更."""

    def test_yaml_stale_check_sell_20s(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml の stale_check_after_sec_sell が 20.0."""
        data = v460_fill_test_yaml
        stale = data.get("stale_order", {})
        assert stale.get("check_after_sec_sell") == 20.0


# =====================================================================
# 6. cancel_reasons: BUY_DYNAMIC_KILL 定数
# =====================================================================


class TestBuyDynamicKillCancelReason:
    """BUY_DYNAMIC_KILL cancel reason 定数テスト."""

    def test_constant_exists(self) -> None:
        assert CR.BUY_DYNAMIC_KILL == "buy_dynamic_kill"

    def test_in_audit_frozenset(self) -> None:
        assert CR.BUY_DYNAMIC_KILL in CR.AUDIT_CANCEL_REASONS


# =====================================================================
# 7. Retrain functionality verification
# =====================================================================


class TestRetrainPipelineIntegrity:
    """retrain パイプラインの整合性検証."""

    def test_retrain_scheduler_importable(self) -> None:
        """retrain_scheduler がインポート可能."""
        assert hasattr(rs, "retrain_model")
        assert hasattr(rs, "load_retrain_config")
        assert hasattr(rs, "run_scheduler")

    def test_retrain_trigger_importable(self) -> None:
        """RetrainTrigger がインポート可能."""
        assert RetrainTrigger is not None

    def test_retrain_trigger_regime_multipliers(self) -> None:
        """RetrainTrigger のレジーム別 interval 倍率."""

        cfg = RetrainTriggerConfig(
            base_interval_sec=3600,
            regime_interval_multipliers={
                "high_vol": 0.5,
                "trending": 0.75,
                "ranging": 1.5,
            },
        )
        trigger = RetrainTrigger(
            results_dir=Path("/tmp/test"),
            config=cfg,
        )
        # high_vol → 短い interval
        trigger.update_regime("high_vol")
        assert trigger.get_effective_interval() == 1800  # 3600 * 0.5

        # ranging → 長い interval
        trigger.update_regime("ranging")
        assert trigger.get_effective_interval() == 5400  # 3600 * 1.5

    def test_retrain_config_loads_from_yaml(self) -> None:
        """load_retrain_config が fill_test.yaml から正常に設定を読み込む."""
        cfg = _load_cached_retrain_config()
        assert cfg["mode"] == "pnl"
        assert "model_path" in cfg

    def test_retrain_scheduler_has_regime_weighting(self) -> None:
        """retrain_scheduler が regime weighting 設定を持つ."""
        assert "regime_weighting_enabled" in _DEFAULT_CONFIG
        assert "regime_sample_weights" in _DEFAULT_CONFIG
        assert "regime_current_boost" in _DEFAULT_CONFIG

    def test_skip_gate_evaluator_hot_reload_exists(self) -> None:
        """SkipGateEvaluator に hot-reload メソッドが存在."""
        assert hasattr(SkipGateEvaluator, "_check_and_reload_model")
        # _compute_file_hash は ztb.utils.run_manifest.compute_file_hash へ委譲
        src = read_source_text(SKIP_GATE_EVALUATOR)
        assert "compute_file_hash" in src

    def test_fill_test_retrain_subprocess_integration(self) -> None:
        """fill_test_cli.py に retrain_scheduler 子プロセス起動ロジックが存在."""
        source = read_source_text(FILL_TEST_CLI)
        assert "retrain_scheduler" in source
        assert "retrain_proc" in source


# =====================================================================
# 8. YAML 一貫性
# =====================================================================


class TestYAMLConsistency:
    """fill_test.yaml の新設定の一貫性."""

    def test_buy_dynamic_kill_yaml_exists(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml に buy_dynamic_kill セクションが存在."""
        data = v460_fill_test_yaml
        buy_kill = data.get("loss_control", {}).get("buy_dynamic_kill", {})
        assert buy_kill.get("enabled") is True
        assert buy_kill.get("window") == 50
        assert buy_kill.get("threshold_bps") == -0.8  # 341# revert: 340#符号修正後復元
        assert buy_kill.get("resume_window") == 10

    def test_trending_offset_asymmetry_yaml(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml に trending_offset_boost_buy/sell が存在."""
        data = v460_fill_test_yaml
        regime = data.get("regime", {})
        assert regime.get("trending_offset_boost_buy") == 1.0
        assert regime.get("trending_offset_boost_sell") == 1.5
_BUY_DYNAMIC_KILL_YAML: dict[str, object] = {
    "loss_control": {
        "buy_dynamic_kill": {
            "enabled": True,
            "window": 30,
            "threshold_bps": -0.6,
            "resume_window": 5,
            "regime_thresholds": {"trending_down": -0.3},
        },
    },
}

_TRENDING_OFFSET_BOOST_YAML: dict[str, object] = {
    "regime": {
        "trending_offset_boost": 1.5,
        "trending_offset_boost_buy": 1.0,
        "trending_offset_boost_sell": 1.8,
    },
}
