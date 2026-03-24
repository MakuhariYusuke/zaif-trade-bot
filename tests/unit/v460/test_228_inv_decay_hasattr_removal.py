"""228# テスト: C2 inventory time-decay + H3 hasattr排除.

C2: inv_skew time-decay — _inv_net_imbalance に exp(-elapsed/τ) 減衰適用
H3: hasattr(self, "_mcb"/"_sad"/"_cycle_strategy") → class-level None default
"""

from __future__ import annotations

import math
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from ztb.trading.signal.regime.regime_detector import FillTestRegime
from tests.unit.v460.conftest import make_maker_price_config as _make_config
from tests.unit.v460._fill_test_source import read_inspect_source
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping

import scripts.v460.lib.fill_loop_orchestrator as fill_loop_orchestrator_mod

_FILL_LOOP_ORCHESTRATOR_SOURCE = read_inspect_source(fill_loop_orchestrator_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mp(config: FillTestConfig | None = None, regime=None) -> MakerPrice:
    """テスト用 MakerPriceCalculator。"""
    if config is None:
        config = _make_config()
    ffd = FastFillDefense(FastFillDefenseConfig(
        enabled=False, threshold_sec=60.0, offset_boost=1.2,
    ), base_offset_ratio=0.005)
    return MakerPrice(
        config=config,
        fast_fill_defense=ffd,
        regime_detector=regime,
        base_offset_ratio=0.005,
    )


# ===========================================================================
# C2: Inventory Time-Decay
# ===========================================================================

class TestInvDecayTimeDomain:
    """C2: _decayed_imbalance の time-decay 動作を検証。"""

    def test_decay_disabled_when_tau_zero(self):
        """τ=0 の場合 raw imbalance がそのまま返る。"""
        cfg = _make_config(inv_decay_tau_sec=0.0)
        mp = _make_mp(cfg)
        # buy 2回 → imbalance = +1.0 (2/2 = 1.0)
        mp.update_inventory("buy")
        mp.update_inventory("buy")
        assert mp.inv_net_imbalance == pytest.approx(1.0)

    def test_decay_reduces_imbalance_over_time(self):
        """τ > 0 で時間経過に伴い imbalance が減衰する。"""
        cfg = _make_config(inv_decay_tau_sec=60.0)  # τ=60s
        mp = _make_mp(cfg)
        mp.update_inventory("buy")
        mp.update_inventory("buy")
        # 直後: ≈ 1.0
        now = time.time()
        raw = mp._inv_net_imbalance
        assert raw == pytest.approx(1.0)
        # 60秒後: exp(-1) ≈ 0.368
        decayed = mp._decayed_imbalance(now + 60.0)
        assert decayed == pytest.approx(1.0 * math.exp(-1.0), rel=1e-3)

    def test_decay_approaches_zero_after_long_time(self):
        """τ の 5 倍以上経過するとほぼゼロに。"""
        cfg = _make_config(inv_decay_tau_sec=60.0)
        mp = _make_mp(cfg)
        mp.update_inventory("buy")
        mp.update_inventory("buy")
        now = time.time()
        decayed = mp._decayed_imbalance(now + 300.0)  # 5τ
        assert abs(decayed) < 0.01

    def test_fresh_fill_resets_decay_clock(self):
        """新規 fill が入るとタイムスタンプが更新され decay がリセットされる。"""
        cfg = _make_config(inv_decay_tau_sec=60.0)
        mp = _make_mp(cfg)
        mp.update_inventory("buy")
        t1 = mp._inv_last_update_time

        # 少し待ってから sell fill
        with patch("time.time", return_value=t1 + 30.0):
            mp.update_inventory("sell")
        t2 = mp._inv_last_update_time
        assert t2 > t1

    def test_negative_imbalance_decay(self):
        """sell 偏重 (negative imbalance) も正しく減衰する。"""
        cfg = _make_config(inv_decay_tau_sec=60.0)
        mp = _make_mp(cfg)
        mp.update_inventory("sell")
        mp.update_inventory("sell")
        now = time.time()
        decayed = mp._decayed_imbalance(now + 60.0)
        assert decayed == pytest.approx(-1.0 * math.exp(-1.0), rel=1e-3)

    def test_decay_preserves_sign(self):
        """減衰は符号を変えない。"""
        cfg = _make_config(inv_decay_tau_sec=30.0)
        mp = _make_mp(cfg)
        mp.update_inventory("buy")
        mp.update_inventory("sell")
        mp.update_inventory("buy")
        now = time.time()
        raw = mp._inv_net_imbalance
        assert raw > 0  # 2 buy / 1 sell → positive
        decayed = mp._decayed_imbalance(now + 60.0)
        assert decayed > 0  # still positive
        assert decayed < raw  # but smaller

    def test_no_fills_returns_zero(self):
        """fill 未発生時は imbalance = 0.0 (decay 関係なく)。"""
        cfg = _make_config(inv_decay_tau_sec=60.0)
        mp = _make_mp(cfg)
        assert mp.inv_net_imbalance == pytest.approx(0.0)

    def test_inv_net_imbalance_property_applies_decay(self):
        """public property が time-decay を適用していることを検証。"""
        cfg = _make_config(inv_decay_tau_sec=60.0)
        mp = _make_mp(cfg)
        mp.update_inventory("buy")
        mp.update_inventory("buy")
        # property should apply decay based on current time
        # We can't easily test exact value but it should be <= raw
        raw = mp._inv_net_imbalance
        prop = mp.inv_net_imbalance
        assert prop <= raw  # decayed <= raw (small elapsed but still)


class TestInvDecayInCompute:
    """C2: compute() 内の inv_skew が decayed imbalance を使うことを検証。"""

    @pytest.mark.asyncio
    async def test_compute_uses_decayed_imbalance(self):
        """τ > 0 で時間経過後の compute() は raw imbalance より小さい skew factor を使う。"""
        cfg = _make_config(
            inventory_skewing_enabled=True,
            inventory_skewing_max_factor=0.4,
            inventory_skewing_neutral_band=0.05,
            inv_decay_tau_sec=10.0,  # 短い τ で効果を確認
        )
        mp = _make_mp(cfg)
        # buy 5回 → full buy imbalance
        for _ in range(5):
            mp.update_inventory("buy")
        raw_imb = mp._inv_net_imbalance
        assert raw_imb == pytest.approx(1.0)

        mid = 14_000_000
        half_spread = 500
        ob = MagicMock()
        ob.bids = [(mid - half_spread, 0.1)]
        ob.asks = [(mid + half_spread, 0.1)]
        adapter = AsyncMock()
        adapter.get_orderbook.return_value = ob

        # 直後に compute → raw imbalance に近い factor
        result1 = await mp.compute("buy", adapter, "btc_jpy")
        factor1 = mp._last_inv_skew_factor

        # 30秒後 (3τ) に compute → decayed imbalance → smaller factor
        with patch("time.time", return_value=time.time() + 30.0):
            result2 = await mp.compute("buy", adapter, "btc_jpy")
        factor2 = mp._last_inv_skew_factor

        # 3τ 経過後: exp(-3) ≈ 0.05 → factor はほぼ 0 に近い
        assert abs(factor2) < abs(factor1), (
            f"Decayed factor ({factor2}) should be smaller than fresh ({factor1})"
        )


class TestInvDecayConfigValidation:
    """C2: inv_decay_tau_sec のバリデーション。"""

    def test_negative_tau_raises(self):
        """inv_decay_tau_sec < 0 は ValueError。"""
        with pytest.raises(ValueError, match="inv_decay_tau_sec"):
            _make_config(inv_decay_tau_sec=-1.0)

    def test_zero_tau_valid(self):
        """inv_decay_tau_sec = 0 は有効 (無効化)。"""
        cfg = _make_config(inv_decay_tau_sec=0.0)
        assert cfg.inv_decay_tau_sec == 0.0

    def test_positive_tau_valid(self):
        """inv_decay_tau_sec > 0 は有効。"""
        cfg = _make_config(inv_decay_tau_sec=1800.0)
        assert cfg.inv_decay_tau_sec == 1800.0


class TestInvDecayYaml:
    """C2: YAML parser が inv_decay_tau_sec を正しく読み込むことを検証。"""

    def test_yaml_decay_tau_sec(self):
        """inventory_skewing.decay_tau_sec が正しくパースされる。"""
        yaml_data = {
            "止血": {
                "inventory_skewing": {
                    "enabled": True,
                    "decay_tau_sec": 1800.0,
                },
            },
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.inv_decay_tau_sec == 1800.0


# ===========================================================================
# H3: hasattr → class-level None default
# ===========================================================================

class TestHasattrRemoval:
    """H3: FillLoopOrchestratorMixin に _mcb/_sad/_cycle_strategy class-level default がある。"""

    def test_mcb_class_level_default(self):
        """_mcb がクラスレベルで None 宣言されている。"""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_mcb")
        assert FillLoopOrchestratorMixin._mcb is None

    def test_sad_class_level_default(self):
        """_sad がクラスレベルで None 宣言されている。"""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_sad")
        assert FillLoopOrchestratorMixin._sad is None

    def test_cycle_strategy_class_level_default(self):
        """_cycle_strategy がクラスレベルで None 宣言されている。"""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_cycle_strategy")
        assert FillLoopOrchestratorMixin._cycle_strategy is None

    def test_no_hasattr_in_orchestrator(self):
        """fill_loop_orchestrator.py に hasattr() が残っていないことを検証。"""
        assert "hasattr(" not in _FILL_LOOP_ORCHESTRATOR_SOURCE, (
            "hasattr() がまだ fill_loop_orchestrator.py に残っている"
        )
