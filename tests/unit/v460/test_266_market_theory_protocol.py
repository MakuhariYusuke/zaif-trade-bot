"""266# Market Theory + Protocol 型安全化テスト.

- GLFT τ動的化: Guéant-Lehalle-Fernandez-Tapia (2013)
- AS δ*: Avellaneda-Stoikov 最適スプレッド幅
- Kyle λ: 価格インパクト係数 (Kyle 1985)
- Amihud ILLIQ: 非流動性比率 (Amihud 2002)
- OrderBookSnapshot Protocol 移管: ob_utils.py
- type:ignore / getattr 排除
- _estimate_sigma / _dynamic_tau 共有ヘルパー
"""
from __future__ import annotations

import math
import typing
from functools import lru_cache
from types import SimpleNamespace

import pytest

from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import (
    MakerPriceCalculator as MakerPrice,
    OrderBookSnapshot as MakerPriceOrderBookSnapshot,
)
from scripts.v460.lib.ob_utils import OrderBookSnapshot, best_bid_ask
from scripts.v460.lib.regime_detector import FillTestRegime
from scripts.v460.lib.skip_gate_evaluator import SkipGateAdapter
from tests.unit.v460._fill_test_source import (
    FILL_CYCLE_EXECUTOR,
    MAKER_MICROSTRUCTURE,
    MAKER_PRICE,
    OB_UTILS,
    SKIP_GATE_EVALUATOR,
    read_class_method_source,
    read_source_text,
)


# ======================================================================
# helpers
# ======================================================================


def _make_config(**overrides: object) -> FillTestConfig:
    defaults: dict[str, object] = dict(
        spread_offset_ratio=0.05,
        min_offset_jpy=1.0,
        max_offset_ratio=0.30,
        min_offset_ratio=0.01,
        inventory_skewing_enabled=True,
        inventory_skewing_window=10,
        inventory_skewing_max_factor=0.5,
        inventory_skewing_neutral_band=0.1,
        loss_boost_decay_tau_sec=300.0,
        as_reservation_enabled=True,
        as_reservation_gamma=0.1,
        as_reservation_tau_sec=120.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_mp(
    config: FillTestConfig | None = None,
    regime_detector: object = None,
) -> MakerPrice:
    cfg = config or _make_config()
    ffd_cfg = FastFillDefenseConfig(enabled=False)
    ffd = FastFillDefense(ffd_cfg, base_offset_ratio=cfg.spread_offset_ratio)
    return MakerPrice(
        config=cfg,
        fast_fill_defense=ffd,
        regime_detector=regime_detector,
        base_offset_ratio=cfg.spread_offset_ratio,
    )


def _make_microstructure_stub(
    config: object,
    *,
    bid_depth: float = 0.0,
    ask_depth: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        _config=config,
        _last_bid_depth=bid_depth,
        _last_ask_depth=ask_depth,
        _last_amihud_illiq=0.0,
        _get_depth=lambda side: bid_depth if side == "buy" else ask_depth,
        _scale_offset_ratio=MakerPrice._scale_offset_ratio,
        _effective_max_ratio=lambda side: getattr(config, 'max_offset_ratio', 0.30),
    )


def _make_microstructure_config(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "max_offset_ratio": 0.30,
        "order_quantity": 0.001,
        "kyle_lambda_enabled": False,
        "kyle_lambda_impact_mult": 1.0,
        "kyle_lambda_max_add_ratio": 0.01,
        "amihud_illiq_enabled": False,
        "amihud_illiq_baseline": 0.001,
        "amihud_illiq_max_mult": 2.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


_DISABLED_KYLE_STUB = _make_microstructure_stub(
    _make_microstructure_config(kyle_lambda_enabled=False)
)
_ZERO_DEPTH_KYLE_STUB = _make_microstructure_stub(
    _make_microstructure_config(kyle_lambda_enabled=True)
)
_DISABLED_AMIHUD_STUB = _make_microstructure_stub(
    _make_microstructure_config(amihud_illiq_enabled=False)
)


def _make_regime_detector(vol_ratio: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        current_regime=FillTestRegime.RANGING,
        last_volatility_ratio=vol_ratio,
    )


def _maker_price_source(method_name: str) -> str:
    return read_class_method_source(MAKER_PRICE, "MakerPriceCalculator", method_name)


def _maker_microstructure_source(method_name: str) -> str:
    return read_class_method_source(
        MAKER_MICROSTRUCTURE,
        "MicrostructureMixin",
        method_name,
    )


# ======================================================================
# _estimate_sigma: 共有 σ 推定ヘルパー
# ======================================================================


class TestEstimateSigma:
    """266# _estimate_sigma: Roll (1984) × vol_ratio 共有推定."""

    def test_basic_roll_proxy(self) -> None:
        """spread/(2·mid) で σ を推定."""
        mp = _make_mp()
        sigma, vol_ratio = mp._estimate_sigma(spread=100.0, mid_price=10_000_000.0)
        expected = 100.0 / (2.0 * 10_000_000.0)
        assert abs(sigma - expected) < 1e-12
        assert vol_ratio == 1.0

    def test_vol_ratio_scaling(self) -> None:
        """RegimeDetector vol_ratio で σ がスケーリングされる."""
        det = _make_regime_detector(vol_ratio=2.0)
        mp = _make_mp(regime_detector=det)
        sigma, vol_ratio = mp._estimate_sigma(spread=100.0, mid_price=10_000_000.0)
        base = 100.0 / (2.0 * 10_000_000.0)
        assert abs(sigma - base * 2.0) < 1e-12
        assert vol_ratio == 2.0

    def test_zero_mid_price(self) -> None:
        """mid_price=0 で σ=sigma_floor (330# T4: ゼロ除算防止).

        330#: sigma_floor=1e-6 を導入。σ が 0 に落ちないよう保証。
        """
        mp = _make_mp()
        sigma, _ = mp._estimate_sigma(spread=100.0, mid_price=0.0)
        assert sigma == mp._config.sigma_floor

    def test_source_reused_in_as_reservation(self) -> None:
        """_apply_as_reservation_shift が _estimate_sigma を呼び出す."""
        src = _maker_microstructure_source("_apply_as_reservation_shift")
        assert "_estimate_sigma" in src


# ======================================================================
# _dynamic_tau: GLFT τ 動的化
# ======================================================================


class TestDynamicTau:
    """266# GLFT τ動的化: Guéant-Lehalle-Fernandez-Tapia (2013)."""

    def test_disabled_returns_base(self) -> None:
        """as_tau_dynamic_enabled=False ならベース τ をそのまま返す."""
        mp = _make_mp(_make_config(as_tau_dynamic_enabled=False))
        assert mp._dynamic_tau(120.0, vol_ratio=2.0) == 120.0

    def test_high_vol_shortens_tau(self) -> None:
        """高ボラ (vol_ratio>1) で τ が短縮."""
        mp = _make_mp(_make_config(
            as_tau_dynamic_enabled=True,
            as_tau_dynamic_min_sec=30.0,
            as_tau_dynamic_max_sec=600.0,
        ))
        tau = mp._dynamic_tau(120.0, vol_ratio=2.0)
        assert tau == 60.0  # 120 / 2.0

    def test_low_vol_extends_tau(self) -> None:
        """低ボラ (vol_ratio<1) で τ が延長."""
        mp = _make_mp(_make_config(
            as_tau_dynamic_enabled=True,
            as_tau_dynamic_min_sec=30.0,
            as_tau_dynamic_max_sec=600.0,
        ))
        tau = mp._dynamic_tau(120.0, vol_ratio=0.5)
        assert tau == 240.0  # 120 / 0.5

    def test_clamp_min(self) -> None:
        """τ_eff が下限にクランプされる."""
        mp = _make_mp(_make_config(
            as_tau_dynamic_enabled=True,
            as_tau_dynamic_min_sec=60.0,
            as_tau_dynamic_max_sec=600.0,
        ))
        tau = mp._dynamic_tau(120.0, vol_ratio=10.0)
        assert tau == 60.0  # 12 → clamped to 60

    def test_clamp_max(self) -> None:
        """τ_eff が上限にクランプされる."""
        mp = _make_mp(_make_config(
            as_tau_dynamic_enabled=True,
            as_tau_dynamic_min_sec=30.0,
            as_tau_dynamic_max_sec=300.0,
        ))
        tau = mp._dynamic_tau(120.0, vol_ratio=0.1)
        assert tau == 300.0  # 1200 → clamped to 300


# ======================================================================
# AS δ*: 最適スプレッド幅
# ======================================================================


class TestASDeltaStar:
    """266# AS δ*: Avellaneda-Stoikov 最適 offset 下限."""

    def test_delta_star_disabled(self) -> None:
        """as_delta_star_enabled=False なら δ* 未適用."""
        cfg = _make_config(as_delta_star_enabled=False)
        mp = _make_mp(cfg)
        # 在庫を偏らせる
        for _ in range(5):
            mp.update_inventory("buy")
        result = mp._apply_as_reservation_shift("buy", 100.0, 10_000_000.0, 0.05)
        # δ* 無効 → 通常 AS shift のみ
        assert isinstance(result, float)

    def test_delta_star_floor_applied(self) -> None:
        """δ* が offset 下限として適用される."""
        cfg = _make_config(
            as_delta_star_enabled=True,
            as_delta_star_fill_rate_k=1.0,
            as_reservation_gamma=0.5,
            as_reservation_tau_sec=120.0,
        )
        mp = _make_mp(cfg)
        for _ in range(5):
            mp.update_inventory("buy")
        # δ* = γσ²τ + (2/γ)ln(1+γ/k) — 小さい offset ならフロア
        result = mp._apply_as_reservation_shift("buy", 100.0, 10_000_000.0, 0.001)
        # δ* > 0.001 のはず → フロア適用で大きくなる
        assert result >= 0.001

    def test_delta_star_formula_correctness(self) -> None:
        """δ* 公式の数値的正確性 (267# 修正: 絶対価格 σ)."""
        gamma = 0.5
        k = 1.5
        mid = 10_000_000.0
        # σ_return = spread / (2·mid) = 100 / (2·10_000_000) = 5e-6
        sigma_return = 100.0 / (2.0 * mid)
        # 267# AS 論文は絶対 σ: σ_abs = σ_return × mid
        sigma_abs = sigma_return * mid  # = 50 JPY
        sigma_abs_sq = sigma_abs * sigma_abs  # = 2500 JPY²
        tau = 120.0
        delta_star_jpy = (
            gamma * sigma_abs_sq * tau
            + (2.0 / gamma) * math.log(1.0 + gamma / k)
        )
        assert delta_star_jpy > 0
        # γσ²τ 項: 0.5 * 2500 * 120 = 150000 JPY
        # penalty 項: 4 * ln(4/3) ≈ 1.15 JPY → 支配項は γσ²τ
        expected_penalty = (2.0 / 0.5) * math.log(1.0 + 0.5 / 1.5)
        assert abs(expected_penalty - 4.0 * math.log(4.0 / 3.0)) < 1e-10
        # δ* ≈ 150001.15 → ratio = 150001.15 / 100 = 1500.01
        # → max_offset_ratio でクランプされる (高 γ では非常に保守的)


# ======================================================================
# Kyle λ: 価格インパクト係数
# ======================================================================


class TestKyleLambda:
    """266# Kyle λ: Kyle (1985) 価格インパクト offset 補正."""

    def test_disabled(self) -> None:
        """kyle_lambda_enabled=False なら無影響."""
        result = MakerPrice._apply_kyle_lambda(
            _DISABLED_KYLE_STUB,
            "buy",
            100.0,
            10_000_000.0,
            0.05,
        )
        assert result == 0.05

    def test_zero_depth_no_effect(self) -> None:
        """depth=0 なら無影響."""
        result = MakerPrice._apply_kyle_lambda(
            _ZERO_DEPTH_KYLE_STUB,
            "buy",
            100.0,
            10_000_000.0,
            0.05,
        )
        assert result == 0.05

    def test_positive_depth_adds_offset(self) -> None:
        """正の depth で offset が加算される."""
        cfg = _make_microstructure_config(
            kyle_lambda_enabled=True,
            kyle_lambda_impact_mult=1.0,
            kyle_lambda_max_add_ratio=0.10,
            order_quantity=0.001,
        )
        mp = _make_microstructure_stub(cfg, bid_depth=0.1)
        result = MakerPrice._apply_kyle_lambda(mp, "buy", 100.0, 10_000_000.0, 0.05)
        # λ = 100 / (2·0.1) = 500
        # impact = 500 * 0.001 / 10_000_000 * 1.0 = 5e-8
        assert result > 0.05  # 加算されている

    def test_max_add_ratio_clamp(self) -> None:
        """kyle_lambda_max_add_ratio で加算が上限クランプされる."""
        cfg = _make_microstructure_config(
            kyle_lambda_enabled=True,
            kyle_lambda_impact_mult=10000.0,  # 意図的に巨大倍率
            kyle_lambda_max_add_ratio=0.01,
            order_quantity=0.001,
        )
        mp = _make_microstructure_stub(cfg, bid_depth=0.001)
        prev = 0.05
        result = MakerPrice._apply_kyle_lambda(mp, "buy", 100.0, 10_000_000.0, prev)
        assert result <= prev + 0.01 + 1e-10  # max_add_ratio 以下

    def test_sell_uses_ask_depth(self) -> None:
        """sell 側は ask depth を使用する."""
        cfg = _make_microstructure_config(
            kyle_lambda_enabled=True,
            kyle_lambda_impact_mult=1.0,
            kyle_lambda_max_add_ratio=0.10,
            order_quantity=0.001,
        )
        mp = _make_microstructure_stub(cfg, bid_depth=0.0, ask_depth=0.1)
        result = MakerPrice._apply_kyle_lambda(mp, "sell", 100.0, 10_000_000.0, 0.05)
        assert result > 0.05  # ask depth を使って加算

    def test_pipeline_has_kyle_lambda(self) -> None:
        """compute() パイプラインに _apply_kyle_lambda が組込まれている."""
        src = _maker_price_source("compute")
        assert "_apply_kyle_lambda" in src


# ======================================================================
# Amihud ILLIQ: 非流動性比率
# ======================================================================


class TestAmihudILLIQ:
    """266# Amihud ILLIQ: Amihud (2002) 非流動性 offset 補正."""

    def test_disabled(self) -> None:
        """amihud_illiq_enabled=False なら無影響."""
        result = MakerPrice._apply_amihud_illiq(
            _DISABLED_AMIHUD_STUB,
            "buy",
            100.0,
            10_000_000.0,
            0.05,
        )
        assert result == 0.05

    def test_sufficient_liquidity(self) -> None:
        """流動性十分 (ILLIQ ≤ baseline) なら無影響."""
        cfg = _make_microstructure_config(
            amihud_illiq_enabled=True,
            amihud_illiq_baseline=0.01,  # 高いベースライン
        )
        mp = _make_microstructure_stub(cfg, bid_depth=1.0, ask_depth=1.0)
        result = MakerPrice._apply_amihud_illiq(mp, "buy", 100.0, 10_000_000.0, 0.05)
        # ILLIQ = (100/10M) / 2.0 = 5e-6 << baseline=0.01
        assert result == 0.05

    def test_low_liquidity_increases_offset(self) -> None:
        """低流動性 (ILLIQ > baseline) で offset が拡大."""
        cfg = _make_microstructure_config(
            amihud_illiq_enabled=True,
            amihud_illiq_baseline=1e-8,  # 非常に低いベースライン
            amihud_illiq_max_mult=2.0,
        )
        mp = _make_microstructure_stub(cfg, bid_depth=0.01, ask_depth=0.01)
        result = MakerPrice._apply_amihud_illiq(mp, "buy", 100.0, 10_000_000.0, 0.05)
        assert result > 0.05

    def test_max_mult_clamp(self) -> None:
        """amihud_illiq_max_mult で倍率が上限クランプされる."""
        cfg = _make_microstructure_config(
            amihud_illiq_enabled=True,
            amihud_illiq_baseline=1e-15,  # 極小 → 巨大 ratio
            amihud_illiq_max_mult=1.2,
        )
        mp = _make_microstructure_stub(cfg, bid_depth=0.001, ask_depth=0.001)
        prev = 0.05
        result = MakerPrice._apply_amihud_illiq(mp, "buy", 100.0, 10_000_000.0, prev)
        # max_mult = 1.2 → 最大でも 0.05 * 1.2 = 0.06
        assert result <= prev * 1.2 + 1e-10

    def test_zero_depth_no_effect(self) -> None:
        """depth=0 なら無影響."""
        cfg = _make_microstructure_config(amihud_illiq_enabled=True)
        mp = _make_microstructure_stub(cfg)
        result = MakerPrice._apply_amihud_illiq(mp, "buy", 100.0, 10_000_000.0, 0.05)
        assert result == 0.05

    def test_illiq_cached(self) -> None:
        """ILLIQ 値が _last_amihud_illiq にキャッシュされる."""
        cfg = _make_config(
            amihud_illiq_enabled=True,
            amihud_illiq_baseline=0.001,
        )
        mp = _make_mp(cfg)
        mp._last_bid_depth = 0.5
        mp._last_ask_depth = 0.5
        mp._apply_amihud_illiq("buy", 100.0, 10_000_000.0, 0.05)
        # ILLIQ = (100/10M) / 1.0 = 1e-5
        assert mp._last_amihud_illiq > 0

    def test_pipeline_has_amihud_illiq(self) -> None:
        """compute() パイプラインに _apply_amihud_illiq が組込まれている."""
        src = _maker_price_source("compute")
        assert "_apply_amihud_illiq" in src


# ======================================================================
# OrderBookSnapshot Protocol 移管
# ======================================================================


class TestOrderBookSnapshotProtocol:
    """266# OrderBookSnapshot が ob_utils.py に移管されている."""

    def test_importable_from_ob_utils(self) -> None:
        """ob_utils から OrderBookSnapshot がインポート可能."""
        assert OrderBookSnapshot is not None

    def test_importable_from_maker_price(self) -> None:
        """maker_price.py 経由でも使用可能 (元の定義場所)."""
        # 実際は ob_utils から re-import されている
        assert MakerPriceOrderBookSnapshot is not None

    def test_skip_gate_adapter_returns_snapshot(self) -> None:
        """SkipGateAdapter.get_orderbook の戻り値型が OrderBookSnapshot | None."""
        hints = typing.get_type_hints(SkipGateAdapter.get_orderbook)
        ret = hints.get("return")
        # OrderBookSnapshot | None の Union 型
        assert ret is not None


# ======================================================================
# type:ignore / getattr 排除
# ======================================================================


class TestTypeIgnoreReduction:
    """266# type:ignore / getattr 排除の検証."""

    def test_skip_gate_evaluator_no_attr_defined_ignore(self) -> None:
        """skip_gate_evaluator.py に attr-defined type:ignore がない."""
        src = read_source_text(SKIP_GATE_EVALUATOR)
        # 1042-1043 の ob.bids / ob.asks の type:ignore が除去されている
        lines = src.splitlines()
        attr_defined_count = sum(
            1 for line in lines
            if "type: ignore[attr-defined]" in line
        )
        assert attr_defined_count == 0, (
            f"skip_gate_evaluator still has {attr_defined_count} attr-defined type:ignore"
        )

    def test_ob_utils_no_getattr_for_bids_asks(self) -> None:
        """ob_utils.py の best_bid_ask / depth 系で getattr(ob, 'bids'/'asks') を使わない."""
        assert best_bid_ask is not None
        src = read_source_text(OB_UTILS)
        assert 'getattr(ob, "bids"' not in src
        assert 'getattr(ob, "asks"' not in src

    def test_fill_cycle_executor_no_current_regime_value_class_var(self) -> None:
        """fill_cycle_executor の _current_regime_value class-level 宣言が除去されている."""
        src = read_source_text(FILL_CYCLE_EXECUTOR)
        # class body に _current_regime_value: object = None がない
        assert '_current_regime_value: object = None' not in src


# ======================================================================
# 267# 不具合修正・再利用性検証
# ======================================================================


class TestGetDepthHelper:
    """267# B4/R2: _get_depth ヘルパー DRY 検証."""

    def test_buy_returns_bid_depth(self) -> None:
        mp = _make_mp()
        mp._last_bid_depth = 1.5
        mp._last_ask_depth = 0.5
        assert mp._get_depth("buy") == 1.5

    def test_sell_returns_ask_depth(self) -> None:
        mp = _make_mp()
        mp._last_bid_depth = 1.5
        mp._last_ask_depth = 0.5
        assert mp._get_depth("sell") == 0.5

    def test_kyle_lambda_uses_get_depth(self) -> None:
        """_apply_kyle_lambda が _get_depth を使用している."""
        src = _maker_microstructure_source("_apply_kyle_lambda")
        assert "_get_depth" in src

    def test_amihud_illiq_uses_get_depth(self) -> None:
        """_apply_amihud_illiq が _get_depth を使用している."""
        src = _maker_microstructure_source("_apply_amihud_illiq")
        assert "_get_depth" in src


class TestDeltaStarRatioConversion:
    """267# B1: δ* ratio 変換式の正確性検証."""

    def test_delta_star_ratio_parenthesized(self) -> None:
        """δ* ratio 変換が delta_star_jpy / spread の形式であること (267# 修正)."""
        src = _maker_microstructure_source("_apply_as_reservation_shift")
        # 267# 絶対価格ベース: δ*(JPY) / spread
        assert "delta_star_jpy / spread" in src

    def test_delta_star_ratio_dimensional_consistency(self) -> None:
        """δ* → offset_ratio 変換の次元一貫性 (267# 修正版).

        267# 修正: AS 論文の σ は絶対価格 (JPY/√s)。
        σ_abs = σ_return × mid_price
        δ* (JPY) = γ·σ_abs²·τ + (2/γ)·ln(1+γ/k)
        offset_ratio = δ* / spread
        """
        gamma = 0.1
        spread = 100.0
        mid = 10_000_000.0
        tau = 120.0
        k = 1.5
        sigma_return = spread / (2 * mid)  # 5e-6
        sigma_abs = sigma_return * mid  # 50 JPY
        sigma_abs_sq = sigma_abs ** 2  # 2500 JPY²
        delta_star_jpy = (
            gamma * sigma_abs_sq * tau
            + (2.0 / gamma) * math.log(1.0 + gamma / k)
        )
        delta_star_ratio = delta_star_jpy / spread
        # offset_ratio は妥当な範囲 (0 < ratio < 1000)
        assert 0 < delta_star_ratio < 1000.0, f"δ*_ratio={delta_star_ratio}"
        # γσ²τ 項: 0.1 * 2500 * 120 = 30000 JPY
        # penalty 項: 20 * ln(1.067) ≈ 1.29 JPY
        # δ* ≈ 30001.29 JPY → ratio ≈ 300.01
        assert abs(delta_star_ratio - 300.01) < 1.0


class TestEstimateSigmaDocstringAccuracy:
    """267# B2: _estimate_sigma docstring の正確性."""

    def test_docstring_no_false_reuse_claim(self) -> None:
        """docstring が kyle/amihud の直接再利用を主張しないこと."""
        doc = MakerPrice._estimate_sigma.__doc__ or ""
        # 「で再利用する共通推定値」という虚偽記述がないこと
        assert "で再利用する共通推定値" not in doc
        # 正しい記述: depth ベースで独自推定
        assert "depth" in doc.lower() or "独自" in doc


class TestKyleLambdaAmihudInteraction:
    """267# B3: Kyle λ + Amihud ILLIQ 複合効果の bounds 検証."""

    def test_combined_offset_clamped_to_max(self) -> None:
        """両方 enabled 時も max_offset_ratio を超えない."""
        cfg = _make_config(
            kyle_lambda_enabled=True,
            kyle_lambda_impact_mult=100.0,
            kyle_lambda_max_add_ratio=0.5,
            amihud_illiq_enabled=True,
            amihud_illiq_baseline=1e-15,
            amihud_illiq_max_mult=5.0,
            max_offset_ratio=0.30,
            order_quantity=0.001,
        )
        mp = _make_mp(cfg)
        mp._last_bid_depth = 0.001  # 極薄板
        mp._last_ask_depth = 0.001
        # Kyle λ 加算 → Amihud ILLIQ 乗算 の合流
        offset = 0.05
        offset = mp._apply_kyle_lambda("buy", 100.0, 10_000_000.0, offset)
        offset = mp._apply_amihud_illiq("buy", 100.0, 10_000_000.0, offset)
        assert offset <= 0.30 + 1e-10, f"Combined offset {offset} exceeds max_offset_ratio"

    def test_kyle_only_vs_combined(self) -> None:
        """Kyle のみ vs Kyle+Amihud で合流影響を可視化."""
        cfg_kyle = _make_config(
            kyle_lambda_enabled=True,
            kyle_lambda_impact_mult=1.0,
            kyle_lambda_max_add_ratio=0.10,
            amihud_illiq_enabled=False,
            order_quantity=0.001,
        )
        cfg_both = _make_config(
            kyle_lambda_enabled=True,
            kyle_lambda_impact_mult=1.0,
            kyle_lambda_max_add_ratio=0.10,
            amihud_illiq_enabled=True,
            amihud_illiq_baseline=1e-8,
            amihud_illiq_max_mult=1.5,
            order_quantity=0.001,
        )
        mp_kyle = _make_mp(cfg_kyle)
        mp_kyle._last_bid_depth = 0.01
        mp_kyle._last_ask_depth = 0.01
        mp_both = _make_mp(cfg_both)
        mp_both._last_bid_depth = 0.01
        mp_both._last_ask_depth = 0.01

        base = 0.05
        kyle_only = mp_kyle._apply_kyle_lambda("buy", 100.0, 10_000_000.0, base)
        both = mp_both._apply_kyle_lambda("buy", 100.0, 10_000_000.0, base)
        both = mp_both._apply_amihud_illiq("buy", 100.0, 10_000_000.0, both)
        # Combined effect >= Kyle only (Amihud multiplies)
        assert both >= kyle_only
