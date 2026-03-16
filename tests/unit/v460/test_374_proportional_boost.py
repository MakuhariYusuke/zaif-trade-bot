"""374# Phase 3.1 テスト: SAC Sidecar Proportional Boost.

カバレッジ対象:
  - sidecar_types.py: compute_sidecar_offset_bps_v2, _shaping_fn
  - fill_config.py: sidecar_* フィールド (5 fields)
  - fill_config_parser.py: sidecar YAML セクション解析
  - config_hot_reload.py: sidecar キーの hot-reload 対象登録
  - cycle_gate_aggregator.py: _apply_sidecar_offset v2 切替

375#/376# 安全制約:
  - max_boost_bps ≤ 0.20 (hard ceiling)
  - dead_zone = 0.10
  - shaping = linear (初期)
  - 377# ladder 検証: 0.10 → 0.15 → 0.20 bps step-up
"""

from __future__ import annotations

import math

import pytest

import scripts.v460.lib.sidecar_types as st
from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml
from scripts.v460.lib.fill_config_validation import validate_fill_config
from scripts.v460.lib.sidecar_types import (
    DEFAULT_SIDECAR_BOOST_BPS,
    DEFAULT_SIDECAR_DEAD_ZONE,
    SidecarSignal,
    _shaping_fn,
    compute_sidecar_offset_bps,
    compute_sidecar_offset_bps_v2,
)


# ════════════════════════════════════════════════════════════════
# §1 compute_sidecar_offset_bps_v2 — コア比例計算
# ════════════════════════════════════════════════════════════════


class TestComputeSidecarOffsetBpsV2:
    """374# Phase 3.1: SAC 連続値 → 比例 offset 変換."""

    def _v2(self, **kwargs):
        return compute_sidecar_offset_bps_v2(**kwargs)

    # --- dead zone ---

    def test_dead_zone_zero_bias(self) -> None:
        """bias=0 → dead zone 内 → offset=0."""
        assert self._v2(bias=0.0, side="buy") == 0.0
        assert self._v2(bias=0.0, side="sell") == 0.0

    def test_dead_zone_at_boundary(self) -> None:
        """|bias| == dead_zone → dead zone 内 → offset=0."""
        assert self._v2(bias=0.10, side="buy") == 0.0
        assert self._v2(bias=-0.10, side="sell") == 0.0

    def test_dead_zone_just_above(self) -> None:
        """|bias| > dead_zone → offset > 0 (同方向 side)."""
        offset = self._v2(bias=0.11, side="buy")
        assert offset > 0.0

    def test_dead_zone_custom(self) -> None:
        """dead_zone=0.5 で |bias|=0.4 → offset=0."""
        assert self._v2(bias=0.4, side="buy", dead_zone=0.5) == 0.0
        # |bias|=0.6 > 0.5 → offset > 0
        assert self._v2(bias=0.6, side="buy", dead_zone=0.5) > 0.0

    # --- 方向性 ---

    def test_buy_bias_buy_side_positive(self) -> None:
        """bias > 0 (BUY方向) + buy side → 攻撃的 (+offset)."""
        offset = self._v2(bias=0.8, side="buy")
        assert offset > 0.0

    def test_buy_bias_sell_side_negative(self) -> None:
        """bias > 0 (BUY方向) + sell side → 保守的 (-offset)."""
        offset = self._v2(bias=0.8, side="sell")
        assert offset < 0.0

    def test_sell_bias_sell_side_positive(self) -> None:
        """bias < 0 (SELL方向) + sell side → 攻撃的 (+offset)."""
        offset = self._v2(bias=-0.8, side="sell")
        assert offset > 0.0

    def test_sell_bias_buy_side_negative(self) -> None:
        """bias < 0 (SELL方向) + buy side → 保守的 (-offset)."""
        offset = self._v2(bias=-0.8, side="buy")
        assert offset < 0.0

    # --- 対称性 ---

    def test_symmetry_buy_sell(self) -> None:
        """同じ |bias| で buy/sell の offset は符号反転."""
        buy = self._v2(bias=0.7, side="buy")
        sell = self._v2(bias=0.7, side="sell")
        assert abs(buy + sell) < 1e-12

    def test_symmetry_positive_negative_bias(self) -> None:
        """bias=+x の buy offset == bias=-x の sell offset."""
        pos_buy = self._v2(bias=0.6, side="buy")
        neg_sell = self._v2(bias=-0.6, side="sell")
        assert abs(pos_buy - neg_sell) < 1e-12

    # --- 比例性 (linear) ---

    def test_proportional_linear(self) -> None:
        """linear shaping: offset は |bias| に比例."""
        o1 = self._v2(bias=0.55, side="buy")
        o2 = self._v2(bias=1.0, side="buy")
        # bias=0.55: normalized = (0.55 - 0.10) / 0.90 = 0.5
        # bias=1.0:  normalized = (1.0 - 0.10) / 0.90 = 1.0
        # → o2 / o1 ≈ 2.0
        assert abs(o2 / o1 - 2.0) < 1e-6

    def test_max_output_at_bias_1(self) -> None:
        """bias=±1.0 → output == max_boost_bps (confidence=1)."""
        offset = self._v2(bias=1.0, side="buy", max_boost_bps=0.15)
        assert abs(offset - 0.15) < 1e-12

    # --- confidence ---

    def test_confidence_scaling(self) -> None:
        """confidence は出力に乗算."""
        full = self._v2(bias=0.8, side="buy", confidence=1.0)
        half = self._v2(bias=0.8, side="buy", confidence=0.5)
        assert abs(half / full - 0.5) < 1e-6

    def test_confidence_zero(self) -> None:
        """confidence=0 → offset=0."""
        assert self._v2(bias=0.8, side="buy", confidence=0.0) == 0.0

    def test_confidence_clamped(self) -> None:
        """confidence > 1.0 はクランプ."""
        c1 = self._v2(bias=0.8, side="buy", confidence=1.0)
        c2 = self._v2(bias=0.8, side="buy", confidence=2.0)
        assert abs(c1 - c2) < 1e-12

    def test_confidence_negative_clamped(self) -> None:
        """confidence < 0.0 は 0.0 にクランプ."""
        assert self._v2(bias=0.8, side="buy", confidence=-0.5) == 0.0

    # --- max_boost_bps ---

    def test_max_boost_custom(self) -> None:
        """max_boost_bps=0.10 → bias=1.0 で 0.10."""
        offset = self._v2(bias=1.0, side="buy", max_boost_bps=0.10)
        assert abs(offset - 0.10) < 1e-12

    def test_max_boost_zero(self) -> None:
        """max_boost_bps=0 → 常に 0."""
        assert self._v2(bias=1.0, side="buy", max_boost_bps=0.0) == 0.0

    # --- shaping ---

    def test_shaping_linear(self) -> None:
        """linear: f(0.5) = 0.5."""
        # bias corresponding to normalized=0.5: 0.10 + 0.5*0.9 = 0.55
        offset = self._v2(bias=0.55, side="buy", max_boost_bps=1.0, shaping="linear")
        expected = 0.5  # 1.0 * 0.5
        assert abs(offset - expected) < 1e-6

    def test_shaping_quadratic(self) -> None:
        """quadratic: f(0.5) = 0.25."""
        offset = self._v2(bias=0.55, side="buy", max_boost_bps=1.0, shaping="quadratic")
        expected = 0.25  # 1.0 * 0.5^2
        assert abs(offset - expected) < 1e-6

    def test_shaping_sigmoid(self) -> None:
        """sigmoid: f(0.5) = tanh(1.5) ≈ 0.9051."""
        offset = self._v2(bias=0.55, side="buy", max_boost_bps=1.0, shaping="sigmoid")
        expected = math.tanh(3.0 * 0.5)  # tanh(1.5)
        assert abs(offset - expected) < 1e-6

    def test_shaping_invalid(self) -> None:
        """不正な shaping 値 → ValueError."""
        with pytest.raises(ValueError, match="shaping"):
            self._v2(bias=0.5, side="buy", shaping="cubic")

    # --- edge cases ---

    def test_dead_zone_equals_one(self) -> None:
        """dead_zone=1.0 → always 0 (denominator=0 guard)."""
        assert self._v2(bias=1.0, side="buy", dead_zone=1.0) == 0.0

    def test_bias_just_above_dead_zone_small_output(self) -> None:
        """bias just above dead_zone → very small offset."""
        offset = self._v2(bias=0.101, side="buy", max_boost_bps=0.15)
        assert 0.0 < offset < 0.01  # tiny but positive


# ════════════════════════════════════════════════════════════════
# §2 _shaping_fn — 内部ヘルパー
# ════════════════════════════════════════════════════════════════


class TestShapingFn:
    """_shaping_fn 内部関数テスト."""

    def _fn(self, normalized: float, shaping: str) -> float:
        return _shaping_fn(normalized, shaping)

    def test_linear_endpoints(self) -> None:
        assert self._fn(0.0, "linear") == 0.0
        assert self._fn(1.0, "linear") == 1.0

    def test_quadratic_endpoints(self) -> None:
        assert self._fn(0.0, "quadratic") == 0.0
        assert self._fn(1.0, "quadratic") == 1.0

    def test_sigmoid_endpoints(self) -> None:
        assert self._fn(0.0, "sigmoid") == 0.0
        assert abs(self._fn(1.0, "sigmoid") - math.tanh(3.0)) < 1e-12

    def test_quadratic_less_than_linear(self) -> None:
        """quadratic は linear より小さい (0 < x < 1)."""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert self._fn(x, "quadratic") < self._fn(x, "linear")


# ════════════════════════════════════════════════════════════════
# §3 v1 vs v2 互換性検証
# ════════════════════════════════════════════════════════════════


class TestV1V2Compatibility:
    """v1 と v2 の動作比較."""

    def test_neutral_zone_v1_vs_v2(self) -> None:
        """v1 neutral (|bias| <= 0.3) → v2 dead_zone (|bias| <= 0.1) で動作が異なる.

        bias=0.2: v1=NEUTRAL(0), v2=positive(dead_zone=0.1 内では超えている)
        → v2 はより細かい粒度で offset を適用。
        """
        bias = 0.2
        v1 = compute_sidecar_offset_bps(bias=bias, side="buy")
        v2 = compute_sidecar_offset_bps_v2(bias=bias, side="buy")
        assert v1 == 0.0  # v1: |0.2| <= 0.3 → NEUTRAL
        assert v2 > 0.0   # v2: |0.2| > 0.1 → positive boost

    def test_strong_bias_same_direction(self) -> None:
        """strong bias (|bias|=0.8): 両方とも同方向 offset."""
        v1_buy = compute_sidecar_offset_bps(bias=0.8, side="buy")
        v2_buy = compute_sidecar_offset_bps_v2(bias=0.8, side="buy")
        assert v1_buy > 0.0  # same direction
        assert v2_buy > 0.0  # same direction

    def test_v2_output_bounded(self) -> None:
        """v2 出力は max_boost_bps 以内."""
        for bias in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for side in ["buy", "sell"]:
                offset = compute_sidecar_offset_bps_v2(
                    bias=bias, side=side, max_boost_bps=0.15,
                )
                assert abs(offset) <= 0.15 + 1e-12


# ════════════════════════════════════════════════════════════════
# §4 FillTestConfig sidecar フィールド
# ════════════════════════════════════════════════════════════════


class TestFillConfigSidecarFields:
    """374# sidecar_* フィールドのデフォルト値検証."""

    def test_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.sidecar_enabled is True
        assert cfg.sidecar_max_boost_bps == 0.15
        assert cfg.sidecar_dead_zone == 0.10
        assert cfg.sidecar_shaping == "linear"
        assert cfg.sidecar_use_v2 is True

    def test_custom_values(self) -> None:
        cfg = FillTestConfig(
            sidecar_enabled=False,
            sidecar_max_boost_bps=0.20,
            sidecar_dead_zone=0.05,
            sidecar_shaping="quadratic",
            sidecar_use_v2=False,
        )
        assert cfg.sidecar_enabled is False
        assert cfg.sidecar_max_boost_bps == 0.20
        assert cfg.sidecar_dead_zone == 0.05
        assert cfg.sidecar_shaping == "quadratic"
        assert cfg.sidecar_use_v2 is False


# ════════════════════════════════════════════════════════════════
# §5 YAML parser — sidecar セクション
# ════════════════════════════════════════════════════════════════


class TestFillConfigParserSidecar:
    """374# sidecar YAML セクションの解析テスト."""

    def test_parse_sidecar_section(self) -> None:
        yaml_cfg = {
            "sidecar": {
                "enabled": True,
                "max_boost_bps": 0.10,
                "dead_zone": 0.15,
                "shaping": "sigmoid",
                "use_v2": True,
            }
        }
        cfg = parse_fill_config_yaml(yaml_cfg)
        assert cfg.sidecar_enabled is True
        assert cfg.sidecar_max_boost_bps == 0.10
        assert cfg.sidecar_dead_zone == 0.15
        assert cfg.sidecar_shaping == "sigmoid"
        assert cfg.sidecar_use_v2 is True

    def test_parse_sidecar_partial(self) -> None:
        """部分指定 — 未指定フィールドはデフォルト値."""
        yaml_cfg = {"sidecar": {"max_boost_bps": 0.20}}
        cfg = parse_fill_config_yaml(yaml_cfg)
        assert cfg.sidecar_max_boost_bps == 0.20
        assert cfg.sidecar_dead_zone == 0.10  # default
        assert cfg.sidecar_shaping == "linear"  # default

    def test_parse_sidecar_absent(self) -> None:
        """sidecar セクション未指定 → 全デフォルト."""
        cfg = parse_fill_config_yaml({})
        assert cfg.sidecar_enabled is True
        assert cfg.sidecar_max_boost_bps == 0.15

    def test_parse_sidecar_disabled(self) -> None:
        yaml_cfg = {"sidecar": {"enabled": False}}
        cfg = parse_fill_config_yaml(yaml_cfg)
        assert cfg.sidecar_enabled is False


# ════════════════════════════════════════════════════════════════
# §6 config_hot_reload — sidecar キーの登録確認
# ════════════════════════════════════════════════════════════════


class TestConfigHotReloadSidecar:
    """374# sidecar キーが hot-reload 対象に登録されていること."""

    def test_sidecar_keys_in_hot_reloadable(self) -> None:
        expected = {
            "sidecar_enabled",
            "sidecar_max_boost_bps",
            "sidecar_dead_zone",
            "sidecar_shaping",
            "sidecar_use_v2",
        }
        assert expected.issubset(_HOT_RELOADABLE_FIELDS)


# ════════════════════════════════════════════════════════════════
# §7 cycle_gate_aggregator — v2 切替テスト
# ════════════════════════════════════════════════════════════════


class TestCycleGateAggregatorSidecarV2:
    """374# _apply_sidecar_offset が v2 を正しく呼び出すこと."""

    def _make_signal(self, bias: float = 0.5, confidence: float = 1.0):
        return SidecarSignal(
            timestamp="2026-03-15T12:00:00+09:00",
            directional_bias=bias,
            confidence=confidence,
        )

    def _make_aggregator(self, **config_overrides):
        cfg = FillTestConfig(**config_overrides)
        return CycleGateAggregator(cfg)

    def _make_result(self):
        return CycleGateResult(blocked=False)

    def test_v2_produces_proportional_offset(self) -> None:
        """sidecar_use_v2=True → 比例オフセット."""
        agg = self._make_aggregator(sidecar_use_v2=True)
        result = self._make_result()
        signal = self._make_signal(bias=0.5)

        agg._apply_sidecar_offset(result, signal, "buy")

        # bias=0.5, dead_zone=0.10, normalized=(0.5-0.1)/0.9≈0.444
        # offset = 0.15 * 0.444 ≈ 0.0667
        assert result.sidecar_offset_bps > 0.0
        assert abs(result.sidecar_offset_bps) < 0.15

    def test_v1_fallback(self) -> None:
        """sidecar_use_v2=False → v1 (離散分類) にフォールバック."""
        agg = self._make_aggregator(sidecar_use_v2=False)
        result = self._make_result()
        signal = self._make_signal(bias=0.5)

        agg._apply_sidecar_offset(result, signal, "buy")

        # v1: bias=0.5 > 0.3 → BUY_BIAS → boost=0.15
        assert result.sidecar_offset_bps == 0.15

    def test_sidecar_disabled(self) -> None:
        """sidecar_enabled=False → offset 不変."""
        agg = self._make_aggregator(sidecar_enabled=False)
        result = self._make_result()
        signal = self._make_signal(bias=0.8)

        agg._apply_sidecar_offset(result, signal, "buy")

        assert result.sidecar_offset_bps == 0.0

    def test_v2_dead_zone_neutral(self) -> None:
        """v2: bias within dead zone → offset=0."""
        agg = self._make_aggregator(sidecar_use_v2=True)
        result = self._make_result()
        signal = self._make_signal(bias=0.05)

        agg._apply_sidecar_offset(result, signal, "buy")

        assert result.sidecar_offset_bps == 0.0

    def test_v2_config_max_boost_respected(self) -> None:
        """max_boost_bps が config から読まれること."""
        agg = self._make_aggregator(
            sidecar_use_v2=True,
            sidecar_max_boost_bps=0.10,
        )
        result = self._make_result()
        signal = self._make_signal(bias=1.0)

        agg._apply_sidecar_offset(result, signal, "buy")

        assert abs(result.sidecar_offset_bps - 0.10) < 1e-12

    def test_v2_metadata_set(self) -> None:
        """sidecar_bias, sidecar_direction が設定される."""
        agg = self._make_aggregator(sidecar_use_v2=True)
        result = self._make_result()
        signal = self._make_signal(bias=0.7)

        agg._apply_sidecar_offset(result, signal, "buy")

        assert result.sidecar_bias == 0.7
        assert result.sidecar_direction == "buy_bias"

    def test_v2_confidence_affects_output(self) -> None:
        """confidence は v2 でも出力に乗算される."""
        agg = self._make_aggregator(sidecar_use_v2=True)
        r1 = self._make_result()
        r2 = self._make_result()

        agg._apply_sidecar_offset(r1, self._make_signal(bias=0.8, confidence=1.0), "buy")
        agg._apply_sidecar_offset(r2, self._make_signal(bias=0.8, confidence=0.5), "buy")

        assert abs(r2.sidecar_offset_bps / r1.sidecar_offset_bps - 0.5) < 1e-6


# ════════════════════════════════════════════════════════════════
# §8 安全検証 — 375#/376# hard ceiling
# ════════════════════════════════════════════════════════════════


class TestSafetyBounds:
    """375#/376# 安全制約の検証."""

    def test_default_max_boost_within_ceiling(self) -> None:
        """DEFAULT_SIDECAR_BOOST_BPS ≤ 0.20."""
        assert DEFAULT_SIDECAR_BOOST_BPS <= 0.20

    def test_config_default_within_ceiling(self) -> None:
        """FillTestConfig.sidecar_max_boost_bps ≤ 0.20."""
        cfg = FillTestConfig()
        assert cfg.sidecar_max_boost_bps <= 0.20

    def test_v2_output_never_exceeds_max_boost(self) -> None:
        """全 bias/side/shaping 組合せで |offset| ≤ max_boost_bps."""
        max_bps = 0.15
        for bias in [-1.0, -0.8, -0.5, -0.11, 0.0, 0.11, 0.5, 0.8, 1.0]:
            for side in ["buy", "sell"]:
                for shaping in ["linear", "quadratic", "sigmoid"]:
                    offset = compute_sidecar_offset_bps_v2(
                        bias=bias, side=side,
                        max_boost_bps=max_bps, shaping=shaping,
                    )
                    assert abs(offset) <= max_bps + 1e-12, (
                        f"Exceeded: bias={bias}, side={side}, "
                        f"shaping={shaping}, offset={offset}"
                    )

    def test_dead_zone_default(self) -> None:
        """DEFAULT_SIDECAR_DEAD_ZONE == 0.10."""
        assert DEFAULT_SIDECAR_DEAD_ZONE == 0.10


# ════════════════════════════════════════════════════════════════
# §9 377# Ladder 検証 — step-up パラメータ
# ════════════════════════════════════════════════════════════════


class TestLadderSteps:
    """377# ladder 検証のパラメータバリエーション."""

    @pytest.mark.parametrize("max_boost", [0.10, 0.15, 0.20])
    def test_ladder_step_bounded(self, max_boost: float) -> None:
        """各 ladder step で |offset| ≤ max_boost."""
        for bias in [-1.0, 1.0]:
            offset = compute_sidecar_offset_bps_v2(
                bias=bias, side="buy", max_boost_bps=max_boost,
            )
            assert abs(offset) <= max_boost + 1e-12

    @pytest.mark.parametrize("max_boost", [0.10, 0.15, 0.20])
    def test_ladder_step_monotonic(self, max_boost: float) -> None:
        """|bias| 増加 → |offset| 単調増加 (linear)."""
        prev = 0.0
        for bias_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
            offset = compute_sidecar_offset_bps_v2(
                bias=bias_val, side="buy", max_boost_bps=max_boost,
            )
            assert offset >= prev
            prev = offset


# ════════════════════════════════════════════════════════════════
# §10 fill_config_validation — sidecar バリデーション横展開テスト
# ════════════════════════════════════════════════════════════════


class TestSidecarValidation:
    """374# 横展開: fill_config_validation.py の sidecar フィールド検証."""

    def _make_config(self, **overrides):
        """最小限の FillTestConfig を生成 (validation 通過するデフォルト)."""
        cfg = FillTestConfig()
        for k, v in overrides.items():
            object.__setattr__(cfg, k, v)
        return cfg

    def test_max_boost_bps_negative_raises(self) -> None:
        """sidecar_max_boost_bps < 0 → ValueError."""
        cfg = self._make_config(sidecar_max_boost_bps=-0.01)
        with pytest.raises(ValueError, match="sidecar_max_boost_bps must be >= 0"):
            validate_fill_config(cfg)

    def test_max_boost_bps_exceeds_ceiling_raises(self) -> None:
        """sidecar_max_boost_bps > 0.20 → ValueError (375#/376# hard ceiling)."""
        cfg = self._make_config(sidecar_max_boost_bps=0.21)
        with pytest.raises(ValueError, match="hard ceiling"):
            validate_fill_config(cfg)

    def test_max_boost_bps_at_ceiling_ok(self) -> None:
        """sidecar_max_boost_bps == 0.20 → 通過."""
        cfg = self._make_config(sidecar_max_boost_bps=0.20)
        # Should not raise
        validate_fill_config(cfg)

    def test_dead_zone_negative_raises(self) -> None:
        """sidecar_dead_zone < 0 → ValueError."""
        cfg = self._make_config(sidecar_dead_zone=-0.1)
        with pytest.raises(ValueError, match="sidecar_dead_zone must be in"):
            validate_fill_config(cfg)

    def test_dead_zone_one_raises(self) -> None:
        """sidecar_dead_zone >= 1.0 → ValueError."""
        cfg = self._make_config(sidecar_dead_zone=1.0)
        with pytest.raises(ValueError, match="sidecar_dead_zone must be in"):
            validate_fill_config(cfg)

    def test_dead_zone_valid_range_ok(self) -> None:
        """sidecar_dead_zone in [0, 1) → 通過."""
        cfg = self._make_config(sidecar_dead_zone=0.5)
        validate_fill_config(cfg)

    def test_shaping_invalid_raises(self) -> None:
        """sidecar_shaping が不正文字列 → ValueError."""
        cfg = self._make_config(sidecar_shaping="cubic")
        with pytest.raises(ValueError, match="sidecar_shaping must be one of"):
            validate_fill_config(cfg)

    @pytest.mark.parametrize("shaping", ["linear", "quadratic", "sigmoid"])
    def test_shaping_valid_ok(self, shaping: str) -> None:
        """有効な shaping 値 → 通過."""
        cfg = self._make_config(sidecar_shaping=shaping)
        validate_fill_config(cfg)

    def test_math_module_level_import(self) -> None:
        """sidecar_types.py の math が module-level import であることを確認."""
        assert hasattr(st, "math"), "math should be a module-level import"
        assert st.math is math, "st.math should be the same math module"
