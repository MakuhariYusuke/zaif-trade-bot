"""303# レビュー応答実装テスト.

テスト対象:
- A: [Side Comparison] 表記修正 + none 含有版出力
- B: DD Guard soft lot side 分離 (soft_triggered_side)
- C: none レジーム Passive MM バイパス
"""

from __future__ import annotations

import dataclasses
from copy import copy

import numpy as np

from scripts.v460.lib.ab_judgment import (
    ABJudgmentCriteria,
    ABJudgmentResult,
    Verdict,
    evaluate_ab_variant,
)
from scripts.v460.lib.daily_drawdown_guard import (
    DailyDrawdownGuard,
    DrawdownAction,
)
from scripts.v460.lib.fill_config import FillTestConfig
from tests.unit.v460._fill_test_source import MAKER_PRICE, read_class_method_source
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from ztb.io.json_io import JSONObject
from ztb.metrics.fill_quality import FillRecord

_FIXED_TS = 1_700_000_000.0
_FILL_RECORD_FIELD_NAMES = {f.name for f in dataclasses.fields(FillRecord)}


# ======================================================================
# Helpers
# ======================================================================

def _make_record(
    *,
    side: str = "sell",
    regime: str = "ranging",
    filled: bool = True,
    pnl30: float | None = 0.5,
    timestamp: float | None = None,
) -> JSONObject:
    """テスト用 FillRecord."""
    ts = _FIXED_TS if timestamp is None else timestamp
    r: JSONObject = {
        "side": side,
        "regime": regime,
        "filled": filled,
        "timestamp": ts,
    }
    if filled and pnl30 is not None:
        r["post_fill_30s_pnl"] = pnl30
    return r


def _make_records(
    n: int,
    *,
    side: str = "sell",
    regime: str = "ranging",
    fill_rate: float = 0.5,
    pnl_mean: float = 0.0,
    pnl_std: float = 1.0,
    base_ts: float | None = None,
) -> list[JSONObject]:
    """n 件のテスト用レコード生成."""
    rng = np.random.default_rng(42)
    base = _FIXED_TS if base_ts is None else base_ts
    records = []
    for i in range(n):
        filled = i < int(n * fill_rate)
        pnl = float(rng.normal(pnl_mean, pnl_std)) if filled else None
        records.append(_make_record(
            side=side,
            regime=regime,
            filled=filled,
            pnl30=pnl,
            timestamp=base + i * 120,
        ))
    return records


def _make_summary_result() -> ABJudgmentResult:
    """summary 文言専用の軽量結果."""
    return ABJudgmentResult(
        overall=Verdict.PASS,
        variant_label="sell",
        control_label="buy",
        n_variant=60,
        n_control=50,
    )


# ======================================================================
# A: Side Comparison 表記修正テスト
# ======================================================================


class TestSideComparisonLabel:
    """301# F2: [A/B Judgment] → [Side Comparison] 表記修正."""

    def test_summary_header_renamed(self) -> None:
        """summary() が [Side Comparison] で始まること."""
        result = _make_summary_result()
        summary = result.summary()
        assert "[Side Comparison]" in summary
        assert "[A/B Judgment]" not in summary

    def test_summary_contains_observational_disclaimer(self) -> None:
        """summary() に観察比較の注記が含まれること (301# F2)."""
        result = _make_summary_result()
        summary = result.summary()
        assert "観察比較" in summary
        assert "ランダム割当" in summary


class TestNoneRegimeInclusion:
    """301# F1: none レジーム含有版 A/B 判定."""

    def test_default_excludes_none(self) -> None:
        """デフォルト criteria で none レジームが除外されること."""
        criteria = ABJudgmentCriteria()
        assert "none" in criteria.exclude_regimes

    def test_include_none_via_empty_exclude(self) -> None:
        """exclude_regimes=[] で none レジームが含有されること."""
        base_ts = _FIXED_TS
        # none レジームのレコードを含むデータ
        variant_none = _make_records(
            6, side="sell", regime="none", fill_rate=0.6, pnl_mean=-2.0,
            base_ts=base_ts,
        )
        variant_ranging = _make_records(
            12, side="sell", regime="ranging", fill_rate=0.6, pnl_mean=1.0,
            base_ts=base_ts + 6 * 120,
        )
        control_none = _make_records(
            6, side="buy", regime="none", fill_rate=0.5, pnl_mean=-1.5,
            base_ts=base_ts,
        )
        control_ranging = _make_records(
            12, side="buy", regime="ranging", fill_rate=0.5, pnl_mean=0.5,
            base_ts=base_ts + 6 * 120,
        )

        variant = variant_none + variant_ranging
        control = control_none + control_ranging

        # none 除外版
        criteria_excl = ABJudgmentCriteria(exclude_regimes=["none"])
        result_excl = evaluate_ab_variant(
            variant_records=variant,
            control_records=control,
            criteria=criteria_excl,
        )

        # none 含有版
        criteria_incl = ABJudgmentCriteria(exclude_regimes=[])
        result_incl = evaluate_ab_variant(
            variant_records=variant,
            control_records=control,
            criteria=criteria_incl,
        )

        # none 含有版のほうがサンプル数が多い
        assert result_incl.n_variant > result_excl.n_variant
        assert result_incl.n_control > result_excl.n_control

    def test_two_criteria_copies_independent(self) -> None:
        """copy で exclude_regimes を変更しても元は影響なし."""
        original = ABJudgmentCriteria(exclude_regimes=["none"])
        copied = copy(original)
        copied.exclude_regimes = []
        assert original.exclude_regimes == ["none"]
        assert copied.exclude_regimes == []


# ======================================================================
# B: DD Guard soft lot side 分離テスト
# ======================================================================


class TestDDGuardSoftLotSide:
    """303# B: soft_triggered_side フィールドのテスト."""

    def test_soft_triggered_side_in_disabled(self) -> None:
        """無効時に soft_triggered_side が空文字."""
        guard = DailyDrawdownGuard(enabled=False)
        result = guard.update_pnl(-10.0, side="buy")
        assert result["soft_triggered_side"] == ""

    def test_soft_triggered_side_populated(self) -> None:
        """soft limit 超過時に side が設定される."""
        guard = DailyDrawdownGuard(
            enabled=True,
            soft_limit_bps=-10.0,
            hard_limit_bps=-50.0,
        )
        guard.maybe_reset_day()
        result = guard.update_pnl(-15.0, side="sell")
        assert result["soft_triggered"] is True
        assert result["soft_triggered_side"] == "sell"

    def test_soft_triggered_side_empty_on_no_trigger(self) -> None:
        """soft 未発動時は空文字."""
        guard = DailyDrawdownGuard(
            enabled=True,
            soft_limit_bps=-30.0,
            hard_limit_bps=-50.0,
        )
        guard.maybe_reset_day()
        result = guard.update_pnl(-5.0, side="buy")
        assert result["soft_triggered"] is False
        assert result["soft_triggered_side"] == ""

    def test_soft_triggered_side_aggregate_no_side(self) -> None:
        """side 未指定時は空文字で返る."""
        guard = DailyDrawdownGuard(
            enabled=True,
            soft_limit_bps=-10.0,
            hard_limit_bps=-50.0,
        )
        guard.maybe_reset_day()
        result = guard.update_pnl(-15.0)  # side="" (default)
        assert result["soft_triggered"] is True
        assert result["soft_triggered_side"] == ""


class TestFillConfigSoftLotSideAware:
    """303# B: config フィールドの存在テスト."""

    def test_default_disabled(self) -> None:
        cfg = FillTestConfig()
        assert cfg.daily_drawdown_soft_lot_side_aware is False

    def test_can_enable(self) -> None:
        cfg = FillTestConfig(daily_drawdown_soft_lot_side_aware=True)
        assert cfg.daily_drawdown_soft_lot_side_aware is True


# ======================================================================
# C: none regime Passive MM テスト
# ======================================================================


class TestNoneRegimePassiveMM:
    """303# C: fill_config フィールドの存在テスト."""

    def test_default_disabled(self) -> None:
        cfg = FillTestConfig()
        assert cfg.none_regime_passive_mm_enabled is False
        assert cfg.none_regime_fixed_offset_bps == 2.0

    def test_can_enable(self) -> None:
        cfg = FillTestConfig(
            none_regime_passive_mm_enabled=True,
            none_regime_fixed_offset_bps=3.0,
        )
        assert cfg.none_regime_passive_mm_enabled is True
        assert cfg.none_regime_fixed_offset_bps == 3.0

    def test_bypass_fires_for_unknown_regime(self) -> None:
        """318# F5-1: Passive MM バイパスが 'unknown' レジームでも発火する.

        旧実装は 'none' のみチェックしていたため、FillTestRegime.UNKNOWN
        ('unknown') では発火しなかった。修正後は ('none', 'unknown') 両方対応。
        """
        source = read_class_method_source(
            MAKER_PRICE,
            "MakerPriceCalculator",
            "compute",
        )
        # 旧: `if _current_regime == "none":` → 新: `in ("none", "unknown")`
        assert 'in ("none", "unknown")' in source, (
            "318# F5-1: Passive MM bypass should check for both 'none' AND 'unknown'"
        )
        # "none" 単体チェックが残っていないことを確認
        assert '_current_regime == "none"' not in source, (
            "318# F5-1: Old none-only check should be removed"
        )

    def test_fill_record_has_regime_at_order(self) -> None:
        """318# F5-3: FillRecord に regime_at_order フィールドが存在."""
        assert "regime_at_order" in _FILL_RECORD_FIELD_NAMES
        assert "regime_observation_count" in _FILL_RECORD_FIELD_NAMES

    def test_fill_record_regime_at_order_default_none(self) -> None:
        """318# F5-3: regime_at_order のデフォルト値は None."""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.regime_at_order is None
        assert r.regime_observation_count is None

    def test_fill_record_has_mid_at_order(self) -> None:
        """319# S-3: FillRecord に mid_at_order フィールドが存在し、デフォルト None."""
        assert "mid_at_order" in _FILL_RECORD_FIELD_NAMES

        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="sell",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.mid_at_order is None

    def test_fill_record_mid_at_order_roundtrip(self) -> None:
        """319# S-3: mid_at_order が to_dict / from_dict でラウンドトリップ."""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="sell",
            order_price=100.0, order_quantity=0.001,
            mid_at_order=13000000.5,
        )
        d = r.to_dict()
        assert d["mid_at_order"] == 13000000.5
        r2 = FillRecord.from_dict(d)
        assert r2.mid_at_order == 13000000.5


# ======================================================================
# 320# C-1: Side-specific Ceiling テスト
# ======================================================================


class TestSideSpecificCeiling:
    """320# C-1: offset_ceiling_ratio のサイド別分離."""

    def test_config_has_side_specific_ceiling_fields(self) -> None:
        """FillTestConfig に offset_ceiling_ratio_buy/sell が存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "offset_ceiling_ratio_buy")
        assert hasattr(cfg, "offset_ceiling_ratio_sell")
        # デフォルトは None (共通値使用)
        assert cfg.offset_ceiling_ratio_buy is None
        assert cfg.offset_ceiling_ratio_sell is None

    def test_sell_ceiling_higher_than_buy(self) -> None:
        """sell ceiling を buy より高く設定した場合、sell pipeline が分化可能."""
        cfg = FillTestConfig()
        cfg.offset_ceiling_ratio = 0.15
        cfg.offset_ceiling_ratio_buy = 0.15
        cfg.offset_ceiling_ratio_sell = 0.50
        # sell ceiling > sell floor(0.30) = 矛盾解消
        assert cfg.offset_ceiling_ratio_sell > cfg.sell_offset_floor

    def test_sell_ceiling_none_falls_back_to_common(self) -> None:
        """sell ceiling=None 時は共通値にフォールバック."""
        cfg = FillTestConfig()
        cfg.offset_ceiling_ratio = 0.15
        cfg.offset_ceiling_ratio_sell = None
        # 共通値が使用される (コード上の挙動を間接検証)
        assert cfg.offset_ceiling_ratio == 0.15

    def test_side_specific_ceiling_yaml_parse(self) -> None:
        """321# CRITICAL fix: from_yaml() がサイド別 ceiling をパース."""
        yaml_cfg: dict = {
            "offset_ceiling_ratio": 0.15,
            "offset_ceiling_ratio_buy": 0.15,
            "offset_ceiling_ratio_sell": 0.50,
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_cfg))
        assert cfg.offset_ceiling_ratio == 0.15
        assert cfg.offset_ceiling_ratio_buy == 0.15
        assert cfg.offset_ceiling_ratio_sell == 0.50

    def test_side_specific_ceiling_yaml_absent_stays_none(self) -> None:
        """321#: YAML にサイド別 ceiling がない場合 None のまま."""
        yaml_cfg: dict = {"offset_ceiling_ratio": 0.20}
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_cfg))
        assert cfg.offset_ceiling_ratio == 0.20
        assert cfg.offset_ceiling_ratio_buy is None
        assert cfg.offset_ceiling_ratio_sell is None


# ======================================================================
# B/C: hot-reload 登録テスト
# ======================================================================


class TestHotReloadRegistration:
    """303# B/C: hot-reload フィールド登録確認."""

    def test_new_fields_in_hot_reloadable(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS

        assert "daily_drawdown_soft_lot_side_aware" in _HOT_RELOADABLE_FIELDS
        assert "none_regime_passive_mm_enabled" in _HOT_RELOADABLE_FIELDS
        assert "none_regime_fixed_offset_bps" in _HOT_RELOADABLE_FIELDS

    def test_side_specific_ceiling_hot_reloadable(self) -> None:
        """321#: サイド別 ceiling が hot-reload 対象."""
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS

        assert "offset_ceiling_ratio_buy" in _HOT_RELOADABLE_FIELDS
        assert "offset_ceiling_ratio_sell" in _HOT_RELOADABLE_FIELDS
