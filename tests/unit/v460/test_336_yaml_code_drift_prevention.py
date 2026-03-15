"""336# YAML↔Code default drift prevention test.

336# で 12 箇所のドリフトが発見された反省を踏まえ、
FillTestConfig() のコードデフォルトと configs/v460/fill_test.yaml の
YAML 値の乖離を監視するテスト。

設計思想:
- YAML は本番稼働の権威的設定 (authoritative config)
- コードデフォルトはテスト互換・フォールバック値
- 許容される差分: YAML が意図的にオーバーライドしているフィールド (KNOWN_YAML_OVERRIDES)
- 検出対象: KNOWN_YAML_OVERRIDES に含まれないのに値が異なるフィールド (=ドリフト)

使い方:
- 新フィールド追加時: YAML で値を設定し、コードデフォルトも同じ値にする
  → テスト通過
- YAML オーバーライド追加時: KNOWN_YAML_OVERRIDES にフィールド名を追加
  → テスト通過
- コードデフォルトを変更して YAML 更新を忘れた場合:
  → テスト失敗 (ドリフト検出)
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path

from scripts.v460.lib.fill_config import FillTestConfig
from tests.unit.v460._yaml_test_helpers import load_yaml_mapping

_YAML_PATH = Path(__file__).resolve().parents[3] / "configs" / "v460" / "fill_test.yaml"

# ======================================================================
# これらのフィールドは YAML が意図的にコードデフォルトと異なる値を設定している。
# 新しい YAML オーバーライドを追加する際はここにも追加すること。
# フィールドがこのセットに含まれていて差分がなくなった場合もテストは失敗する
# (allowlist の整理を促すため)。
# ======================================================================
KNOWN_YAML_OVERRIDES: frozenset[str] = frozenset({
    # --- 機能有効化 (code=False, YAML=True) ---
    "amihud_illiq_enabled",
    "as_delta_star_enabled",
    "as_reservation_enabled",
    "as_tau_dynamic_enabled",
    "bayesian_regime_enabled",
    "buy_as_guard_enabled",
    "buy_dynamic_kill_enabled",
    "buy_dynamic_kill_inv_relaxation_enabled",
    "buy_velocity_skip_enabled",
    "daily_drawdown_enabled",
    "daily_drawdown_soft_lot_side_aware",
    "dual_kill_quiescence_enabled",
    "dynamic_cycle_interval_enabled",
    "enable_macro_regime",
    "enable_time_filter",
    "fast_fill_defense_enabled",
    "inv_skew_regime_gate_enabled",
    "inventory_skewing_enabled",
    "kyle_lambda_enabled",
    "loss_cap_auto",
    "low_vol_boost_proportional",
    "low_vol_offset_boost_enabled",
    "none_regime_passive_mm_enabled",
    "offset_stage_recording_enabled",
    "per_side_dd_enabled",
    "queue_position_tracking_enabled",
    "regime_adaptive_enabled",
    "sell_dynamic_kill_enabled",
    "sell_dynamic_kill_inv_relaxation_enabled",
    "sell_velocity_skip_enabled",
    "sigma_parkinson_enabled",
    "skip_buy_unknown_regime",
    "skip_gate_adaptive_threshold",
    "skip_gate_enabled",
    "skip_gate_ev_as_offset_enabled",
    "skip_gate_ev_weighted_enabled",
    "skip_gate_use_ob_features",
    "skip_ranging_buy_low_vol",
    "skip_sell_trending",
    "skip_sell_trending_up_only",
    "skip_sell_unknown_regime",
    "spread_adaptive_enabled",
    "stale_order_enabled",
    "trending_sell_as_offset_enabled",
    "velocity_offset_proportional",
    "velocity_skip_as_offset_enabled",
    "vg_inv_skew_damping_enabled",
    "vg_vpin_buy_extra_mult",
    "vg_vpin_continuous_enabled",
    "volatility_guard_enabled",
    # --- 数値パラメータのチューニング (YAML ≠ code default) ---
    "adapt_recency_window",
    "buy_dynamic_kill_regime_thresholds",
    "buy_dynamic_kill_ewma_time_decay_tau_sec",
    "buy_dynamic_kill_max_duration_sec",
    "buy_velocity_skip_threshold_bps",
    "dd_cooldown_rearm_budget_bps",
    "dd_cooldown_release_sec",
    "e3_sampling_ratio",
    "execution_final_clamp_hard_skip_mult",  # 421# P0: Execution Final Clamp
    "fast_fill_offset_boost_sell",
    "fast_fill_threshold_sec_buy",
    "fast_fill_threshold_sec_sell",
    "glft_dynamic_k_enabled",
    "hard_skip_utc_hours",
    "loss_boost_offset_mult",
    "max_consecutive_trending_sell_skip",
    "min_spread_jpy",
    "narrow_spread_boost_buy",
    "narrow_spread_boost_sell",
    "narrow_spread_bps",
    "none_regime_fixed_offset_bps",
    "offset_ceiling_ratio",
    "offset_ceiling_ratio_buy",
    "offset_ceiling_ratio_sell",
    "order_timeout_sec_sell",
    "post_fill_wait_sec_sell",
    "regime_adaptive_extra_buy",
    "regime_adaptive_extra_sell",
    "regime_mid_confidence_offset_boost",   # 397# mid-confidence paradox guard
    "regime_ranging_offset_discount",
    "regime_trending_offset_boost_buy",
    "regime_trending_offset_boost_sell",
    "sell_dynamic_kill_regime_thresholds",
    "sell_dynamic_kill_ewma_time_decay_tau_sec",
    "sell_hour_offset_boost",
    "sell_max_spread_jpy",
    "sell_offset_floor",
    "sell_velocity_skip_threshold_bps",
    "sigma_clustering_enabled",
    "vg_vpin_buy_extra_mult",
    "skip_gate_as_threshold",
    "skip_gate_as_threshold_buy",
    "skip_gate_as_threshold_sell",
    "skip_gate_ev_max_consecutive_skip",
    "skip_gate_ev_one_sided_threshold_shift",
    "skip_gate_hour_offsets",
    "skip_gate_mode",
    "skip_gate_model_path",
    "skip_gate_model_path_buy",
    "skip_gate_model_path_buy_long",
    "skip_gate_model_path_sell",
    "skip_gate_model_path_sell_short",
    "skip_gate_narrow_spread_offset",
    "skip_gate_narrow_spread_threshold_jpy",
    "skip_gate_pnl_threshold",
    "skip_gate_regime_thresholds",
    "skip_gate_target_skip_rate_buy",
    "skip_utc_hours",
    "skip_utc_hours_buy",
    "skip_utc_hours_sell",
    "spread_offset_ratio_sell",
    "stale_check_after_sec_buy",
    "stale_check_after_sec_sell",
    "stale_drift_bps_buy",
    "stale_drift_bps_sell",
    "stale_max_reprice_buy",
    "stale_max_reprice_sell",
    "stale_reprice_min_delta_jpy",
    "stale_reprice_skip_gate_offset",
    "stale_reprice_tighten",
    "toxic_fill_veto_threshold_bps",
    "trending_down_buy_offset_boost",
    "trending_down_sell_offset_boost",
    "trending_up_buy_offset_boost",
    "trending_up_sell_offset_boost",
    "unknown_buy_offset_boost",
    "volatility_guard_velocity_threshold_bps",
    "volatility_guard_vpin_threshold",
    "vpin_vol_sync_enabled",
    "wide_spread_bps",
})


@lru_cache(maxsize=1)
def _load_yaml_config() -> FillTestConfig:
    return FillTestConfig.from_yaml(load_yaml_mapping(_YAML_PATH))


@lru_cache(maxsize=1)
def _load_code_config() -> FillTestConfig:
    return FillTestConfig()


class TestYamlCodeDefaultDrift:
    """336# YAML↔Code デフォルト値ドリフト検出."""

    def _load_yaml_config(self) -> FillTestConfig:
        return _load_yaml_config()

    def test_no_unexpected_drift(self) -> None:
        """KNOWN_YAML_OVERRIDES 外のフィールドでドリフトが無いことを検証.

        このテストが失敗した場合:
        1. コードデフォルトを YAML に合わせて修正する (推奨)
        2. または KNOWN_YAML_OVERRIDES に追加する (意図的オーバーライドの場合)
        """
        from_yaml = self._load_yaml_config()
        from_code = _load_code_config()

        drifted: list[str] = []
        for fld in dataclasses.fields(from_yaml):
            if fld.name in KNOWN_YAML_OVERRIDES:
                continue
            yaml_val = getattr(from_yaml, fld.name)
            code_val = getattr(from_code, fld.name)
            if yaml_val != code_val:
                drifted.append(
                    f"  {fld.name}: code={code_val!r} yaml={yaml_val!r}"
                )

        assert not drifted, (
            "YAML↔Code ドリフト検出！以下のフィールドで値が異なります:\n"
            + "\n".join(drifted)
            + "\n\n修正方法: コードデフォルト値を YAML 値に合わせるか、"
            + " KNOWN_YAML_OVERRIDES に追加してください。"
        )

    def test_allowlist_is_clean(self) -> None:
        """KNOWN_YAML_OVERRIDES に含まれるが実際には差分がないフィールドを検出.

        allowlist が肥大化して「ドリフト隠し」にならないよう、
        コードデフォルトが YAML と一致したフィールドの除去を促す。
        """
        from_yaml = self._load_yaml_config()
        from_code = _load_code_config()

        stale: list[str] = []
        for name in KNOWN_YAML_OVERRIDES:
            yaml_val = getattr(from_yaml, name, _SENTINEL)
            code_val = getattr(from_code, name, _SENTINEL)
            if yaml_val is _SENTINEL or code_val is _SENTINEL:
                stale.append(f"  {name}: フィールドが存在しません")
            elif yaml_val == code_val:
                stale.append(
                    f"  {name}: 値が一致 ({code_val!r}) → allowlist から除去可"
                )

        assert not stale, (
            "KNOWN_YAML_OVERRIDES の整理が必要です:\n"
            + "\n".join(stale)
            + "\n\n修正方法: 一致しているフィールドを KNOWN_YAML_OVERRIDES から除去してください。"
        )

    def test_yaml_file_exists(self) -> None:
        """テスト前提: YAML ファイルが存在すること."""
        assert _YAML_PATH.exists(), f"YAML not found: {_YAML_PATH}"

    def test_field_count_sanity(self) -> None:
        """FillTestConfig のフィールド数が大幅に変化していないこと (God Object 監視)."""
        n_fields = len(dataclasses.fields(FillTestConfig))
        # 336# 時点: 390 fields. ±20 は許容、それ以上は要レビュー
        assert 350 <= n_fields <= 450, (
            f"FillTestConfig のフィールド数が {n_fields} です。"
            f" 350-450 の範囲外です — God Object 化の兆候かもしれません。"
        )


_SENTINEL = object()
