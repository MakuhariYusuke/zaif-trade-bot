"""169# Config Hot-Reload — YAML 変更をプロセス再起動なしで反映.

SkipGate モデル hot-reload (126#) と同パターン:
  - mtime ベースのファイル変更検知 (polling, configurable interval)
  - 安全なフィールドのみ差分更新 (構造体再構築が必要なものは対象外 or 明示的再構築)
  - 失敗時は旧設定を維持 (防御的設計)

Usage (FillLoopOrchestratorMixin 内)::

    self._config_reloader = ConfigHotReloader(config, yaml_path, yaml_cfg)
    # 各サイクル末尾:
    self._config_reloader.maybe_reload(self)
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.maker_price import MakerPriceCalculator
    from scripts.v460.lib.time_filter import TimeFilter

logger = logging.getLogger(__name__)


class _HotReloadableRunner(Protocol):
    """ConfigHotReloader が runner に要求する最小インタフェース.

    FillTestRunner の circular import を避けつつ型安全を確保する
    構造的サブタイピング (PEP 544).
    """

    _time_filter: TimeFilter
    _maker_price: MakerPriceCalculator
    _fast_fill_defense: object  # 210# D: FFD sync に必要
    _git_sha: str

    def _rebuild_sell_kill_mgr(self) -> None: ...
    def _rebuild_buy_kill_mgr(self) -> None: ...
    def _rebuild_daily_drawdown_guard(self) -> None: ...
    def _rebuild_fast_fill_defense(self) -> None: ...
    def _rebuild_cycle_strategy(self) -> None: ...  # 179#


# ======================================================================
# 安全にホットリロード可能なフィールドの定義
# ======================================================================

# ランタイム中に直接代入で反映できるフィールド
# (構造体の再構築が不要、または再構築を明示的に行うもの)
_HOT_RELOADABLE_FIELDS: frozenset[str] = frozenset({
    # --- offset / price 関連 ---
    "spread_offset_ratio",
    "spread_offset_ratio_buy",
    "spread_offset_ratio_sell",
    "min_offset_jpy",
    "max_offset_ratio",
    "min_offset_ratio",
    # --- regime offset 関連 ---
    "regime_trending_offset_boost",
    "regime_trending_offset_boost_buy",
    "regime_trending_offset_boost_sell",
    # 176# B: 方向×サイド別 offset boost
    "trending_up_buy_offset_boost",
    "trending_up_sell_offset_boost",
    "trending_down_buy_offset_boost",
    "trending_down_sell_offset_boost",
    "regime_high_vol_offset_boost",
    "regime_ranging_offset_discount",
    "low_vol_offset_boost_enabled",
    "low_vol_offset_boost",
    "low_vol_threshold",
    "skip_ranging_buy_low_vol",
    # --- time_filter ---
    "enable_time_filter",
    "skip_utc_hours",
    "skip_utc_hours_buy",
    "skip_utc_hours_sell",
    "regime_adaptive_enabled",
    "regime_adaptive_extra_buy",
    "regime_adaptive_extra_sell",
    # --- SkipGate 閾値 (モデル自体は別途 hot-reload) ---
    "skip_gate_enabled",
    "skip_gate_buy_enabled",
    "skip_gate_sell_enabled",
    "skip_gate_as_threshold",
    "skip_gate_as_threshold_buy",
    "skip_gate_as_threshold_sell",
    "skip_gate_pnl_threshold",
    "skip_gate_max_skip_rate",
    "skip_gate_adaptive_threshold",
    "skip_gate_target_skip_rate_buy",
    "skip_gate_target_skip_rate_sell",
    "skip_gate_hour_offsets",
    # 183# narrow spread adverse guard
    "skip_gate_narrow_spread_threshold_jpy",
    "skip_gate_narrow_spread_offset",
    # 187# clamp YAML外部化
    "skip_gate_offset_floor",
    "skip_gate_offset_ceil",
    # 188# C-1: ev_weighted SkipGate
    "skip_gate_ev_weighted_enabled",
    "skip_gate_ev_w30",
    "skip_gate_ev_w120",
    # 190# A/B: ev_weighted 連続 skip 安全弁 + 片側 balance threshold 緩和
    "skip_gate_ev_max_consecutive_skip",
    "skip_gate_ev_one_sided_threshold_shift",
    # 193# EV soft offset
    "skip_gate_ev_as_offset_enabled",
    "skip_gate_ev_offset_sensitivity",
    "skip_gate_ev_offset_min_mult",
    "skip_gate_ev_offset_max_mult",
    "skip_gate_ev_emergency_skip_threshold",
    # 189# D: MacroRegime
    "enable_macro_regime",
    "macro_regime_conflict_action",
    # --- dynamic kill ---
    "sell_dynamic_kill_enabled",
    "sell_dynamic_kill_window",
    "sell_dynamic_kill_threshold_bps",
    "sell_dynamic_kill_resume_window",
    "sell_dynamic_kill_regime_thresholds",
    "buy_dynamic_kill_enabled",
    "buy_dynamic_kill_window",
    "buy_dynamic_kill_threshold_bps",
    "buy_dynamic_kill_resume_window",
    "buy_dynamic_kill_regime_thresholds",
    # --- lot 関連 ---
    "order_quantity",
    "max_lot",
    "regime_lot_multipliers",
    "enable_confidence_lot",
    "confidence_lot_scale",
    "confidence_lot_floor",
    # --- stale / reprice ---
    "stale_order_enabled",
    "stale_check_after_sec",
    "stale_drift_bps",
    "stale_max_reprice",
    "stale_reprice_tighten",
    "stale_reprice_skip_gate_offset",
    # 175# stale side 別フィールド
    "stale_check_after_sec_buy",
    "stale_check_after_sec_sell",
    "stale_drift_bps_buy",
    "stale_drift_bps_sell",
    "stale_max_reprice_buy",
    "stale_max_reprice_sell",
    # --- narrow spread ---
    "narrow_spread_pause_enabled",
    "narrow_spread_pause_bps",
    "narrow_spread_pause_max_consecutive",
    # --- fast fill defense ---
    "fast_fill_defense_enabled",
    "fast_fill_threshold_sec",
    "fast_fill_threshold_sec_buy",     # 174# side 別
    "fast_fill_threshold_sec_sell",    # 174# side 別
    "fast_fill_offset_boost",
    "fast_fill_offset_boost_buy",      # 174# side 別
    "fast_fill_offset_boost_sell",     # 174# side 別
    # --- 日次ドローダウン ---
    "daily_drawdown_enabled",
    "daily_drawdown_hard_limit_bps",
    "daily_drawdown_soft_limit_bps",
    # --- safety ---
    "loss_cap_jpy",
    "loss_cap_ratio",
    "soft_loss_cap_ratio",
    # --- sell/buy guard ---
    "skip_sell_unknown_regime",
    "skip_buy_unknown_regime",
    "skip_sell_trending",
    "skip_sell_trending_up_only",
    # 252# Sell Asymmetric Mode: high_vol でも sell skip (Glosten-Milgrom)
    "sell_asymmetric_high_vol_enabled",
    # 196# trending sell soft offset
    "trending_sell_as_offset_enabled",
    "trending_sell_offset_boost_factor",
    # 253# 削除済み: balance_forced_apply_trending_offset (234# dead config)
    "max_consecutive_trending_sell_skip",
    "sell_guard_inv_bypass_threshold",  # 171# Guard Paradox 対策
    "skip_balance_forced",
    "balance_forced_deadlock_limit",
    "balance_forced_rescue_enabled",
    "balance_forced_rescue_offset_mult",
    # --- velocity skip ---
    "sell_velocity_skip_enabled",
    "sell_velocity_skip_threshold_bps",
    "buy_velocity_skip_enabled",
    "buy_velocity_skip_threshold_bps",
    # 195/196# velocity soft offset
    "velocity_skip_as_offset_enabled",
    "velocity_offset_boost_factor",
    "velocity_offset_proportional",
    "velocity_offset_max_mult",
    # --- VG ---
    "volatility_guard_enabled",
    "volatility_guard_velocity_threshold_bps",
    "volatility_guard_offset_boost_factor",
    # --- cycle timing ---
    "cycle_interval_sec",
    "order_timeout_sec",
    "order_timeout_sec_sell",
    "post_fill_wait_sec",       # 174# base (sell フォールバック先)
    "post_fill_wait_sec_sell",
    # --- misc ---
    "as_deadzone_bps",
    "min_spread_jpy",
    "e3_sampling_ratio",
    "progress_log_interval",
    # --- 173# maker_price 単純値読み取り ---
    "sell_offset_floor",
    "sell_offset_floor_inv_discount",
    "sell_max_spread_jpy",
    "unknown_buy_offset_boost",
    "fallback_stale_sec",
    # --- 215# P0-B: 防御パラメータ (202#-210# 追加分) ---
    # 202# A: 単一サイクル大損失クールダウン
    "loss_cooldown_threshold_bps",
    "loss_cooldown_interval_mult",
    # 207# §3: 大損後 offset 防御拡大
    "loss_boost_offset_mult",
    # 207# §1: Toxic Fill 同一サイド拒否
    "toxic_fill_veto_threshold_bps",
    "toxic_fill_veto_cycles",
    # 209# M-3: 片側連続実行制限
    "one_sided_consecutive_limit",
    "one_sided_consecutive_interval_mult",
    # 205# §9.5: 片側 DD ガード
    "per_side_dd_enabled",
    "per_side_dd_hard_limit_bps",
    "per_side_dd_halt_cycles",
    # 205# §9.4: 時間帯 Hard Skip
    "hard_skip_utc_hours",
    # 209# M-4: max cycle sleep cap
    "max_cycle_sleep_sec",
    # 212# §3.2: Soft DD interval multiplier
    "soft_drawdown_interval_multiplier",
})

# 構造体再構築が必要なコンポーネントのマッピング
# field_prefix -> コールバック名
_COMPONENT_REBUILD_PREFIXES: dict[str, str] = {
    "sell_dynamic_kill_": "_rebuild_sell_kill_mgr",
    "buy_dynamic_kill_": "_rebuild_buy_kill_mgr",
    "daily_drawdown_": "_rebuild_daily_drawdown_guard",
    "per_side_dd_": "_rebuild_daily_drawdown_guard",  # 215# P0-B: 片側 DD も同再構築
    "fast_fill_": "_rebuild_fast_fill_defense",
}


class ConfigHotReloader:
    """YAML config hot-reload manager.

    サイクル間の自然なリロードポイントで呼び出され、
    YAML ファイルの mtime を確認し、変更があれば安全なフィールドのみ差分更新。
    """

    def __init__(
        self,
        config: FillTestConfig,
        yaml_path: str | Path | None,
        yaml_cfg: dict[str, object],
        check_interval_sec: float = 120.0,
    ) -> None:
        self._config = config
        self._yaml_path: Path | None = (
            Path(yaml_path) if yaml_path is not None else None
        )
        self._yaml_cfg = yaml_cfg
        self._check_interval_sec = check_interval_sec
        self._last_check_time: float = time.time()
        self._last_mtime: float = self._get_mtime()
        self._reload_count: int = 0
        self._last_reload_time: float = 0.0

    @property
    def reload_count(self) -> int:
        return self._reload_count

    def _get_mtime(self) -> float:
        """YAML ファイルの最終更新時刻を取得."""
        if self._yaml_path is None:
            return 0.0
        try:
            return os.path.getmtime(self._yaml_path)
        except OSError:
            return 0.0

    def maybe_reload(self, runner: _HotReloadableRunner) -> bool:
        """mtime check → 変更検出時にリロード実行.

        Args:
            runner: FillTestRunner インスタンス (コンポーネント再構築用)

        Returns:
            True if reload was performed.
        """
        now = time.time()
        if now - self._last_check_time < self._check_interval_sec:
            return False

        self._last_check_time = now
        current_mtime = self._get_mtime()

        if current_mtime <= self._last_mtime:
            return False

        # mtime changed → reload
        logger.info(
            f"[config_hot_reload] YAML change detected "
            f"(mtime {self._last_mtime:.0f} → {current_mtime:.0f}), "
            f"reloading config..."
        )
        self._last_mtime = current_mtime

        try:
            return self._do_reload(runner)
        except Exception as e:
            logger.error(
                f"[config_hot_reload] Reload FAILED, keeping old config: {e}",
                exc_info=True,
            )
            return False

    def _do_reload(self, runner: _HotReloadableRunner) -> bool:
        """実際のリロード処理."""
        from scripts.v460.lib.config_loader import load_fill_test_config
        from scripts.v460.lib.fill_config import FillTestConfig

        if self._yaml_path is None:
            return False

        # 新 YAML 読込 + FillTestConfig 構築 (バリデーション含む)
        new_yaml_cfg = load_fill_test_config(self._yaml_path)
        new_config = FillTestConfig.from_yaml(new_yaml_cfg)

        # 差分検出 & 適用
        changed_fields: list[str] = []
        skipped_fields: list[str] = []
        rebuild_needed: set[str] = set()

        for f in dataclasses.fields(self._config):
            if f.name not in _HOT_RELOADABLE_FIELDS:
                continue

            old_val = getattr(self._config, f.name)
            new_val = getattr(new_config, f.name)

            if old_val != new_val:
                setattr(self._config, f.name, new_val)
                changed_fields.append(f.name)
                logger.info(
                    f"[config_hot_reload]   {f.name}: {old_val!r} → {new_val!r}"
                )

                # コンポーネント再構築が必要か判定
                for prefix, callback_name in _COMPONENT_REBUILD_PREFIXES.items():
                    if f.name.startswith(prefix):
                        rebuild_needed.add(callback_name)

        # ホットリロード対象外だが変更されたフィールドを通知
        for f in dataclasses.fields(self._config):
            if f.name in _HOT_RELOADABLE_FIELDS:
                continue
            old_val = getattr(self._config, f.name)
            new_val = getattr(new_config, f.name)
            if old_val != new_val:
                skipped_fields.append(f.name)

        if not changed_fields:
            logger.info("[config_hot_reload] No hot-reloadable fields changed")
            if skipped_fields:
                logger.warning(
                    f"[config_hot_reload] Non-reloadable fields changed "
                    f"(restart required): {skipped_fields}"
                )
            return False

        # コンポーネント再構築
        for callback_name in rebuild_needed:
            try:
                callback = getattr(runner, callback_name, None)
                if callback is not None:
                    callback()
                    logger.info(
                        f"[config_hot_reload]   component rebuilt: {callback_name}"
                    )
                    # 210# H2: FFD 再構築後に MakerPriceCalculator の参照を同期
                    # _rebuild_fast_fill_defense() は runner._fast_fill_defense を
                    # 新インスタンスに差し替えるが、_maker_price 側は旧参照を保持したまま
                    # になるため、明示的に同期する。
                    if callback_name == "_rebuild_fast_fill_defense":
                        _ffd = getattr(runner, "_fast_fill_defense", None)
                        if _ffd is not None:
                            runner._maker_price.update_fast_fill_defense(_ffd)
                            logger.info(
                                "[config_hot_reload]   MakerPriceCalculator._fast_fill_defense synced"
                            )
            except Exception as e:
                logger.error(
                    f"[config_hot_reload]   component rebuild FAILED: {callback_name}: {e}",
                    exc_info=True,
                )

        # TimeFilter 再構築 (config から直接読み取るため再構築が必要)
        if any(f.startswith(("enable_time_filter", "skip_utc_hours", "regime_adaptive_")) for f in changed_fields):
            try:
                from scripts.v460.lib.time_filter import TimeFilter
                runner._time_filter = TimeFilter(self._config)
                logger.info("[config_hot_reload]   TimeFilter rebuilt")
            except Exception as e:
                logger.error(f"[config_hot_reload]   TimeFilter rebuild FAILED: {e}")

        # MakerPriceCalculator の base offset 更新
        if any(f.startswith("spread_offset_ratio") for f in changed_fields):
            runner._maker_price.base_offset_ratio = self._config.spread_offset_ratio
            runner._maker_price.base_offset_ratio_buy = self._config.spread_offset_ratio_buy
            runner._maker_price.base_offset_ratio_sell = self._config.spread_offset_ratio_sell
            logger.info("[config_hot_reload]   MakerPriceCalculator offsets updated")

        # git SHA の再取得
        try:
            from ztb.utils.git_utils import get_git_sha
            new_sha = get_git_sha()
            if new_sha != runner._git_sha:
                old_sha = runner._git_sha
                runner._git_sha = new_sha
                logger.info(
                    f"[config_hot_reload]   git SHA updated: {old_sha} → {new_sha}"
                )
        except Exception as e:
            logger.warning(f"[config_hot_reload]   git SHA update failed: {e}")

        # 179# regime_policy セクション変更 → CycleStrategy 再構築
        # NOTE: _yaml_cfg.update() の前に比較する (更新後は old==new になるため)
        _old_rp = self._yaml_cfg.get("regime_policy", {})
        _new_rp = new_yaml_cfg.get("regime_policy", {})
        _rp_changed = _old_rp != _new_rp

        # YAML cfg も更新 (AdaptationEngine 等が参照)
        self._yaml_cfg.update(new_yaml_cfg)

        if _rp_changed:
            try:
                runner._rebuild_cycle_strategy()
                logger.info("[config_hot_reload]   CycleStrategy rebuilt (regime_policy changed)")
                changed_fields.append("regime_policy")
            except Exception as e:
                logger.error(
                    f"[config_hot_reload]   CycleStrategy rebuild FAILED: {e}",
                    exc_info=True,
                )

        self._reload_count += 1
        self._last_reload_time = time.time()

        logger.info(
            f"[config_hot_reload] Reload #{self._reload_count} complete: "
            f"{len(changed_fields)} fields updated"
        )
        if skipped_fields:
            logger.warning(
                f"[config_hot_reload] Non-reloadable fields changed "
                f"(restart required): {skipped_fields}"
            )

        return True
