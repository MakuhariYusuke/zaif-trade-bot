# AI Code Review Request: 230#/231# Fast Fill Defense Hardening

## Project Context

BTC/JPY market-making bot (v460) on coincheck exchange. Python 3.11, asyncio.
The bot places limit orders on both sides and profits from the spread.
"Fast fill defense" (FFD) detects adverse selection — when an informed trader picks off stale quotes — and widens the spread temporarily to protect against losses.

This review covers **two commits** (230# + 231#) that harden FFD logic based on market microstructure theory and fix defensive coding issues found in a pre-deployment audit.

---

## Changes Overview (12 files, +852 / -40 lines)

| # | Severity | Summary |
|---|----------|---------|
| H-1 | HIGH | FFD Layer 2 deadzone — normal spread cost no longer triggers L2 |
| H-2 | HIGH | FFD boost gradual release — Kyle 1985 informed trader model |
| H-3 | HIGH | MCB/SAD None guard — 4 locations in orchestrator |
| H-4 | HIGH | regime_detector hasattr→explicit init |
| M-1 | MEDIUM | fill_cycle_executor hasattr elimination (8/10) |
| R1 | HIGH (review) | TTL expiry streak reset |
| R2 | HIGH (review) | Slow fill + negative PnL streak counting |
| R3 | HIGH (review) | Adverse fill TTL refresh |
| R4 | MEDIUM (review) | import_state JSON null safety |
| R5 | MEDIUM (review) | Config validation upper bounds |
| R8 | MEDIUM (review) | L1+L2 simultaneous fire log label |

---

## File 1: `scripts/v460/lib/fast_fill_defense.py` (306 lines, main logic)

### Full source after changes:

```python
"""100# FastFillDefense — side-aware 即約定防御モジュール.

run_fill_test.py からの God Object 分割:
- per-side boost 状態管理 (sell fast-fill が次 buy に伝播する問題を解消)
- two-layer 負エッジ検出 (即時 proxy + post-fill PnL)
- side-specific boost cap (sell offset 0.12 に対し common 0.05 で計算していた問題を修正)

098# §3.1, 099# §2 で特定されたP0問題への対応。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FastFillDefenseConfig:
    """即約定防御の設定 (fill_test.yaml fast_fill_defense セクション対応)."""

    enabled: bool = False
    threshold_sec: float = 5.0
    threshold_sec_buy: Optional[float] = None
    threshold_sec_sell: Optional[float] = None
    offset_boost: float = 2.0
    offset_boost_buy: Optional[float] = None
    offset_boost_sell: Optional[float] = None
    # 102# YAML化: offset 上限・下限
    max_offset_ratio: float = 0.30
    min_offset_ratio: float = 0.01
    # 175# boost TTL (seconds): time_filter中の古いboostが残る問題を防止
    boost_ttl_sec: float = 600.0
    # 230# H-1: Layer 2 deadzone — normal spread cost で誤発火しない閾値
    # AS理論: maker のスプレッドコスト (~2-3 bps) は正常な損失。
    # pnl < -deadzone のときのみ adverseSelection と判定する。
    l2_deadzone_bps: float = 3.0
    # 230# H-2: boost 解除に必要な連続正常 fill 数 (Kyle 1985)
    # 情報トレーダーは複数 fill にわたり取引するため、1回の正常 fill で
    # 安全宣言するのは尚早。N 回連続で正常なら threat 終了と判断。
    boost_release_streak: int = 3


@dataclass
class _SideState:
    """side 別の boost 状態."""

    boost_active: bool = False
    boost_multiplier: float = 1.0
    boost_activated_at: float = 0.0  # 175# TTL decay 用 timestamp
    normal_fill_streak: int = 0  # 230# H-2: 連続正常 fill カウンタ


class FastFillDefense:
    """Side-aware 即約定防御.

    098# P0-3: has_negative_edge が fill_price vs mid_at_fill のみでは
    sell fast-fill AS の 50% を見逃す問題を two-layer で解消。

    Layer 1 (即時): fill_price vs mid_at_fill (従来の proxy)
    Layer 2 (post-fill): post_fill_30s_pnl < 0 なら次サイクルを遅延防御

    099# 追加: sell skip は保持すべき (逆選択)、buy-first 戦略。
    """

    def __init__(
        self,
        config: FastFillDefenseConfig,
        base_offset_ratio: float,
        base_offset_ratio_buy: Optional[float] = None,
        base_offset_ratio_sell: Optional[float] = None,
    ) -> None:
        self._config = config
        self._base_offset_ratio = base_offset_ratio
        self._base_offset_ratio_buy = base_offset_ratio_buy
        self._base_offset_ratio_sell = base_offset_ratio_sell
        # Per-side state (P0-5: sell boost が buy に伝播しないように分離)
        self._state_buy = _SideState()
        self._state_sell = _SideState()

    def _get_state(self, side: str) -> _SideState:
        return self._state_buy if side == "buy" else self._state_sell

    def _resolve_side_value(
        self,
        side: str,
        common_value: float,
        *,
        buy_value: Optional[float] = None,
        sell_value: Optional[float] = None,
    ) -> float:
        """side 別値を解決し、未設定時は共通値へフォールバックする."""
        if side == "buy":
            return common_value if buy_value is None else buy_value
        if side == "sell":
            return common_value if sell_value is None else sell_value
        return common_value

    def get_boost_multiplier(self, side: str) -> float:
        """現サイクルの offset に適用する boost 乗数を返す."""
        state = self._get_state(side)
        # 175# TTL decay: boost_ttl_sec を超過したら自動解除
        if state.boost_active and self._config.boost_ttl_sec > 0:
            if (time.time() - state.boost_activated_at) > self._config.boost_ttl_sec:
                old_mult = state.boost_multiplier
                state.boost_active = False
                state.boost_multiplier = 1.0
                state.normal_fill_streak = 0  # 231# R1: TTL期限切れで streak もクリア
                logger.info(
                    f"[fast_fill_defense] TTL expired ({side}): "
                    f"multiplier {old_mult:.2f}→1.00 "
                    f"(ttl={self._config.boost_ttl_sec:.0f}s)"
                )
        return state.boost_multiplier

    def is_boost_active(self, side: str) -> bool:
        """指定 side で boost が有効かどうか."""
        return self._get_state(side).boost_active

    def _resolve_threshold_sec(self, side: str) -> float:
        """side 別の fast_fill 判定閾値秒数を解決."""
        return self._resolve_side_value(
            side,
            self._config.threshold_sec,
            buy_value=self._config.threshold_sec_buy,
            sell_value=self._config.threshold_sec_sell,
        )

    def _resolve_boost(self, side: str) -> float:
        """side 別の offset boost 倍率を解決."""
        return self._resolve_side_value(
            side,
            self._config.offset_boost,
            buy_value=self._config.offset_boost_buy,
            sell_value=self._config.offset_boost_sell,
        )

    def _resolve_base_offset(self, side: str) -> float:
        """side 別の base offset ratio を解決.

        P0-5 fix: boost cap 計算に common 0.05 ではなく sell=0.12 等の
        side-specific 値を使用する。
        """
        return self._resolve_side_value(
            side,
            self._base_offset_ratio,
            buy_value=self._base_offset_ratio_buy,
            sell_value=self._base_offset_ratio_sell,
        )

    def _compute_capped_multiplier(self, side: str, raw_boost: float) -> float:
        """boost 乗数を side 別 base_offset_ratio で cap する.

        098# P1-2: cap = max_offset_ratio / base_offset を side 別に計算。
        sell (base=0.12) → cap=2.5, buy (base=0.05) → cap=6.0
        """
        if raw_boost <= 1.0:
            return 1.0

        safe_floor = max(self._config.min_offset_ratio, 1e-12)
        base = max(self._resolve_base_offset(side), safe_floor)
        cap = max(1.0, self._config.max_offset_ratio / base)
        return min(raw_boost, cap)

    def evaluate_fill(
        self,
        side: str,
        queue_wait_sec: float,
        fill_price: Optional[float],
        mid_at_fill: Optional[float],
        *,
        post_fill_pnl_bps: Optional[float] = None,
    ) -> None:
        """約定結果を評価し、boost 状態を更新する.

        Two-layer detection:
        - Layer 1: fill_price vs mid_at_fill (即時判定)
        - Layer 2: post_fill_pnl_bps < 0 (30s 後の確定 PnL)

        Args:
            side: "buy" or "sell"
            queue_wait_sec: 約定待ち時間 (秒)
            fill_price: 約定価格
            mid_at_fill: 約定時の mid price
            post_fill_pnl_bps: 30s 後の PnL (bps), 利用可能であれば
        """
        if not self._config.enabled:
            return

        state = self._get_state(side)
        ff_threshold = self._resolve_threshold_sec(side)
        is_fast = queue_wait_sec <= ff_threshold

        # Layer 1: 即時 proxy (fill_price vs mid)
        has_negative_edge_l1 = (
            fill_price is not None
            and mid_at_fill is not None
            and (
                (side == "buy" and fill_price > mid_at_fill)
                or (side == "sell" and fill_price < mid_at_fill)
            )
        )

        # Layer 2: post-fill PnL check (098# §3.1: 50%見逃し問題の対策)
        # 230# H-1: deadzone — 通常スプレッドコスト範囲の軽微な負PnLは無視
        _dz = self._config.l2_deadzone_bps
        has_negative_edge_l2 = (
            post_fill_pnl_bps is not None and post_fill_pnl_bps < -_dz
        )

        has_negative_edge = has_negative_edge_l1 or has_negative_edge_l2

        if is_fast and has_negative_edge:
            if not state.boost_active:
                state.boost_active = True
                raw_boost = self._resolve_boost(side)
                state.boost_multiplier = self._compute_capped_multiplier(side, raw_boost)
                _layer = (
                    "L1+L2" if (has_negative_edge_l1 and has_negative_edge_l2)
                    else ("L1" if has_negative_edge_l1 else "L2(pnl)")
                )
                logger.info(
                    f"[fast_fill_defense] Activated ({side}): "
                    f"wait={queue_wait_sec:.1f}s (< {ff_threshold}s), "
                    f"negative edge detected ({_layer}). "
                    f"multiplier→{state.boost_multiplier:.2f}"
                )
            # 231# R3: 新規/継続に関わらず TTL リフレッシュ + streak リセット
            state.boost_activated_at = time.time()
            state.normal_fill_streak = 0
        elif state.boost_active:
            # 231# R2: slow fill でも negative edge があれば streak リセット
            if has_negative_edge:
                state.normal_fill_streak = 0
            else:
                state.normal_fill_streak += 1
            _required = max(1, self._config.boost_release_streak)
            if state.normal_fill_streak >= _required:
                old_mult = state.boost_multiplier
                state.boost_multiplier = 1.0
                state.boost_active = False
                state.normal_fill_streak = 0
                logger.info(
                    f"[fast_fill_defense] Deactivated ({side}): "
                    f"{_required} consecutive normal fills, "
                    f"multiplier {old_mult:.2f}→1.00"
                )
            else:
                logger.debug(
                    f"[fast_fill_defense] Normal fill streak ({side}): "
                    f"{state.normal_fill_streak}/{_required}"
                )

    def reset_on_unfilled(self, side: str) -> None:
        """未約定時のブースト永続化防止."""
        if not self._config.enabled:
            return
        state = self._get_state(side)
        if state.boost_active:
            old_mult = state.boost_multiplier
            state.boost_multiplier = 1.0
            state.boost_active = False
            state.normal_fill_streak = 0  # 230# H-2
            logger.info(
                f"[fast_fill_defense] Reset on unfilled ({side}): "
                f"multiplier {old_mult:.2f}→1.00"
            )

    def update_base_offsets(
        self,
        base: float,
        buy: Optional[float] = None,
        sell: Optional[float] = None,
    ) -> None:
        """param_adapter 等による base offset 更新を反映."""
        self._base_offset_ratio = base
        self._base_offset_ratio_buy = buy
        self._base_offset_ratio_sell = sell

    def export_state(self) -> dict[str, object]:
        """226# hot-reload 時の boost 状態保存."""
        return {
            "buy_boost_active": self._state_buy.boost_active,
            "buy_boost_multiplier": self._state_buy.boost_multiplier,
            "buy_boost_activated_at": self._state_buy.boost_activated_at,
            "buy_normal_fill_streak": self._state_buy.normal_fill_streak,
            "sell_boost_active": self._state_sell.boost_active,
            "sell_boost_multiplier": self._state_sell.boost_multiplier,
            "sell_boost_activated_at": self._state_sell.boost_activated_at,
            "sell_normal_fill_streak": self._state_sell.normal_fill_streak,
        }

    def import_state(self, state: dict[str, object]) -> None:
        """226# hot-reload 後の boost 状態復元."""
        self._state_buy.boost_active = bool(state.get("buy_boost_active") or False)
        self._state_buy.boost_multiplier = float(state.get("buy_boost_multiplier") or 1.0)
        self._state_buy.boost_activated_at = float(state.get("buy_boost_activated_at") or 0.0)
        self._state_buy.normal_fill_streak = int(state.get("buy_normal_fill_streak") or 0)
        self._state_sell.boost_active = bool(state.get("sell_boost_active") or False)
        self._state_sell.boost_multiplier = float(state.get("sell_boost_multiplier") or 1.0)
        self._state_sell.boost_activated_at = float(state.get("sell_boost_activated_at") or 0.0)
        self._state_sell.normal_fill_streak = int(state.get("sell_normal_fill_streak") or 0)
```

---

## File 2: `scripts/v460/lib/fill_config.py` (diff only — relevant sections)

### New config fields (after existing FFD fields):

```python
    # 230# H-1: Layer 2 deadzone — 正常 spread cost を誤検知しない閾値 (bps)
    ffd_l2_deadzone_bps: float = 3.0
    # 230# H-2: boost 解除に必要な連続正常 fill 数 (Kyle 1985)
    ffd_boost_release_streak: int = 3
```

### New validation in `__post_init__`:

```python
        # 230# FFD 新規パラメータのバリデーション
        if not (0.0 <= self.ffd_l2_deadzone_bps <= 100.0):
            raise ValueError(
                f"ffd_l2_deadzone_bps must be in [0, 100], got {self.ffd_l2_deadzone_bps}"
            )
        if not (1 <= self.ffd_boost_release_streak <= 20):
            raise ValueError(
                f"ffd_boost_release_streak must be in [1, 20], got {self.ffd_boost_release_streak}"
            )
```

### New YAML parsing:

```python
        # 230# H-1/H-2: Layer 2 deadzone + boost release streak
        if "l2_deadzone_bps" in ffd:
            kwargs["ffd_l2_deadzone_bps"] = float(ffd["l2_deadzone_bps"])
        if "boost_release_streak" in ffd:
            kwargs["ffd_boost_release_streak"] = int(ffd["boost_release_streak"])
```

---

## File 3: `scripts/v460/lib/fill_loop_orchestrator.py` (diff only)

4 locations changed from:
```python
if self._mcb.config.enabled:
```
to:
```python
if self._mcb is not None and self._mcb.config.enabled:
```

Same pattern for `_sad` (2 locations each for MCB and SAD).

Context: `_mcb` and `_sad` are initialized as class-level `None` defaults (set in 228#) and only populated when their respective configs are enabled. The orchestrator has 4 code paths that check `.config.enabled` — during halt-mode feed-through (2 locations) and main-loop check (2 locations).

---

## File 4: `scripts/v460/lib/regime_detector.py` (diff only)

### Added to `__init__`:
```python
        # 230# H-4: 明示的初期化 (hasattr 排除)
        self._last_result: RegimeResult | None = None
        self._last_velocity_pct: float = 0.0
```

### Changed property accessors:
```python
# Before:
if hasattr(self, "_last_result") and self._last_result is not None:
# After:
if self._last_result is not None:

# Before:
_vel = getattr(self, "_last_velocity_pct", 0.0)
# After:
_vel = self._last_velocity_pct
```

---

## File 5: `scripts/v460/lib/fill_cycle_executor.py` (diff only — 8 changes)

All `hasattr(self, "_X")` patterns converted to `self._X is not None`:

| Line | Before | After |
|------|--------|-------|
| ~247 | `hasattr(self, "_cycle_strategy")` | `self._cycle_strategy is not None` |
| ~258 | `hasattr(self, "_cycle_strategy")` | `self._cycle_strategy is not None` |
| ~263 | `hasattr(self, "_cycle_strategy")` | `self._cycle_strategy is not None` |
| ~342 | `hasattr(self, "_cycle_strategy")` | `self._cycle_strategy is not None` |
| ~553 | `hasattr(self, "_cycle_strategy")` | `self._cycle_strategy is not None` |
| ~558 | `hasattr(self, "_regime_detector") and ...` | `self._regime_detector is not None` |
| ~1126 | `hasattr(self, "_macro_regime_detector") and ...` | `self._macro_regime_detector is not None` |

2 legitimate `hasattr(self, "_current_regime_value")` retained (mixin method existence check).

---

## File 6: `scripts/v460/run_fill_test.py` (diff only)

Two `FastFillDefenseConfig(...)` constructor calls (initial + hot-reload) now pass the new params:

```python
                l2_deadzone_bps=config.ffd_l2_deadzone_bps,
                boost_release_streak=config.ffd_boost_release_streak,
```

---

## File 7: `configs/v460/fill_test.yaml` (diff only)

```yaml
fast_fill_defense:
  # ... existing fields ...
  # 230# H-1: Layer 2 deadzone — 正常 spread cost を超えた損失のみ AS 認定
  l2_deadzone_bps: 3.0
  # 230# H-2: boost 解除に必要な連続正常 fill 数 (Kyle 1985)
  boost_release_streak: 3
```

---

## File 8: `tests/unit/v460/test_230_ffd_deadzone_streak_guards.py` (568 lines — full)

```python
"""230# テスト: FFD deadzone/streak + MCB/SAD None guard + hasattr 排除.

変更概要:
  H-1: FFD Layer 2 deadzone — 正常 spread cost で L2 誤発火しない
  H-2: FFD boost gradual release — Kyle 1985 連続正常 fill streak 要求
  H-3: MCB/SAD None guard — _mcb/_sad is None で AttributeError 回避
  H-4: regime_detector hasattr→init 変換
  M-1: fill_cycle_executor hasattr 排除 (8/10, 2 legitimate 残留)
  Config: ffd_l2_deadzone_bps / ffd_boost_release_streak 新規バリデーション
"""

from __future__ import annotations

import inspect
import time

import pytest

from scripts.v460.lib.fast_fill_defense import (
    FastFillDefense,
    FastFillDefenseConfig,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.regime_detector import FillTestRegimeDetector, RegimeConfig


# ======================================================================
# Helpers
# ======================================================================


def _make_ffd(
    *,
    deadzone_bps: float = 3.0,
    streak: int = 3,
    threshold_sec: float = 5.0,
    offset_boost: float = 2.0,
) -> FastFillDefense:
    """テスト用 FFD を生成."""
    cfg = FastFillDefenseConfig(
        enabled=True,
        threshold_sec=threshold_sec,
        offset_boost=offset_boost,
        l2_deadzone_bps=deadzone_bps,
        boost_release_streak=streak,
    )
    return FastFillDefense(cfg, base_offset_ratio=0.005)


def _activate_boost(ffd: FastFillDefense, side: str = "buy") -> None:
    """fast fill + L1 negative edge で boost を確実に有効化."""
    fp = 101_000 if side == "buy" else 99_000
    ffd.evaluate_fill(
        side,
        queue_wait_sec=1.0,
        fill_price=fp,
        mid_at_fill=100_000,
    )
    assert ffd.is_boost_active(side)


def _normal_fill(
    ffd: FastFillDefense,
    side: str = "buy",
    *,
    post_fill_pnl_bps: float | None = None,
) -> None:
    """正常 fill (not fast, no negative edge)."""
    ffd.evaluate_fill(
        side,
        queue_wait_sec=60.0,
        fill_price=99_500 if side == "buy" else 100_500,
        mid_at_fill=100_000,
        post_fill_pnl_bps=post_fill_pnl_bps,
    )


# ======================================================================
# H-1: FFD Layer 2 deadzone (AS theory)
# ======================================================================


class TestFFDL2Deadzone:
    """H-1: 正常スプレッドコスト (~2-3bps) による L2 誤発火を防止."""

    def test_within_deadzone_no_trigger(self) -> None:
        """pnl = -2.5bps, deadzone = 3.0bps → |pnl| < deadzone → L2 不発動."""
        ffd = _make_ffd(deadzone_bps=3.0)
        ffd.evaluate_fill(
            "sell",
            queue_wait_sec=3.0,
            fill_price=100_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-2.5,
        )
        assert not ffd.is_boost_active("sell")

    def test_beyond_deadzone_triggers(self) -> None:
        """pnl = -5.0bps, deadzone = 3.0bps → |pnl| > deadzone → L2 発動."""
        ffd = _make_ffd(deadzone_bps=3.0)
        ffd.evaluate_fill(
            "sell",
            queue_wait_sec=3.0,
            fill_price=100_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-5.0,
        )
        assert ffd.is_boost_active("sell")

    def test_exactly_at_deadzone_boundary_no_trigger(self) -> None:
        """pnl = -3.0bps, deadzone = 3.0bps → not strictly less → no trigger."""
        ffd = _make_ffd(deadzone_bps=3.0)
        ffd.evaluate_fill(
            "sell",
            queue_wait_sec=3.0,
            fill_price=100_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-3.0,
        )
        assert not ffd.is_boost_active("sell")

    def test_deadzone_zero_behaves_like_old(self) -> None:
        """deadzone=0 → 旧挙動 (pnl<0 で即発動)."""
        ffd = _make_ffd(deadzone_bps=0.0)
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=2.0,
            fill_price=99_900,
            mid_at_fill=100_000,
            post_fill_pnl_bps=-0.5,
        )
        assert ffd.is_boost_active("buy")

    def test_positive_pnl_never_triggers_l2(self) -> None:
        """pnl>0 → deadzone 関係なく L2 は不発動."""
        ffd = _make_ffd(deadzone_bps=0.0)
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=2.0,
            fill_price=99_500,
            mid_at_fill=100_000,
            post_fill_pnl_bps=1.0,
        )
        assert not ffd.is_boost_active("buy")

    def test_l1_still_works_regardless_of_deadzone(self) -> None:
        """L1 (fill_price vs mid) は deadzone に影響されない."""
        ffd = _make_ffd(deadzone_bps=100.0)
        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=2.0,
            fill_price=101_000,
            mid_at_fill=100_000,
        )
        assert ffd.is_boost_active("buy")


# ======================================================================
# H-2: FFD boost gradual release (Kyle 1985)
# ======================================================================


class TestFFDBoostGradualRelease:
    """H-2: 情報漸次伝播 — N回連続正常fillで解除."""

    def test_single_normal_fill_not_enough(self) -> None:
        """streak=3 + 1 normal fill → boost 維持."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")

    def test_two_normal_fills_not_enough(self) -> None:
        """streak=3 + 2 normal fills → boost 維持."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")

    def test_three_normal_fills_deactivates(self) -> None:
        """streak=3 + 3 normal fills → boost 解除."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        for _ in range(3):
            _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")
        assert ffd.get_boost_multiplier("buy") == 1.0

    def test_streak_resets_on_new_adverse(self) -> None:
        """途中で再度 adverse fill → streak リセット."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")

        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")

        ffd.evaluate_fill(
            "buy",
            queue_wait_sec=1.0,
            fill_price=101_000,
            mid_at_fill=100_000,
        )
        assert ffd.is_boost_active("buy")
        assert ffd._get_state("buy").normal_fill_streak == 0

        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd.is_boost_active("buy")
        _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")

    def test_streak_one_behaves_like_old(self) -> None:
        """streak=1 → 旧挙動 (1 normal fill で即解除)."""
        ffd = _make_ffd(streak=1)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")

    def test_sell_side_streak(self) -> None:
        """sell 側でも streak logic が動作."""
        ffd = _make_ffd(streak=2)
        _activate_boost(ffd, "sell")
        _normal_fill(ffd, "sell")
        assert ffd.is_boost_active("sell")
        _normal_fill(ffd, "sell")
        assert not ffd.is_boost_active("sell")

    def test_side_isolation_with_streak(self) -> None:
        """buy streak は sell に影響しない."""
        ffd = _make_ffd(streak=2)
        _activate_boost(ffd, "buy")
        _activate_boost(ffd, "sell")

        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert not ffd.is_boost_active("buy")
        assert ffd.is_boost_active("sell")

    def test_reset_on_unfilled_clears_streak(self) -> None:
        """未約定リセットで streak もクリア."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd._get_state("buy").normal_fill_streak == 2

        ffd.reset_on_unfilled("buy")
        assert not ffd.is_boost_active("buy")
        assert ffd._get_state("buy").normal_fill_streak == 0

    def test_slow_adverse_pnl_resets_streak(self) -> None:
        """231# R2: slow fill でも negative PnL があれば streak リセット."""
        ffd = _make_ffd(streak=3, deadzone_bps=2.0)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")
        _normal_fill(ffd, "buy")
        assert ffd._get_state("buy").normal_fill_streak == 2

        _normal_fill(ffd, "buy", post_fill_pnl_bps=-5.0)
        assert ffd._get_state("buy").normal_fill_streak == 0
        assert ffd.is_boost_active("buy")

    def test_ttl_expiry_resets_streak(self) -> None:
        """231# R1: TTL 期限切れで streak もクリア."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")

        state = ffd._get_state("buy")
        state.boost_activated_at = time.time() - 700.0
        ffd.get_boost_multiplier("buy")

        assert not ffd.is_boost_active("buy")
        assert state.normal_fill_streak == 0

    def test_adverse_refreshes_ttl(self) -> None:
        """231# R3: 継続 adverse fill で TTL がリフレッシュ."""
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")

        t_before = ffd._get_state("buy").boost_activated_at
        time.sleep(0.01)
        ffd.evaluate_fill(
            "buy", queue_wait_sec=1.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        t_after = ffd._get_state("buy").boost_activated_at
        assert t_after > t_before


# ======================================================================
# H-2 state persistence
# ======================================================================


class TestFFDStreakStatePersistence:
    """export/import で normal_fill_streak が保存・復元される."""

    def test_export_includes_streak(self) -> None:
        ffd = _make_ffd(streak=3)
        _activate_boost(ffd, "buy")
        _normal_fill(ffd, "buy")

        state = ffd.export_state()
        assert state["buy_normal_fill_streak"] == 1
        assert state["sell_normal_fill_streak"] == 0

    def test_import_restores_streak(self) -> None:
        ffd = _make_ffd(streak=3)
        state = {
            "buy_boost_active": True,
            "buy_boost_multiplier": 2.0,
            "buy_boost_activated_at": 12345.0,
            "buy_normal_fill_streak": 2,
            "sell_boost_active": False,
            "sell_boost_multiplier": 1.0,
            "sell_boost_activated_at": 0.0,
            "sell_normal_fill_streak": 0,
        }
        ffd.import_state(state)
        assert ffd._state_buy.normal_fill_streak == 2
        assert ffd._state_sell.normal_fill_streak == 0

    def test_import_null_streak_defaults_zero(self) -> None:
        """旧バージョン state (streak なし) → 0 にフォールバック."""
        ffd = _make_ffd(streak=3)
        state = {
            "buy_boost_active": True,
            "buy_boost_multiplier": 2.0,
            "buy_boost_activated_at": 12345.0,
            "sell_boost_active": False,
            "sell_boost_multiplier": 1.0,
            "sell_boost_activated_at": 0.0,
        }
        ffd.import_state(state)
        assert ffd._state_buy.normal_fill_streak == 0
        assert ffd._state_sell.normal_fill_streak == 0

    def test_import_none_value_streak(self) -> None:
        """231# R4: JSON の null 値で import がクラッシュしない."""
        ffd = _make_ffd(streak=3)
        state = {
            "buy_boost_active": True,
            "buy_boost_multiplier": None,
            "buy_boost_activated_at": None,
            "buy_normal_fill_streak": None,
            "sell_boost_active": None,
            "sell_boost_multiplier": None,
            "sell_boost_activated_at": None,
            "sell_normal_fill_streak": None,
        }
        ffd.import_state(state)
        assert ffd._state_buy.boost_multiplier == 1.0
        assert ffd._state_buy.normal_fill_streak == 0
        assert ffd._state_sell.boost_active is False


# ======================================================================
# H-3: MCB/SAD None guard
# ======================================================================


class TestMCBSADNoneGuard:
    """H-3: _mcb/_sad is None 時に AttributeError しない."""

    def test_orchestrator_mcb_none_attribute(self) -> None:
        from scripts.v460.lib import fill_loop_orchestrator as mod
        src = inspect.getsource(mod)
        assert "self._mcb is not None and self._mcb.config.enabled" in src

    def test_orchestrator_sad_none_attribute(self) -> None:
        from scripts.v460.lib import fill_loop_orchestrator as mod
        src = inspect.getsource(mod)
        assert "self._sad is not None and self._sad.config.enabled" in src

    def test_no_bare_mcb_config_access(self) -> None:
        from scripts.v460.lib import fill_loop_orchestrator as mod
        src = inspect.getsource(mod)
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if "self._mcb.config.enabled" in stripped:
                assert "self._mcb is not None" in stripped

    def test_no_bare_sad_config_access(self) -> None:
        from scripts.v460.lib import fill_loop_orchestrator as mod
        src = inspect.getsource(mod)
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if "self._sad.config.enabled" in stripped:
                assert "self._sad is not None" in stripped


# ======================================================================
# H-4: regime_detector hasattr→init
# ======================================================================


class TestRegimeDetectorInit:
    """H-4: _last_result / _last_velocity_pct が __init__ で初期化."""

    def test_no_hasattr_in_source(self) -> None:
        from scripts.v460.lib import regime_detector as mod
        src = inspect.getsource(mod)
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "hasattr(" not in stripped

    def test_no_getattr_fallback_in_source(self) -> None:
        from scripts.v460.lib import regime_detector as mod
        src = inspect.getsource(mod)
        assert 'getattr(self, "_last_velocity_pct"' not in src

    def test_last_volatility_ratio_before_update(self) -> None:
        rd = FillTestRegimeDetector(RegimeConfig())
        assert rd.last_volatility_ratio == 1.0

    def test_current_confidence_before_update(self) -> None:
        rd = FillTestRegimeDetector(RegimeConfig())
        assert rd.current_confidence == 0.0

    def test_init_attributes_exist(self) -> None:
        rd = FillTestRegimeDetector(RegimeConfig())
        assert hasattr(rd, "_last_result")
        assert hasattr(rd, "_last_velocity_pct")
        assert rd._last_result is None
        assert rd._last_velocity_pct == 0.0


# ======================================================================
# M-1: fill_cycle_executor hasattr 排除
# ======================================================================


class TestFillCycleExecutorHasattr:
    """M-1: fill_cycle_executor の hasattr を is not None に変換."""

    def test_no_hasattr_cycle_strategy(self) -> None:
        from scripts.v460.lib import fill_cycle_executor as mod
        src = inspect.getsource(mod)
        assert 'hasattr(self, "_cycle_strategy")' not in src

    def test_no_hasattr_regime_detector(self) -> None:
        from scripts.v460.lib import fill_cycle_executor as mod
        src = inspect.getsource(mod)
        assert 'hasattr(self, "_regime_detector")' not in src

    def test_no_hasattr_macro_regime_detector(self) -> None:
        from scripts.v460.lib import fill_cycle_executor as mod
        src = inspect.getsource(mod)
        assert 'hasattr(self, "_macro_regime_detector")' not in src

    def test_legitimate_hasattr_current_regime_value_remains(self) -> None:
        from scripts.v460.lib import fill_cycle_executor as mod
        src = inspect.getsource(mod)
        assert 'hasattr(self, "_current_regime_value")' in src


# ======================================================================
# Config validation
# ======================================================================


class TestConfigValidation230:
    """230# 新規フィールドのバリデーション."""

    def test_ffd_l2_deadzone_bps_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.ffd_l2_deadzone_bps == 3.0

    def test_ffd_boost_release_streak_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.ffd_boost_release_streak == 3

    def test_ffd_l2_deadzone_bps_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="ffd_l2_deadzone_bps"):
            FillTestConfig(ffd_l2_deadzone_bps=-1.0)

    def test_ffd_l2_deadzone_bps_over_100_raises(self) -> None:
        with pytest.raises(ValueError, match="ffd_l2_deadzone_bps"):
            FillTestConfig(ffd_l2_deadzone_bps=101.0)

    def test_ffd_l2_deadzone_bps_zero_ok(self) -> None:
        cfg = FillTestConfig(ffd_l2_deadzone_bps=0.0)
        assert cfg.ffd_l2_deadzone_bps == 0.0

    def test_ffd_boost_release_streak_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="ffd_boost_release_streak"):
            FillTestConfig(ffd_boost_release_streak=0)

    def test_ffd_boost_release_streak_over_20_raises(self) -> None:
        with pytest.raises(ValueError, match="ffd_boost_release_streak"):
            FillTestConfig(ffd_boost_release_streak=21)

    def test_ffd_boost_release_streak_one_ok(self) -> None:
        cfg = FillTestConfig(ffd_boost_release_streak=1)
        assert cfg.ffd_boost_release_streak == 1


# ======================================================================
# FastFillDefenseConfig defaults
# ======================================================================


class TestFFDConfigDefaults:
    """230# FFDConfig の新規デフォルト値."""

    def test_l2_deadzone_default(self) -> None:
        cfg = FastFillDefenseConfig()
        assert cfg.l2_deadzone_bps == 3.0

    def test_boost_release_streak_default(self) -> None:
        cfg = FastFillDefenseConfig()
        assert cfg.boost_release_streak == 3

    def test_side_state_normal_fill_streak_default(self) -> None:
        from scripts.v460.lib.fast_fill_defense import _SideState
        s = _SideState()
        assert s.normal_fill_streak == 0
```

---

## File 9: `tests/unit/v460/test_100_fast_fill_defense.py` (diff only)

Two existing tests updated for backward compatibility with new defaults:

1. `test_layer2_post_fill_pnl_negative`: Now passes `l2_deadzone_bps=2.0` so the test PnL value `-5bps` still exceeds the deadzone.
2. `test_normal_fill_deactivates`: Now passes `boost_release_streak=1` to preserve old "1 normal fill = deactivate" behavior.

---

## Review Questions

Please examine the above changes critically and answer:

### Correctness & State Machine

1. **State machine completeness**: The `evaluate_fill()` method implements a 3-branch state machine (`if is_fast and has_negative_edge` / `elif state.boost_active` / implicit else). Here is the full decision table:

   | is_fast | has_negative_edge | boost_active | Action |
   |---------|-------------------|--------------|--------|
   | T | T | F | **activate**: mult=boost, streak=0, ttl=now |
   | T | T | T | **refresh**: ttl=now, streak=0 |
   | T | F | F | (no-op) |
   | T | F | T | streak++, check deactivate |
   | F | T | F | (no-op) — **is this correct?** |
   | F | T | T | streak=0 (adverse but slow) |
   | F | F | F | (no-op) |
   | F | F | T | streak++, check deactivate |

   Row 5 is interesting: a slow fill with adverse PnL, but boost is not yet active. Should this be a no-op? The current logic only activates boost on fast fills, meaning a slow fill with -30bps PnL is silently ignored. Is this by design (only fast fills indicate adverse selection) or a gap?

2. **State consistency**: Is `normal_fill_streak` correctly reset in ALL code paths? Please verify:
   - Activation (fast+adverse): ✓ streak=0
   - Deactivation (streak≥N): ✓ streak=0
   - TTL expiry: ✓ streak=0 (R1 fix)
   - Unfilled reset: ✓ streak=0
   - import_state: ✓ defaults to 0
   - Adverse refresh (fast+adverse while active): ✓ streak=0
   - Slow adverse (while active): ✓ streak=0 (R2 fix)
   - **Any path we missed?**

3. **TTL refresh (R3) — infinite defense**: Moving `boost_activated_at = time.time()` outside `if not state.boost_active:` means every adverse fast fill refreshes the TTL. Could an attacker keep the bot's spread permanently widened by sending 1 adverse fill every 9 minutes (within 600s TTL)? Is this desirable? In Kyle (1985), the informed trader wants to trade at favorable prices — wouldn't a permanently wide spread be the bot's intended defense against a persistent attacker?

4. **Slow fill + negative edge (R2) — TTL not refreshed**: When `is_fast=False` and `has_negative_edge=True`, we enter the `elif state.boost_active:` branch and reset the streak to 0, but we do NOT refresh `boost_activated_at`. This means if an attacker switches from fast fills to slow fills, the TTL continues counting down from the last fast fill. After 600s, the defense expires even if slow adverse fills are ongoing. Is this the intended behavior?

### Defensive Coding

5. **import_state null safety (R4)**: The `state.get("key") or default` pattern treats ALL falsy values as "missing". Specific concern:
   - `boost_activated_at=0.0`: `float(0.0 or 0.0)` → `float(0.0)` → `0.0` ✓ (works because default is also 0.0)
   - `boost_multiplier=0.0`: `float(0.0 or 1.0)` → `float(1.0)` → `1.0` — **Is it ever valid to import `boost_multiplier=0.0`?** If so, this silently changes it to 1.0.
   - `boost_active=False`: `bool(False or False)` → `bool(False)` → `False` ✓
   - **Alternative**: Would `x if x is not None else default` be more precise?

6. **Side parameter validation**: `side: str` is passed throughout without input validation. `_get_state()` returns `_state_sell` for any non-"buy" string (including typos, empty string, etc.). Should there be an assertion or `Literal["buy", "sell"]` type annotation?

### Config & Bounds

7. **Config bounds appropriateness**: `l2_deadzone_bps ∈ [0, 100]` and `boost_release_streak ∈ [1, 20]`. For a BTC/JPY market maker:
   - BTC/JPY typical spread: ~5-15 bps. Is deadzone=100 bps too generous as an upper bound?
   - At 30-120s cycle intervals, streak=20 means 10-40 minutes of sustained normal fills before deactivation. Is 20 too aggressive for `boost_release_streak` max? Or too conservative?

### Architecture

8. **`get_boost_multiplier` side effect**: This "getter" checks TTL and mutates state (deactivates boost, resets streak, logs). Callers expect a pure getter. Should this be separated into `check_ttl_expiry()` + `get_boost_multiplier()`, or is the encapsulation acceptable given single-threaded asyncio?

9. **Thread safety**: `_SideState` is mutable and unsynchronized. In asyncio, coroutines can yield between any two `await` statements. Since `evaluate_fill()` is synchronous (no `await`), is there any risk of interleaving between `evaluate_fill()` and `get_boost_multiplier()` in the event loop?

10. **Missing coverage**: The tests verify source code patterns for MCB/SAD None guards (via `inspect.getsource`). These are fragile — they break if the source is refactored. Should there be runtime integration tests that instantiate the orchestrator with `_mcb=None` and `_sad=None` and verify no `AttributeError`?

### State Transition Diagram

For reference, here is the state machine:

```
                          ┌─────────────────────────────┐
                          │                             │
                          ▼                             │
    ┌──────────┐   fast+adverse   ┌──────────────┐      │
    │ INACTIVE │ ────────────────→│   ACTIVE     │──────┘
    │ mult=1.0 │                  │ mult=N.N     │  fast+adverse
    │ streak=0 │                  │ streak=0     │  (TTL refresh)
    └──────────┘                  │ ttl=time()   │
         ▲                        └──────┬───────┘
         │                               │
         │                    ┌──────────┼──────────┐
         │                    │          │          │
         │              slow+adverse  normal    TTL expire
         │              streak=0    streak++   streak=0
         │                    │          │     mult=1.0
         │                    │          │          │
         │                    ▼          ▼          │
         │              ┌────────┐  streak≥N?      │
         │              │ ACTIVE │  ───Yes──→───────┤
         │              │ (stay) │                  │
         │              └────────┘                  │
         │                                          │
         │◄─────────── deactivate ◄─────────────────┘
         │              (streak≥N or TTL or unfilled)
         │
    ┌────┴─────┐
    │ unfilled │ → reset (mult=1.0, streak=0)
    └──────────┘
```

---

## Architectural Context

### FFD in the broader system

```
FillLoopOrchestratorMixin (fill_loop_orchestrator.py, ~2020 lines)
 │
 ├─ FillCycleExecutorMixin (fill_cycle_executor.py, ~1207 lines)
 │   └─ Records fill results, evaluates post-fill PnL, calls FFD.evaluate_fill()
 │
 ├─ FastFillDefense (fast_fill_defense.py, ~306 lines)  ← THIS REVIEW
 │   └─ Per-side boost state, two-layer detection, streak-based release
 │
 ├─ MicroCircuitBreaker (MCB) — short-term price shock detector
 │   └─ Pauses trading for N seconds on σ-deviation
 │
 ├─ SpreadAnomalyDetector (SAD) — liquidity drought detector
 │   └─ Halts on abnormal spread widening
 │
 ├─ FillTestRegimeDetector (regime_detector.py, ~418 lines)
 │   └─ Trending/Ranging/Unknown classification
 │
 └─ CycleStrategy — regime-dependent parameter selection
```

### Data flow for FFD

```
1. Orchestrator places limit order
2. Order fills (or times out → reset_on_unfilled)
3. FillCycleExecutor measures queue_wait_sec, fill_price, mid_at_fill
4. FillCycleExecutor waits 30s, measures post_fill_pnl_bps
5. FFD.evaluate_fill(side, queue_wait_sec, fill_price, mid_at_fill, post_fill_pnl_bps)
6. Next cycle: FFD.get_boost_multiplier(side) → offset *= multiplier
```

### Hot-reload flow

```
1. YAML change detected
2. Old FFD: export_state() → dict (JSON-serializable)
3. New FFD instance created with new config
4. New FFD: import_state(dict)
5. Trading continues with new config, preserved boost state
```

## Known Design Trade-offs

| Decision | Rationale | Risk |
|----------|-----------|------|
| `side: str` not `Literal["buy", "sell"]` | Historical API; changing would require orchestrator-wide refactor | Typo → defaults to sell state |
| `get_boost_multiplier` has TTL side effect | Keeps TTL check encapsulated; alternative requires caller discipline | Violates getter purity |
| `or default` instead of `if x is not None else default` | Simpler, and all `0`/`False` defaults are intentionally falsy | `boost_multiplier=0.0` silently becomes `1.0` |
| L2 only activates on fast fills | Slow fill = market order at bad price, but not "picked off" | Persistent slow adverse ignored for activation |
| TTL not refreshed on slow adverse | Slow adverse doesn't prove attacker is still active (could be noise) | Defense expires during slow attack |
| MCB/SAD tests are source-string-based | Runtime test requires full orchestrator mock (~30 dependencies) | Tests break on refactoring |

---

## Test Results

- **60 tests** in `test_230_ffd_deadzone_streak_guards.py`: all passed
- **Full v460 regression**: 3154 passed, 0 failed
