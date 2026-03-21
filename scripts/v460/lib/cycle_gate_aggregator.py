"""193# CycleGateAggregator — per-cycle skip 判定の一元化.

192# §3.1 の指摘「同じ判断が 4 箇所に分散」を解消する。

設計原則:
  - 全 per-cycle skip/block 判定を 1 箇所に集約
  - Hard blocker と Soft modifier の明示的分離
  - 全判定の audit trail を CycleGateResult に一元記録
  - 既存テストの後方互換性を最大限維持

MAX LINES: 800

旧アーキテクチャ:
  orchestrator → A10-A14 (scattered if/continue)
  executor     → B3 (narrow_spread_pause)
  skip_gate    → C2, C4-C5 (unknown_sell, velocity_skip)
  maker_price  → D1-D3 (ValueError raise)

新アーキテクチャ:
  orchestrator → cycle_gate.evaluate(context) → CycleGateResult
  executor     → (B3 はここで判定済み)
  skip_gate    → (C2, C4-C5 はここで判定済み、ML のみ残留)
  maker_price  → (D1-D3 はここで事前チェック済み)

Gate 分類の市場理論的根拠 (273# I6 / 274#)
──────────────────────────────────────────────
**Hard Gates (Gate 4, 5, 8, 9)** — 市場構造的安全:
  - Kill Gate (4, 5): Glosten-Milgrom (1985) の逆選択コスト。rolling PnL が
    閾値以下 = informed flow に搾取されており、取引継続は期待負 PnL。
    halt recovery 中でも免除不可 (逆選択は halt 前後で解消しない)。
  - Spread/Price Gate (8, 9): 執行品質のハード制約。狭小スプレッドでは
    maker 優位性 (bid-ask capture) が消滅し、取引コストが利益を上回る。
    Roll (1984) の有効スプレッド理論による。

**Soft Gates (Gate 1, 2, 3, 6, 7)** — 政策的判断:
  - Regime Gate (1, 2, 3, 7): 市場レジームに基づく「条件付き期待値」判断。
    unknown/trending/ranging は Oracle の不確実性を反映し、期待 PnL の
    条件付き推定が負になる状況を回避する政策。しかし halt 解除直後は
    ポジション再構築が最優先であり、短期的な政策緩和が合理的 (273# I6)。
  - Velocity Gate (6): 価格速度が急変する場合の逆選択リスク警告。
    Kyle (1985) の λ (price impact) に対応するが、推定精度が低く
    hard gate とするには不十分。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.sidecar_types import SidecarSignal
    from ztb.risk.sell_dynamic_kill import ToxicityAssessment

# 241# S-3: ToxicityLevel 値をモジュールレベルでキャッシュ (hot path import 回避)
from ztb.risk.sell_dynamic_kill import ToxicityLevel

_ToxicityLevel_GREEN = ToxicityLevel.GREEN
_ToxicityLevel_KILL = ToxicityLevel.KILL

logger = logging.getLogger(__name__)

# ゲート blocking_reason → cancel_reason 定数のマッピング
_GATE_TO_CANCEL_REASON: dict[str, str] = {
    "unknown_regime_buy_skip": "unknown_regime_buy_skip",
    "ranging_low_vol_skip": "ranging_low_vol_skip",
    "trending_sell_skip": "trending_sell_skip",
    "buy_dynamic_kill": "buy_dynamic_kill",
    "sell_dynamic_kill": "sell_dynamic_kill",
    "rule_velocity_sell_skip": "skip_gate_rule_velocity_sell",
    "rule_velocity_buy_skip": "skip_gate_rule_velocity_buy",
    "rule_skip_unknown_sell": "unknown_regime_sell_skip",
    "narrow_spread_pause": "narrow_spread_pause",        # 197# Gate 8
    "spread_too_narrow": "spread_too_narrow",            # 197# Gate 9
    "sell_guard_reject": "sell_guard_reject",             # 197# Gate 9
    "toxicity_participation_skip": "toxicity_participation_skip",  # 240# Toxicity Budget
    "composite_risk_exceeded": "composite_risk_exceeded",  # 491# Composite Risk Score
}


@dataclass
class GateCheckResult:
    """個別ゲートの判定結果."""

    gate_name: str
    blocked: bool
    reason: str = ""
    detail: str = ""
    offset_mult: float | None = None  # 196# soft mode offset 乗数


@dataclass
class CycleGateResult:
    """全ゲート評価の統合結果.

    - blocked=True: このサイクルは skip すべき
    - blocked=False: 取引続行可
    - blocking_reason: skip の最初の理由
    - checks: 全ゲートの判定結果チェーン (audit trail)
    - trending_offset_mult: 196# trending sell soft mode offset 乗数
    - degraded_liquidation: 234# Kill Gate 阻止時の縮退清算モード
      (348# balance_forced 撤廃 / 522# inventory_escape 完全撤廃)
    """

    blocked: bool = False
    blocking_reason: str = ""
    checks: list[GateCheckResult] = field(default_factory=list)
    trending_offset_mult: float | None = None  # 196# trending sell → offset boost
    dual_kill_bypassed: bool = False  # 223# DUAL KILL bypass 発動フラグ
    degraded_liquidation: bool = False  # 234# 縮退清算モード
    degraded_reason: str = ""  # 234# 縮退理由 (どの gate が triggered か)
    # 240# Toxicity Budget (232# §2.2)
    toxicity_offset_mult: float = 1.0  # 逆選択リスクに応じた offset 乗数
    participation_rate: float = 1.0    # 確率的参加率 (1.0=全参加, 0.0=全停止)
    # 365# P5: SAC Sidecar — Asymmetric Maker offset
    sidecar_offset_bps: float = 0.0    # 正=攻撃的(midに近い), 負=保守的
    sidecar_direction: str = "neutral"  # buy_bias / sell_bias / neutral
    sidecar_bias: float = 0.0          # raw bias 値 [-1,+1] (audit用)
    # 451# P1-2: compound suppression 診断用投機的ゲート評価
    speculative_checks: list[GateCheckResult] = field(default_factory=list)
    # 487# P0: sidecar attribution 可観測性 (orchestrator で設定)
    sidecar_confidence: float | None = None
    sidecar_model_version: str | None = None
    sidecar_signal_status: str = "missing"
    # 491# Composite Risk Score (490# architectural pivot)
    composite_risk_score: float = 0.0   # soft gate risk weight の加算値
    composite_risk_details: list[str] = field(default_factory=list)  # gate→weight 内訳

    @property
    def audit_summary(self) -> str:
        """全ゲート判定のワンライン要約."""
        parts: list[str] = []
        for c in self.checks:
            mark = "✗" if c.blocked else "✓"
            parts.append(f"{mark}{c.gate_name}")
        # 451# P1-2: compound suppression speculative info
        for c in self.speculative_checks:
            mark = "✗" if c.blocked else "✓"
            parts.append(f"({mark}{c.gate_name})")
        return " → ".join(parts)

    @property
    def cancel_reason(self) -> str:
        """blocking_reason → cancel_reasons モジュールの定数文字列."""
        return _GATE_TO_CANCEL_REASON.get(self.blocking_reason, self.blocking_reason)

    @property
    def blocking_category(self) -> str:
        """244# blocking_reason の GuardCategory 分類 (market/system/recovery)."""
        if not self.blocking_reason:
            return ""
        from scripts.v460.lib.guard_reason_classifier import classify_guard
        return classify_guard(f"gate_{self.blocking_reason}").value


class CycleGateAggregator:
    """193# per-cycle skip 判定の一元ゲート.

    orchestrator の各 skip チェックをメソッド化し、
    evaluate() で全てを順次実行。
    最初の Hard Blocker で即 blocked=True を返す。

    ────────────────────────────────────────────────────
    責務: per-cycle の「この side で取引すべきか」判定を一元管理
    NOT 責務: ループ制御, balance check, system-level halt
    ────────────────────────────────────────────────────
    """

    #: 219# unknown regime 連続ブロックのバイパス閾値
    #: 277# config 化 → config.unknown_regime_max_consecutive で外部設定可能
    #: 後方互換: クラス属性として残す (テストが参照)
    #: 336# drift fix: fill_config デフォルト 5 に合わせる
    UNKNOWN_REGIME_MAX_CONSECUTIVE: int = 5

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        # 277#: config から実際の閾値を取得 (YAML 上書き対応)
        self.UNKNOWN_REGIME_MAX_CONSECUTIVE = config.unknown_regime_max_consecutive
        # 324# M-2: per-side 分離 (321# §5 M-2/319# M-6)
        # buy/sell 交互呼出しで混合カウントされる問題を解消
        self._consecutive_unknown_blocks: dict[str, int] = {"buy": 0, "sell": 0}

    def evaluate(
        self,
        *,
        side: str,
        regime: str | None,
        vol_ratio: float | None,
        inv_net_imbalance: float,
        is_buy_killed: bool,
        is_sell_killed: bool,
        spread_jpy: float | None = None,
        mid_price: float | None = None,
        price_velocity_bps: float | None = None,
        # 安全弁の状態 (orchestrator が管理するカウンタ)
        trending_sell_skip_count: int = 0,
        buy_side_insufficient: bool = False,
        # 240# Toxicity Budget (232# §2.2)
        buy_toxicity: ToxicityAssessment | None = None,
        sell_toxicity: ToxicityAssessment | None = None,
        # 273# I6: halt 解除後 soft gate grace period
        halt_recovery_active: bool = False,
        # 365# P5: SAC Sidecar signal
        sidecar_signal: SidecarSignal | None = None,
    ) -> CycleGateResult:
        """全 per-cycle ゲートを順次評価.

        Args:
            side: "buy" or "sell"
            regime: 現在の market regime (None = unknown)
            vol_ratio: 直近のボラティリティ比率
            inv_net_imbalance: 在庫偏重 [-1, +1]
            is_buy_killed: buy_dynamic_kill が発動中
            is_sell_killed: sell_dynamic_kill が発動中
            spread_jpy: 直近の spread (JPY)、None = 未取得
            mid_price: 直近の mid price、None = 未取得
            price_velocity_bps: 直近60秒の価格速度 (bps)
            trending_sell_skip_count: 連続 trending sell skip カウンタ
            buy_side_insufficient: buy 側の残高不足フラグ (HF4 安全弁用)

        Returns:
            CycleGateResult: blocked=True なら skip, checks に全判定記録
        """
        result = CycleGateResult()
        _regime = regime or "unknown"

        # 491# Composite Risk Score mode flag
        _composite = self._config.composite_risk_enabled

        # 219# unknown regime 連続バイパス: N サイクル連続 unknown で強制通過
        # 277# config 化: config.unknown_regime_max_consecutive
        # 324# M-2: per-side カウンタ参照
        _side_count = self._consecutive_unknown_blocks.get(side, 0)
        _unknown_bypass = (
            _regime == "unknown"
            and _side_count >= self.UNKNOWN_REGIME_MAX_CONSECUTIVE
        )
        if _unknown_bypass:
            logger.warning(
                f"[219#] unknown regime bypass: {_side_count} "
                f"consecutive blocks — forcing through (side={side})"
            )

        # --- Gate 1: unknown_regime_buy_skip ---
        g1 = self._check_unknown_regime_buy(side, _regime, _unknown_bypass)
        result.checks.append(g1)
        if g1.blocked:
            if _composite and not halt_recovery_active:
                w = self._config.composite_risk_weight_unknown_regime
                result.composite_risk_score += w
                result.composite_risk_details.append(f"G1:unknown_buy={w:.2f}")
            elif not halt_recovery_active:
                result.blocked = True
                result.blocking_reason = g1.reason
                self._consecutive_unknown_blocks[side] = _side_count + 1
                return result

        # 229# M-5 fix: Gate 1 通過かつ非 unknown → カウンタリセット
        # Gate 2-6 の early return で reset 漏れを防ぐ
        # 324# M-2: per-side リセット
        if _regime != "unknown":
            self._consecutive_unknown_blocks[side] = 0

        # --- Gate 2: B1' ranging_buy_low_vol ---
        g2 = self._check_ranging_buy_low_vol(side, _regime, vol_ratio)
        result.checks.append(g2)
        if g2.blocked:
            if _composite and not halt_recovery_active:
                w = self._config.composite_risk_weight_ranging_low_vol
                result.composite_risk_score += w
                result.composite_risk_details.append(f"G2:ranging_buy_lv={w:.2f}")
            elif not halt_recovery_active:
                result.blocked = True
                result.blocking_reason = g2.reason
                if side == "buy" and is_buy_killed:
                    g4_spec = self._check_buy_dynamic_kill(side, is_buy_killed)
                    result.speculative_checks.append(g4_spec)
                    if g4_spec.blocked:
                        logger.info(
                            "[451#] compound suppression: ranging_low_vol + "
                            "buy_dynamic_kill would both block"
                        )
                return result

        # --- Gate 2b: 475# ranging_sell_low_vol (buy 側と対称) ---
        g2b = self._check_ranging_sell_low_vol(side, _regime, vol_ratio)
        result.checks.append(g2b)
        if g2b.blocked:
            if _composite and not halt_recovery_active:
                w = self._config.composite_risk_weight_ranging_low_vol
                result.composite_risk_score += w
                result.composite_risk_details.append(f"G2b:ranging_sell_lv={w:.2f}")
            elif not halt_recovery_active:
                result.blocked = True
                result.blocking_reason = g2b.reason
                return result

        # --- Gate 3: trending_sell_skip ---
        g3 = self._check_trending_sell(
            side, _regime, inv_net_imbalance,
            trending_sell_skip_count=trending_sell_skip_count,
            buy_side_insufficient=buy_side_insufficient,
        )
        result.checks.append(g3)
        if g3.blocked:
            if _composite and not halt_recovery_active:
                w = self._config.composite_risk_weight_trending_sell
                result.composite_risk_score += w
                result.composite_risk_details.append(f"G3:trending_sell={w:.2f}")
            elif not halt_recovery_active:
                result.blocked = True
                result.blocking_reason = g3.reason
                return result
        # 196# trending_sell soft mode: propagate offset mult
        if g3.offset_mult is not None:
            result.trending_offset_mult = g3.offset_mult

        # --- Gate 4: buy_dynamic_kill --- (Hard Gate: Boolean 維持)
        _dual_kill = is_buy_killed and is_sell_killed
        _dual_kill_bypass = False
        if _dual_kill:
            if self._config.dual_kill_quiescence_enabled:
                logger.info(
                    f"[249#] DUAL KILL quiescence: both buy/sell killed — "
                    f"resting (side={side}, no bypass)"
                )
            else:
                _dual_kill_bypass = True
                result.dual_kill_bypassed = True
                logger.warning(
                    f"[219#] DUAL KILL bypass: both buy/sell killed — "
                    f"allowing {side} through to break deadlock"
                )

        g4 = self._check_buy_dynamic_kill(
            side, is_buy_killed, _dual_kill_bypass,
        )
        result.checks.append(g4)
        if g4.blocked:
            result.blocked = True
            result.blocking_reason = g4.reason
            return result
        elif buy_toxicity is not None:
            self._apply_toxicity_graded(result, buy_toxicity, g4, side)

        # --- Gate 5: sell_dynamic_kill --- (Hard Gate: Boolean 維持)
        g5 = self._check_sell_dynamic_kill(
            side, is_sell_killed, inv_net_imbalance,
            dual_kill_bypass=_dual_kill_bypass,
        )
        result.checks.append(g5)
        if g5.blocked:
            result.blocked = True
            result.blocking_reason = g5.reason
            return result
        elif sell_toxicity is not None:
            self._apply_toxicity_graded(result, sell_toxicity, g5, side)

        # --- Gate 6: velocity_skip (旧 C4-C5) ---
        g6 = self._check_velocity_skip(side, price_velocity_bps)
        result.checks.append(g6)
        if g6.blocked:
            if _composite and not halt_recovery_active:
                w = self._config.composite_risk_weight_velocity
                result.composite_risk_score += w
                result.composite_risk_details.append(f"G6:velocity={w:.2f}")
            elif not halt_recovery_active:
                result.blocked = True
                result.blocking_reason = g6.reason
                return result

        # --- Gate 7: unknown_regime_sell_skip (旧 C2) ---
        g7 = self._check_unknown_regime_sell(side, _regime, _unknown_bypass)
        result.checks.append(g7)
        if g7.blocked:
            if _composite and not halt_recovery_active:
                w = self._config.composite_risk_weight_unknown_regime
                result.composite_risk_score += w
                result.composite_risk_details.append(f"G7:unknown_sell={w:.2f}")
            elif not halt_recovery_active:
                result.blocked = True
                result.blocking_reason = g7.reason
                self._consecutive_unknown_blocks[side] = _side_count + 1
                return result

        # 229# M-5: non-unknown リセットは Gate 1 直後で実施済み

        # --- Gate 8: narrow_spread_pause --- (Hard Gate: Boolean 維持)
        g8 = self._check_narrow_spread(spread_jpy, mid_price)
        result.checks.append(g8)
        if g8.blocked:
            result.blocked = True
            result.blocking_reason = g8.reason
            return result

        # --- Gate 9: maker_price pre-check --- (Hard Gate: Boolean 維持)
        g9 = self._check_maker_price_precheck(side, spread_jpy)
        result.checks.append(g9)
        if g9.blocked:
            result.blocked = True
            result.blocking_reason = g9.reason
            return result

        # --- 491# Composite Risk Score 閾値判定 ---
        if _composite and result.composite_risk_score > 0:
            _threshold = self._config.composite_risk_threshold
            if result.composite_risk_score >= _threshold:
                result.blocked = True
                result.blocking_reason = "composite_risk_exceeded"
                logger.info(
                    f"[491#] Composite risk block: score={result.composite_risk_score:.2f} "
                    f">= threshold={_threshold:.2f} [{', '.join(result.composite_risk_details)}]"
                )
                return result
            else:
                logger.debug(
                    f"[491#] Composite risk pass: score={result.composite_risk_score:.2f} "
                    f"< threshold={_threshold:.2f} [{', '.join(result.composite_risk_details)}]"
                )

        # --- 365# P5: Sidecar offset injection ---
        if sidecar_signal is not None:
            self._apply_sidecar_offset(result, sidecar_signal, side)

        return result

    # ================================================================
    # 365# P5: SAC Sidecar — offset injection
    # ================================================================

    def _apply_sidecar_offset(
        self,
        result: CycleGateResult,
        signal: SidecarSignal,
        side: str,
    ) -> None:
        """365# P5 / 374# P3.1: SAC sidecar の directional_bias を offset に注入.

        374# Phase 3.1: sidecar_use_v2=True (デフォルト) の場合は
        compute_sidecar_offset_bps_v2 を使用し、SAC 連続値を比例的に変換。
        sidecar_use_v2=False の場合は従来の v1 (離散分類) を維持。

        Config hot-reload 対応: sidecar_max_boost_bps, sidecar_dead_zone,
        sidecar_shaping はランタイム中に YAML 変更で即時反映可能。
        """
        if not self._config.sidecar_enabled:
            return

        if self._config.sidecar_use_v2:
            from scripts.v460.lib.sidecar_types import (
                compute_sidecar_offset_bps_v2,
            )

            offset = compute_sidecar_offset_bps_v2(
                bias=signal.directional_bias,
                side=side,
                max_boost_bps=self._config.sidecar_max_boost_bps,
                dead_zone=self._config.sidecar_dead_zone,
                confidence=signal.confidence,
                shaping=self._config.sidecar_shaping,
            )
        else:
            from scripts.v460.lib.sidecar_types import (
                compute_sidecar_offset_bps,
            )

            offset = compute_sidecar_offset_bps(
                bias=signal.directional_bias,
                side=side,
                confidence=signal.confidence,
            )

        result.sidecar_offset_bps = offset
        result.sidecar_bias = signal.directional_bias
        result.sidecar_direction = signal.direction.name.lower()

        if offset != 0.0:
            _version = "v2" if self._config.sidecar_use_v2 else "v1"
            logger.info(
                f"[374#] Sidecar {side} ({_version}): "
                f"bias={signal.directional_bias:+.3f} "
                f"→ {result.sidecar_direction} → offset={offset:+.4f}bps"
            )

    # ================================================================
    # 240# Toxicity Budget — 段階的応答ヘルパー
    # 241# C-1 fix: gate 非 block 時 (pre-kill ゾーン) に適用
    # ================================================================

    def _apply_toxicity_graded(
        self,
        result: CycleGateResult,
        toxicity: ToxicityAssessment,
        gate_check: GateCheckResult,
        side: str,
    ) -> bool:
        """240# Toxicity Budget 段階的応答を適用.

        241# C-1 fix: Gate 4/5 が block しなかったとき (pre-kill ゾーン) に呼ばれ、
        toxicity assessment が YELLOW/ORANGE なら offset 拡大 / 参加率低下で対応する。

        Glosten-Milgrom の逆選択プレミアム:
        - YELLOW: spread を拡大して逆選択コストを相殺
        - ORANGE: spread 拡大 + 確率的に不参加で期待損失を制限

        Returns:
            True: 段階的応答が適用された
            False: GREEN/KILL → 適用なし
        """
        if toxicity.level in (_ToxicityLevel_GREEN, _ToxicityLevel_KILL):
            return False

        # YELLOW / ORANGE: 段階的応答を適用
        result.toxicity_offset_mult = max(
            result.toxicity_offset_mult, toxicity.offset_mult,
        )
        result.participation_rate = min(
            result.participation_rate, toxicity.participation_rate,
        )
        logger.info(
            f"[240#] Toxicity budget {side}: {toxicity.level.value} "
            f"(score={toxicity.score:.3f}, offset_mult={toxicity.offset_mult:.2f}, "
            f"participation={toxicity.participation_rate:.2f})"
        )
        return True

    # ================================================================
    # 個別ゲート — 各メソッドは GateCheckResult を返す
    # ================================================================

    def _check_unknown_regime_buy(
        self, side: str, regime: str,
        unknown_bypass: bool = False,
    ) -> GateCheckResult:
        """A10: unknown regime での buy スキップ.

        234# balance_forced でも Gate を適用 (gate bypass 廃止).
        235# balance_forced 引数を削除 (dead parameter cleanup).
        """
        if (
            self._config.skip_buy_unknown_regime
            and side == "buy"
            and not unknown_bypass  # 219#
            and regime == "unknown"
        ):
            return GateCheckResult(
                gate_name="unknown_regime_buy",
                blocked=True,
                reason="unknown_regime_buy_skip",
                detail="133# P0-09: unknown regime buy avg -1.384bps",
            )
        return GateCheckResult(gate_name="unknown_regime_buy", blocked=False)

    def _check_ranging_buy_low_vol(
        self, side: str, regime: str, vol_ratio: float | None,
    ) -> GateCheckResult:
        """A11: B1' ranging buy at low vol ハードスキップ.

        195# ソフト化: ranging_buy_low_vol_as_offset=True 時は
        hard skip せず maker_price の low_vol_offset_boost に委譲.
        235# balance_forced 引数を削除 (dead parameter cleanup).
        """
        if (
            self._config.skip_ranging_buy_low_vol
            and side == "buy"
            and regime == "ranging"
            and vol_ratio is not None
            and vol_ratio < self._config.low_vol_threshold
        ):
            # 195#: ソフトモード時は block しない (maker_price low_vol_offset_boost が対応)
            if self._config.ranging_buy_low_vol_as_offset:
                return GateCheckResult(
                    gate_name="ranging_buy_low_vol",
                    blocked=False,
                    detail=(
                        f"195# B1'→offset: vol_ratio={vol_ratio:.4f} "
                        f"< {self._config.low_vol_threshold} "
                        f"(maker_price low_vol_boost で対応)"
                    ),
                )
            return GateCheckResult(
                gate_name="ranging_buy_low_vol",
                blocked=True,
                reason="ranging_low_vol_skip",
                detail=f"169# B1': vol_ratio={vol_ratio:.4f} < {self._config.low_vol_threshold}",
            )
        return GateCheckResult(gate_name="ranging_buy_low_vol", blocked=False)

    def _check_ranging_sell_low_vol(
        self, side: str, regime: str, vol_ratio: float | None,
    ) -> GateCheckResult:
        """475# B1'-sell: ranging sell at low vol — buy 側と対称のフィルタ.

        ranging + sell + low vol → spread 獲得期待が薄く、逆選択リスクが相対的に高い。
        soft mode 時は hard skip せず offset boost に委譲。
        """
        if (
            self._config.skip_ranging_sell_low_vol
            and side == "sell"
            and regime == "ranging"
            and vol_ratio is not None
            and vol_ratio < self._config.low_vol_threshold
        ):
            if self._config.ranging_sell_low_vol_as_offset:
                return GateCheckResult(
                    gate_name="ranging_sell_low_vol",
                    blocked=False,
                    detail=(
                        f"475# B1'-sell→offset: vol_ratio={vol_ratio:.4f} "
                        f"< {self._config.low_vol_threshold} "
                        f"(offset boost で対応)"
                    ),
                )
            return GateCheckResult(
                gate_name="ranging_sell_low_vol",
                blocked=True,
                reason="ranging_sell_low_vol_skip",
                detail=f"475# B1'-sell: vol_ratio={vol_ratio:.4f} < {self._config.low_vol_threshold}",
            )
        return GateCheckResult(gate_name="ranging_sell_low_vol", blocked=False)

    def _check_trending_sell(
        self,
        side: str,
        regime: str,
        inv_net_imbalance: float,
        *,
        trending_sell_skip_count: int = 0,
        buy_side_insufficient: bool = False,
    ) -> GateCheckResult:
        """A12: trending / high_vol regime での sell 抑制 (Sell Asymmetric Gate).

        安全弁 (連続スキップ, HF4, inv_bypass) もここで判定。
        234# balance_forced でも Gate を統一適用 (gate bypass 廃止).
        235# balance_forced 引数を削除 (dead parameter cleanup).

        251# Sell Asymmetric Mode (248# P1):
        Glosten-Milgrom の情報非対称モデルでは、情報優位者が流動性を
        消費する (informed flow) と、MM は逆選択により平均的に損失する。
        trending_up での sell = 「上昇トレンドで BTC を手放す」= directional alpha の放棄。
        high_vol での sell = 情報優位者の活動が最も激しい環境での無防備な流動性提供。
        いずれも「No Trade = 正常」(242#) 思想が適用されるべき環境。

        TP (利確) 例外: sell_guard_inv_bypass_threshold で在庫偏重が大きい場合のみ
        sell を許可し、「過剰在庫の利確」という合理的な sell を残す。
        """
        # 251# Sell Asymmetric: trending + high_vol を統合判定
        _is_trending = regime in ("trending", "trending_up", "trending_down")
        _is_high_vol = regime == "high_vol"

        if not (
            self._config.skip_sell_trending
            and side == "sell"
            and (_is_trending or (_is_high_vol and self._config.sell_asymmetric_high_vol_enabled))
        ):
            return GateCheckResult(gate_name="trending_sell", blocked=False)

        # 176# A: trending_up_only モード
        # 251# high_vol は trending_up_only の制約外 (独立したリスク要因)
        if (
            self._config.skip_sell_trending_up_only
            and _is_trending
            and regime != "trending_up"
        ):
            return GateCheckResult(
                gate_name="trending_sell",
                blocked=False,
                detail=f"176# A: {regime} is not trending_up, allowed",
            )

        # 196# ソフトモード: block しない、offset boost で対応
        # bypass 条件 (HF4, inv_bypass, consecutive) は不要 — sell は常に発注される
        if self._config.trending_sell_as_offset_enabled:
            _boost = self._config.trending_sell_offset_boost_factor
            return GateCheckResult(
                gate_name="trending_sell",
                blocked=False,
                detail=(
                    f"196# trending_sell→offset: {regime} sell "
                    f"→ offset_mult={_boost:.1f}"
                ),
                offset_mult=_boost,
            )

        # 158# §20-B: 連続 skip 安全弁
        _max_consec = self._config.max_consecutive_trending_sell_skip
        if (
            _max_consec > 0
            and trending_sell_skip_count >= _max_consec
        ):
            return GateCheckResult(
                gate_name="trending_sell",
                blocked=False,
                detail=f"158# §20-B: consecutive={trending_sell_skip_count} >= {_max_consec}, safety valve",
            )

        # 166# HF4: buy 側残高不足 → sell 許可でリバランス
        if buy_side_insufficient:
            return GateCheckResult(
                gate_name="trending_sell",
                blocked=False,
                detail="166# HF4: buy side insufficient, forcing sell for rebalance",
            )

        # 171# Guard Paradox: inv bypass
        _inv_bypass_th = self._config.sell_guard_inv_bypass_threshold
        if _inv_bypass_th > 0 and inv_net_imbalance >= _inv_bypass_th:
            return GateCheckResult(
                gate_name="trending_sell",
                blocked=False,
                detail=f"171# inv_bypass: imb={inv_net_imbalance:.3f} >= {_inv_bypass_th}",
            )

        # 基本条件を満たす → skip
        return GateCheckResult(
            gate_name="trending_sell",
            blocked=True,
            reason="trending_sell_skip",
            detail=f"155# §9: {regime} sell avg -0.687bps",
        )

    def _check_buy_dynamic_kill(
        self, side: str, is_buy_killed: bool,
        dual_kill_bypass: bool = False,
    ) -> GateCheckResult:
        """A13: buy 動的 kill.

        234# balance_forced でも Kill Gate は絶対権限 (gate bypass 廃止).
        235# balance_forced 引数を削除 (dead parameter cleanup).
        """
        if (
            self._config.buy_dynamic_kill_enabled
            and side == "buy"
            and not dual_kill_bypass  # 219# dual-kill deadlock breaker
            and is_buy_killed
        ):
            return GateCheckResult(
                gate_name="buy_dynamic_kill",
                blocked=True,
                reason="buy_dynamic_kill",
                detail=f"157# §19: rolling PnL below {self._config.buy_dynamic_kill_threshold_bps}bps",
            )
        return GateCheckResult(gate_name="buy_dynamic_kill", blocked=False)

    def _check_sell_dynamic_kill(
        self,
        side: str,
        is_sell_killed: bool,
        inv_net_imbalance: float,
        dual_kill_bypass: bool = False,
    ) -> GateCheckResult:
        """A14: sell 動的 kill.

        234# balance_forced でも Kill Gate は絶対権限 (gate bypass 廃止).
        235# balance_forced 引数を削除 (dead parameter cleanup).
        """
        # 171# inv bypass
        _inv_bypass = (
            self._config.sell_guard_inv_bypass_threshold > 0
            and inv_net_imbalance >= self._config.sell_guard_inv_bypass_threshold
        )
        if (
            self._config.sell_dynamic_kill_enabled
            and side == "sell"
            and not _inv_bypass
            and not dual_kill_bypass  # 219# dual-kill deadlock breaker
            and is_sell_killed
        ):
            return GateCheckResult(
                gate_name="sell_dynamic_kill",
                blocked=True,
                reason="sell_dynamic_kill",
                detail=f"133# P0-10: rolling PnL below {self._config.sell_dynamic_kill_threshold_bps}bps",
            )
        return GateCheckResult(gate_name="sell_dynamic_kill", blocked=False)

    def _check_velocity_skip(
        self, side: str, price_velocity_bps: float | None,
    ) -> GateCheckResult:
        """C4-C5: velocity-based skip (旧 skip_gate_evaluator 内).

        NOTE: price_velocity_bps は Gate 評価前に外部で取得されている必要がある。
        210# H3: orchestrator から MakerPriceCalculator.last_mid_trend_bps
        (OB mid 差分ベース instant velocity) が渡される。名前は _60s だが、
        実際は instant velocity。符号規約は同一 (正=上昇)。
        取得不可の場合は None → skip しない。
        195# ソフト化: velocity_skip_as_offset_enabled 時は
        hard skip せず skip_gate_evaluator → executor の offset boost に委譲。
        """
        if price_velocity_bps is None:
            return GateCheckResult(gate_name="velocity_skip", blocked=False)

        # 195#: ソフトモード時はここで block しない
        # (skip_gate_evaluator で offset boost として処理)
        if self._config.velocity_skip_as_offset_enabled:
            return GateCheckResult(gate_name="velocity_skip", blocked=False)

        if (
            self._config.sell_velocity_skip_enabled
            and side == "sell"
            and price_velocity_bps > self._config.sell_velocity_skip_threshold_bps
        ):
            return GateCheckResult(
                gate_name="velocity_skip",
                blocked=True,
                reason="rule_velocity_sell_skip",
                detail=f"165# AS-R1: velocity={price_velocity_bps:.2f}bps > {self._config.sell_velocity_skip_threshold_bps}",
            )

        if (
            self._config.buy_velocity_skip_enabled
            and side == "buy"
            and price_velocity_bps < self._config.buy_velocity_skip_threshold_bps
        ):
            return GateCheckResult(
                gate_name="velocity_skip",
                blocked=True,
                reason="rule_velocity_buy_skip",
                detail=f"165# AS-R1: velocity={price_velocity_bps:.2f}bps < {self._config.buy_velocity_skip_threshold_bps}",
            )

        return GateCheckResult(gate_name="velocity_skip", blocked=False)

    def _check_unknown_regime_sell(
        self, side: str, regime: str,
        unknown_bypass: bool = False,
    ) -> GateCheckResult:
        """C2: unknown regime での sell skip.

        219# unknown_bypass: 連続ブロック超過時の強制通過。
        234# balance_forced でも Gate を適用 (gate bypass 廃止).
        235# balance_forced 引数を削除 (dead parameter cleanup).
        """
        if (
            self._config.skip_sell_unknown_regime
            and side == "sell"
            and not unknown_bypass  # 219# 連続ブロックバイパス
            and regime == "unknown"
        ):
            return GateCheckResult(
                gate_name="unknown_regime_sell",
                blocked=True,
                reason="rule_skip_unknown_sell",
                detail="124# unknown regime sell skip",
            )
        return GateCheckResult(gate_name="unknown_regime_sell", blocked=False)

    def _check_narrow_spread(
        self,
        spread_jpy: float | None,
        mid_price: float | None,
    ) -> GateCheckResult:
        """197# Gate 8: narrow_spread_pause の Gate 統合 (旧 B3).

        spread が閾値未満の場合にサイクルスキップ。
        cached spread を使用するため実際の spread とは若干のラグあり。
        """
        if not self._config.narrow_spread_pause_enabled:
            return GateCheckResult(gate_name="narrow_spread", blocked=False)
        if spread_jpy is None or mid_price is None or mid_price <= 0:
            return GateCheckResult(gate_name="narrow_spread", blocked=False)

        spread_bps = spread_jpy / mid_price * 10000.0
        if spread_bps >= self._config.narrow_spread_pause_bps:
            return GateCheckResult(gate_name="narrow_spread", blocked=False)

        return GateCheckResult(
            gate_name="narrow_spread",
            blocked=True,
            reason="narrow_spread_pause",
            detail=(
                f"197# Gate8: spread={spread_bps:.1f}bps "
                f"< {self._config.narrow_spread_pause_bps}bps"
            ),
        )

    def _check_maker_price_precheck(
        self,
        side: str,
        spread_jpy: float | None,
    ) -> GateCheckResult:
        """197# Gate 9: maker_price ValueError の事前チェック (advisory).

        cached spread を使い、maker_price.compute() が ValueError を
        raise する可能性が高いケースを事前 *検出* するが、ブロックしない。

        blocked=True にすると Gate→compute() 未実行→キャッシュ更新なし
        →永久デッドロック のフィードバックループが発生するため、
        advisory-only (blocked=False)。
        実際の判定は executor の try/except が最終防衛線。
        """
        if spread_jpy is None:
            return GateCheckResult(gate_name="maker_price_pre", blocked=False)

        # D1: spread_too_narrow (min_spread_jpy 未満) — advisory only
        if spread_jpy < self._config.min_spread_jpy:
            return GateCheckResult(
                gate_name="maker_price_pre",
                blocked=False,
                reason="spread_too_narrow",
                detail=(
                    f"197# Gate9-advisory: spread={spread_jpy:.0f}JPY "
                    f"< min={self._config.min_spread_jpy:.0f}JPY "
                    f"(executor try/except が最終防衛)"
                ),
            )

        # D3: sell_guard max_spread 超過 — advisory only
        _max = self._config.sell_max_spread_jpy
        if side == "sell" and _max > 0 and spread_jpy > _max:
            return GateCheckResult(
                gate_name="maker_price_pre",
                blocked=False,
                reason="sell_guard_reject",
                detail=(
                    f"197# Gate9-advisory: sell spread={spread_jpy:.0f}JPY "
                    f"> max={_max:.0f}JPY "
                    f"(executor try/except が最終防衛)"
                ),
            )

        return GateCheckResult(gate_name="maker_price_pre", blocked=False)
