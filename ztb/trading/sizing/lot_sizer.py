"""Canonical dynamic lot sizing helpers for trading sizing policy.

033# 実装.
264# Kelly Criterion 統合.

概要:
  fill_test の FillMetrics + 累積 PnL から order_quantity を段階的に増減する。
  パフォーマンスが良好なら増量、悪化なら縮小して損失を抑制する。

  264# Kelly Criterion:
    f* = (p·b − q) / b
    - p = 勝率 (post_fill_30s_pnl > 0 の割合)
    - q = 1 − p
    - b = 平均勝ち幅 / 平均負け幅
    Fractional Kelly (f*/2) でリスク調整。ロットの「天井」として使用。

ルール:
  - 収益性: 累積 PnL ≥ 0 AND 直近 PnL 平均 ≥ 0  → 増量候補
  - 約定率: fill_rate ≥ min_fill_rate                → 増量候補
  - AS: AS_ratio ≤ max_as_ratio                      → 増量候補
  - 上記 3 条件が全て満たされたとき → 1 段階増量
  - いずれか悪化時 → 1 段階減量 (最小ロットまで)
  - 損失キャップ接近時 → 強制的に最小ロットへ縮小
  - Kelly 天井: step-based 増量結果を Kelly 推奨ロット以下にクランプ (264#)

安全設計:
  - ハードリミット: min_lot / max_lot でクランプ
  - 段階調整: lot_step ずつ (急激な変更を防止)
  - 損失キャップ統合: 000# §3.9 の 10,000 JPY 実損上限を尊重
  - サンプル不足時は hold (判断材料不足)
 - Kelly Criterion: Fractional Kelly (half-Kelly) + 上限キャップ (264#)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 264# Kelly Criterion データクラス
# ------------------------------------------------------------------
@dataclass
class KellyEstimate:
    """Kelly Criterion 推定結果."""

    win_rate: float  # p: 勝率 (0.0-1.0)
    win_loss_ratio: float  # b: 平均勝ち幅/平均負け幅
    kelly_fraction: float  # f*: 理論的最適ベット比率
    fractional_kelly: float  # f*/fraction で調整後
    recommended_lot: float  # BTC ロットサイズ (clamp 済み)
    sample_count: int  # 使用サンプル数
    reason: str  # 人間向け説明


@dataclass
class LotSizingConfig:
    """動的ロットサイジングの設定.

    347# 有効化必要残高目安:
    - confidence_lot: 最低 3 mBTC (min_lot切下げ後 0.0005 BTC shrink が機能)
    - lot_sizer (step-based): 最低 5 mBTC (0.002 lot で運用開始)
    - Kelly Criterion: 最低 5 mBTC (kelly_equity_btc 設定必須)
    - regime_lot_multiplier: 最低 3 mBTC (0.5x が min_lot 以上)
    """

    # 現在のロットサイズ
    current_lot: float = 0.001

    # ロット制約 (Coincheck BTC 最小= 0.001)
    min_lot: float = 0.001
    max_lot: float = 0.005  # 保守的上限: 5 mBTC
    lot_step: float = 0.00000001  # 348# satoshi 精度 (1e-8 BTC)

    # 増量条件 (全て満たす必要あり)
    min_fill_rate: float = 0.70  # 約定率下限
    max_as_ratio: float = 0.30  # AS 比率上限
    min_recent_pnl_bps: float = 0.0  # 直近平均 PnL 下限 (bps)

    # 損失キャップ (000# §3.9)
    loss_cap_jpy: float = 10_000.0  # 累積実損上限
    loss_cap_warning_ratio: float = 0.7  # 上限の 70% で警告・縮小

    # 適応判定に必要な最小サンプル数
    min_samples: int = 50

    # 131# D1: レジーム連動ロット制御
    # unknown/trending レジームでは増量を抑制 (118# §8.6: unknown AS=60.2%)
    regime_guard_enabled: bool = True
    regime_hold_regimes: tuple[str, ...] = ("unknown",)  # 増量を hold するレジーム
    regime_decrease_regimes: tuple[str, ...] = ()  # 減量を強制するレジーム

    # 264# Kelly Criterion
    kelly_enabled: bool = False  # True で Kelly 天井を適用
    kelly_fraction: float = 0.5  # Fractional Kelly (0.5 = half-Kelly)
    kelly_min_win_samples: int = 30  # Kelly 推定に必要な最小 win+loss サンプル
    kelly_max_fraction: float = 0.25  # f* 上限 (過剰リスク防止)
    kelly_equity_btc: float = 0.0  # 口座残高 BTC (0 → Kelly 無効)


@dataclass
class LotSizingResult:
    """ロットサイジング判断結果."""

    previous_lot: float
    new_lot: float
    action: str  # "increase" | "decrease" | "hold" | "cap_shrink"
    reason: str
    fill_rate: float
    as_ratio: float
    recent_pnl_bps: float
    cumulative_pnl_jpy: float
    sample_count: int

    @property
    def changed(self) -> bool:
        return self.previous_lot != self.new_lot


def compute_lot_size(
    fill_rate: float,
    as_ratio: float,
    recent_pnl_bps: float,
    cumulative_pnl_jpy: float,
    sample_count: int,
    config: LotSizingConfig | None = None,
    regime_tag: str = "n/a",
    kelly_estimate: KellyEstimate | None = None,
) -> LotSizingResult:
    """fill_test メトリクスに基づきロットサイズの推奨値を算出.

    Args:
        fill_rate: 約定率 (0.0-1.0).
        as_ratio: 逆選択比率 (0.0-1.0).
        recent_pnl_bps: 直近 N 件の平均 PnL (bps).
        cumulative_pnl_jpy: 累積実損益 (JPY). 負値 = 損失.
        sample_count: 総約定数.
        config: サイジング設定.
        regime_tag: 131# D1 レジームタグ ("ranging", "trending", "unknown" 等).

    Returns:
        LotSizingResult with recommended new_lot.
    """
    if config is None:
        config = LotSizingConfig()

    current = config.current_lot

    def _make_result(
        new_lot: float, action: str, reason: str
    ) -> LotSizingResult:
        clamped = clamp_lot(new_lot, config)
        return LotSizingResult(
            previous_lot=current,
            new_lot=clamped,
            action=action,
            reason=reason,
            fill_rate=fill_rate,
            as_ratio=as_ratio,
            recent_pnl_bps=recent_pnl_bps,
            cumulative_pnl_jpy=cumulative_pnl_jpy,
            sample_count=sample_count,
        )

    # --- 安全チェック: 損失キャップ接近 ---
    loss_warning_threshold = -config.loss_cap_jpy * config.loss_cap_warning_ratio
    if cumulative_pnl_jpy <= loss_warning_threshold:
        return _make_result(
            config.min_lot,
            "cap_shrink",
            f"損失キャップ接近 (累積={cumulative_pnl_jpy:.0f} JPY, "
            f"警告閾値={loss_warning_threshold:.0f} JPY) → 最小ロットに縮小",
        )

    # --- サンプル不足 → hold ---
    if sample_count < config.min_samples:
        return _make_result(
            current,
            "hold",
            f"サンプル不足 ({sample_count} < {config.min_samples})",
        )

    # --- 131# D1: レジーム連動ガード ---
    if config.regime_guard_enabled and regime_tag != "n/a":
        if regime_tag in config.regime_decrease_regimes:
            if current > config.min_lot:
                new = current - config.lot_step
                return _make_result(
                    new,
                    "decrease",
                    f"レジーム減量ガード (regime={regime_tag}) → ロット減量",
                )
            return _make_result(
                current,
                "hold",
                f"レジーム減量ガード (regime={regime_tag}) だが既に最小ロット",
            )
        if regime_tag in config.regime_hold_regimes:
            return _make_result(
                current,
                "hold",
                f"レジーム増量抑制 (regime={regime_tag}) → hold",
            )

    # --- 条件判定 ---
    good_fill = fill_rate >= config.min_fill_rate
    good_as = as_ratio <= config.max_as_ratio
    good_pnl = recent_pnl_bps >= config.min_recent_pnl_bps

    if good_fill and good_as and good_pnl:
        # 全条件クリア → 増量
        if current >= config.max_lot:
            return _make_result(
                current,
                "hold",
                f"全条件クリアだが上限到達 ({current:.4f} >= {config.max_lot:.4f})",
            )
        new = current + config.lot_step

        # 264# Kelly 天井: step-based 増量結果を Kelly 推奨ロット以下にクランプ
        if kelly_estimate is not None and kelly_estimate.recommended_lot > 0:
            kelly_lot = kelly_estimate.recommended_lot
            if new > kelly_lot:
                new = max(current, kelly_lot)  # 減量はしない (天井のみ)
                return _make_result(
                    new,
                    "hold" if new == current else "increase",
                    f"Kelly 天井適用: step→{current + config.lot_step:.4f} "
                    f"→ Kelly={kelly_lot:.4f} ({kelly_estimate.reason})",
                )

        return _make_result(
            new,
            "increase",
            f"全条件クリア (fill={fill_rate:.1%}, AS={as_ratio:.1%}, "
            f"PnL={recent_pnl_bps:+.2f}bps) → ロット増量",
        )

    # --- いずれかの条件が悪化 ---
    bad_reasons: list[str] = []
    if not good_fill:
        bad_reasons.append(f"fill_rate={fill_rate:.1%} < {config.min_fill_rate:.0%}")
    if not good_as:
        bad_reasons.append(f"AS={as_ratio:.1%} > {config.max_as_ratio:.0%}")
    if not good_pnl:
        bad_reasons.append(f"PnL={recent_pnl_bps:+.2f}bps < {config.min_recent_pnl_bps:+.2f}")

    if current <= config.min_lot:
        # 既に最小 → hold
        return _make_result(
            current,
            "hold",
            f"条件未達 ({', '.join(bad_reasons)}) だが既に最小ロット",
        )

    new = current - config.lot_step
    return _make_result(
        new,
        "decrease",
        f"条件未達 ({', '.join(bad_reasons)}) → ロット減量",
    )


def clamp_lot(value: float, config: LotSizingConfig | None = None) -> float:
    """ロットサイズをハードリミット内に制限.

    348# satoshi 精度: 小数第8位で丸める (1 satoshi = 1e-8 BTC)。
    """
    if config is None:
        config = LotSizingConfig()
    clamped = max(config.min_lot, min(config.max_lot, value))
    return round(clamped, 8)


def compute_cumulative_pnl_jpy(
    records: list,
) -> float:
    """FillRecord リストから累積 PnL (JPY) を算出.

    Args:
        records: FillRecord のリスト.

    Returns:
        累積 PnL (JPY). 負値 = 損失.
    """
    total = 0.0
    for r in records:
        if r.filled and r.post_fill_30s_pnl is not None and r.fill_price:
            # bps → 実額: pnl_bps / BPS_FACTOR * price * quantity
            total += r.post_fill_30s_pnl / 10_000 * r.fill_price * r.order_quantity
    return total


def compute_recent_pnl_bps(
    records: list,
    window: int = 50,
) -> float:
    """直近 window 件の平均 PnL (bps) を算出.

    Args:
        records: FillRecord のリスト (時系列順).
        window: 直近何件を使うか.

    Returns:
        平均 PnL (bps). レコード不足時は 0.0.
    """
    filled = [r for r in records if r.filled and r.post_fill_30s_pnl is not None]
    if not filled:
        return 0.0
    recent = filled[-window:]
    pnl_values = [r.post_fill_30s_pnl for r in recent if r.post_fill_30s_pnl is not None]
    total: float = sum(pnl_values)
    return total / len(recent)


# ------------------------------------------------------------------
# 264# Kelly Criterion — 理論的最適ロットサイジング
# ------------------------------------------------------------------
def compute_kelly_fraction(
    records: list,
    *,
    min_samples: int = 30,
    max_fraction: float = 0.25,
    fractional: float = 0.5,
) -> KellyEstimate | None:
    """FillRecord から Kelly Criterion の最適ベット比率を算出.

    Kelly の公式 (二値アウトカム):
        f* = (p·b − q) / b
    where:
        p = 勝率 (PnL > 0 の約定割合)
        q = 1 − p
        b = 平均勝ちPnL / 平均負けPnL (絶対値比)

    Fractional Kelly: f*/fractional で保守化 (default: half-Kelly = f*/2)。

    Args:
        records: FillRecord のリスト.
        min_samples: Kelly 推定に必要な最小 win+loss 件数.
        max_fraction: f* 上限 (0.25 = 最大25%ベット).
        fractional: Kelly 比率の縮小係数 (0.5 = half-Kelly).

    Returns:
        KellyEstimate or None (サンプル不足時).
    """
    # 約定 + PnL 計測済みのレコードを抽出
    filled = [
        r for r in records
        if r.filled
        and r.post_fill_30s_pnl is not None
        and r.fill_price is not None
        and r.fill_price > 0
    ]

    wins = [r for r in filled if r.post_fill_30s_pnl > 0]
    losses = [r for r in filled if r.post_fill_30s_pnl < 0]

    total_decisive = len(wins) + len(losses)
    if total_decisive < min_samples:
        return None

    # 勝率 p
    p = len(wins) / total_decisive
    q = 1.0 - p

    # 平均勝ち/負けの bps 絶対値
    avg_win_bps = sum(r.post_fill_30s_pnl for r in wins) / len(wins) if wins else 0.0
    avg_loss_bps = abs(
        sum(r.post_fill_30s_pnl for r in losses) / len(losses)
    ) if losses else 0.0

    # b = win/loss ratio (b=0 or loss=0 → Kelly 計算不能)
    if avg_loss_bps < 1e-9:
        # 損失なし → 全力投入 (Kelly=1.0) → max_fraction で制限
        b = float("inf")
        kelly_f = max_fraction
    elif avg_win_bps < 1e-9:
        # 勝ちなし → Kelly ≤ 0
        b = 0.0
        kelly_f = 0.0
    else:
        b = avg_win_bps / avg_loss_bps
        kelly_f = (p * b - q) / b

    # Kelly ≤ 0 → edge がない → ベットしない
    if kelly_f <= 0:
        return KellyEstimate(
            win_rate=p,
            win_loss_ratio=b if math.isfinite(b) else 0.0,
            kelly_fraction=kelly_f,
            fractional_kelly=0.0,
            recommended_lot=0.0,
            sample_count=total_decisive,
            reason=f"Kelly≤0 (no edge): p={p:.3f}, b={b:.3f}, f*={kelly_f:.4f}",
        )

    # Fractional Kelly + 上限キャップ
    frac_kelly = min(kelly_f * fractional, max_fraction)

    return KellyEstimate(
        win_rate=p,
        win_loss_ratio=b if math.isfinite(b) else 999.0,
        kelly_fraction=kelly_f,
        fractional_kelly=frac_kelly,
        recommended_lot=0.0,  # 呼び出し側で equity から算出
        sample_count=total_decisive,
        reason=(
            f"Kelly: p={p:.3f}, b={b:.3f}, f*={kelly_f:.4f}, "
            f"frac={frac_kelly:.4f} ({fractional:.0%} Kelly)"
        ),
    )


def kelly_recommended_lot(
    kelly: KellyEstimate,
    equity_btc: float,
    config: LotSizingConfig | None = None,
) -> float:
    """Kelly 推定結果から推奨ロットサイズ (BTC) を算出.

    lot = fractional_kelly × equity_btc, clamped to [min_lot, max_lot].

    Args:
        kelly: compute_kelly_fraction() の結果.
        equity_btc: 口座の BTC 建て残高 (JPY残高 / BTC価格 + BTC残高).
        config: ロットサイジング設定 (min/max clamp 用).

    Returns:
        推奨ロットサイズ (BTC, lot_step 刻みに丸め).
    """
    if config is None:
        config = LotSizingConfig()

    if kelly.fractional_kelly <= 0 or equity_btc <= 0:
        return config.min_lot

    raw_lot = kelly.fractional_kelly * equity_btc
    # lot_step 刻みに切り捨て
    stepped = math.floor(raw_lot / config.lot_step) * config.lot_step
    return clamp_lot(stepped, config)
