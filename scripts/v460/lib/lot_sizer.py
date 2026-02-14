"""
方策 B: 動的ロットサイジング — fill_test 実績に基づく発注数量の段階調整.

033# 実装.

概要:
  fill_test の FillMetrics + 累積 PnL から order_quantity を段階的に増減する。
  パフォーマンスが良好なら増量、悪化なら縮小して損失を抑制する。

ルール:
  - 収益性: 累積 PnL ≥ 0 AND 直近 PnL 平均 ≥ 0  → 増量候補
  - 約定率: fill_rate ≥ min_fill_rate                → 増量候補
  - AS: AS_ratio ≤ max_as_ratio                      → 増量候補
  - 上記 3 条件が全て満たされたとき → 1 段階増量
  - いずれか悪化時 → 1 段階減量 (最小ロットまで)
  - 損失キャップ接近時 → 強制的に最小ロットへ縮小

安全設計:
  - ハードリミット: min_lot / max_lot でクランプ
  - 段階調整: lot_step ずつ (急激な変更を防止)
  - 損失キャップ統合: 000# §3.9 の 10,000 JPY 実損上限を尊重
  - サンプル不足時は hold (判断材料不足)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LotSizingConfig:
    """動的ロットサイジングの設定."""

    # 現在のロットサイズ
    current_lot: float = 0.001

    # ロット制約 (Coincheck BTC 最小= 0.001)
    min_lot: float = 0.001
    max_lot: float = 0.005  # 保守的上限: 5 mBTC
    lot_step: float = 0.001  # 1 mBTC 刻み

    # 増量条件 (全て満たす必要あり)
    min_fill_rate: float = 0.70  # 約定率下限
    max_as_ratio: float = 0.30  # AS 比率上限
    min_recent_pnl_bps: float = 0.0  # 直近平均 PnL 下限 (bps)

    # 損失キャップ (000# §3.9)
    loss_cap_jpy: float = 10_000.0  # 累積実損上限
    loss_cap_warning_ratio: float = 0.7  # 上限の 70% で警告・縮小

    # 適応判定に必要な最小サンプル数
    min_samples: int = 50


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
    config: Optional[LotSizingConfig] = None,
) -> LotSizingResult:
    """fill_test メトリクスに基づきロットサイズの推奨値を算出.

    Args:
        fill_rate: 約定率 (0.0-1.0).
        as_ratio: 逆選択比率 (0.0-1.0).
        recent_pnl_bps: 直近 N 件の平均 PnL (bps).
        cumulative_pnl_jpy: 累積実損益 (JPY). 負値 = 損失.
        sample_count: 総約定数.
        config: サイジング設定.

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


def clamp_lot(value: float, config: Optional[LotSizingConfig] = None) -> float:
    """ロットサイズをハードリミット内に制限.

    Coincheck BTC の精度に合わせて小数第4位で丸める。
    """
    if config is None:
        config = LotSizingConfig()
    clamped = max(config.min_lot, min(config.max_lot, value))
    return round(clamped, 4)


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
            # bps → 実額: pnl_bps * 1e-4 * price * quantity
            total += r.post_fill_30s_pnl * 1e-4 * r.fill_price * r.order_quantity
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
