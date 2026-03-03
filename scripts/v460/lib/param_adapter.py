"""
方策 A: パラメータ適応 — fill_test 実測データに基づくスプレッドオフセット自動調整.

028# §3.1 準拠.
032# 実装.

概要:
  SAC モデル自体は変えず、fill_test の FillMetrics から
  spread_offset_ratio を自動調整する。

ルール:
  - fill_rate < min_fill_rate  → offset 増加 (板の内側へ寄せて約定率向上)
  - AS_ratio > max_as_ratio    → offset 減少 (板の外側に退避して逆選択回避)
  - 両方異常 → AS 回避を優先 (損失を抑える方が重要)
  - 変化は step_ratio ずつ段階的に (急激な変更を防止)
  - clamp でハードリミット適用
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class AdaptationConfig:
    """パラメータ適応の設定."""

    # 現在のオフセット比率
    current_offset_ratio: float = 0.05

    # fill_rate 閾値 — 下回ると offset を増やす
    min_fill_rate: float = 0.80
    # AS_ratio 閾値 — 上回ると offset を減らす
    max_as_ratio: float = 0.15

    # 1 回の調整量 (比率の絶対値)
    step_ratio: float = 0.01
    # ハードリミット
    min_offset_ratio: float = 0.01  # これ以上小さくしない
    max_offset_ratio: float = 0.30  # これ以上大きくしない

    # 適応判定に必要な最小サンプル数
    min_samples: int = 50


@dataclass
class AdaptationResult:
    """適応の判断結果."""

    previous_offset: float
    new_offset: float
    action: str  # "increase" | "decrease" | "hold"
    reason: str
    fill_rate: float
    as_ratio: float
    sample_count: int

    @property
    def changed(self) -> bool:
        return self.previous_offset != self.new_offset


def compute_adaptation(
    fill_rate: float,
    as_ratio: float,
    sample_count: int,
    config: AdaptationConfig | None = None,
) -> AdaptationResult:
    """fill_test メトリクスに基づき spread_offset_ratio の推奨値を算出.

    Args:
        fill_rate: 約定率 (0.0-1.0). FillMetrics.fill_rate_p90 を推奨.
        as_ratio: 逆選択比率 (0.0-1.0). FillMetrics.adverse_selection_ratio を推奨.
        sample_count: 総注文数. サンプル不足時は hold.
        config: 適応設定. None なら defaults.

    Returns:
        AdaptationResult with recommended new_offset.
    """
    if config is None:
        config = AdaptationConfig()

    current = config.current_offset_ratio

    # サンプル不足 → hold (判断材料不足で変更しない)
    if sample_count < config.min_samples:
        return AdaptationResult(
            previous_offset=current,
            new_offset=current,
            action="hold",
            reason=f"サンプル不足 ({sample_count} < {config.min_samples})",
            fill_rate=fill_rate,
            as_ratio=as_ratio,
            sample_count=sample_count,
        )

    low_fill = fill_rate < config.min_fill_rate
    high_as = as_ratio > config.max_as_ratio

    if high_as and low_fill:
        # 084# 修正: 両方異常 → hold (デッドロック防止)
        # 旧ロジック: AS 回避優先で offset 縮小 → fill rate さらに低下 → 負のスパイラル
        # 新ロジック: 縮小も増加もせず hold — 別の対策 (time_filter, SkipGate) に委ねる
        return AdaptationResult(
            previous_offset=current,
            new_offset=current,
            action="hold",
            reason=(
                f"AS 超過 ({as_ratio:.1%} > {config.max_as_ratio:.0%}) "
                f"& fill_rate 低下 ({fill_rate:.1%} < {config.min_fill_rate:.0%}) "
                f"→ デッドロック防止のため hold (084#)"
            ),
            fill_rate=fill_rate,
            as_ratio=as_ratio,
            sample_count=sample_count,
        )

    if high_as:
        # AS のみ異常 → offset 縮小 (板の外側へ退避)
        new = max(config.min_offset_ratio, current - config.step_ratio)
        return AdaptationResult(
            previous_offset=current,
            new_offset=new,
            action="decrease",
            reason=(
                f"AS 超過 ({as_ratio:.1%} > {config.max_as_ratio:.0%}) "
                f"→ offset 縮小して逆選択回避"
            ),
            fill_rate=fill_rate,
            as_ratio=as_ratio,
            sample_count=sample_count,
        )

    if low_fill:
        # fill_rate のみ低い → offset 増加 (板の内側へ寄せる)
        new = min(config.max_offset_ratio, current + config.step_ratio)
        return AdaptationResult(
            previous_offset=current,
            new_offset=new,
            action="increase",
            reason=(
                f"fill_rate 低下 ({fill_rate:.1%} < {config.min_fill_rate:.0%}) "
                f"→ offset 増加して約定率向上"
            ),
            fill_rate=fill_rate,
            as_ratio=as_ratio,
            sample_count=sample_count,
        )

    # 両方正常 → hold
    return AdaptationResult(
        previous_offset=current,
        new_offset=current,
        action="hold",
        reason=(
            f"正常範囲内 (fill_rate={fill_rate:.1%}, AS={as_ratio:.1%}) "
            f"→ 変更不要"
        ),
        fill_rate=fill_rate,
        as_ratio=as_ratio,
        sample_count=sample_count,
    )


def clamp_offset(value: float, config: AdaptationConfig | None = None) -> float:
    """offset 値を設定のハードリミット内に制限."""
    if config is None:
        config = AdaptationConfig()
    return max(config.min_offset_ratio, min(config.max_offset_ratio, value))


@dataclass
class SideAdaptationResult:
    """088# side 分離適応の結果."""

    buy: AdaptationResult
    sell: AdaptationResult

    @property
    def any_changed(self) -> bool:
        return self.buy.changed or self.sell.changed


def compute_side_adaptation(
    buy_fill_rate: float,
    buy_as_ratio: float,
    buy_sample_count: int,
    sell_fill_rate: float,
    sell_as_ratio: float,
    sell_sample_count: int,
    *,
    buy_config: AdaptationConfig | None = None,
    sell_config: AdaptationConfig | None = None,
) -> SideAdaptationResult:
    """088# side 分離適応: buy/sell を独立に最適化.

    087# §3.2 P1-3 指摘: buy/sell 混合適応は sell 劣後を埋める速度が遅い。
    side 別メトリクスで offset を別更新する。

    Args:
        buy_fill_rate: buy 側約定率.
        buy_as_ratio: buy 側逆選択比率.
        buy_sample_count: buy 側サンプル数.
        sell_fill_rate: sell 側約定率.
        sell_as_ratio: sell 側逆選択比率.
        sell_sample_count: sell 側サンプル数.
        buy_config: buy 側適応設定.
        sell_config: sell 側適応設定.

    Returns:
        SideAdaptationResult with independent buy/sell recommendations.
    """
    buy_result = compute_adaptation(
        fill_rate=buy_fill_rate,
        as_ratio=buy_as_ratio,
        sample_count=buy_sample_count,
        config=buy_config,
    )
    sell_result = compute_adaptation(
        fill_rate=sell_fill_rate,
        as_ratio=sell_as_ratio,
        sample_count=sell_sample_count,
        config=sell_config,
    )

    if buy_result.changed:
        logger.info(
            f"[方策A] buy offset: {buy_result.previous_offset:.4f} → "
            f"{buy_result.new_offset:.4f} ({buy_result.action}: {buy_result.reason})"
        )
    if sell_result.changed:
        logger.info(
            f"[方策A] sell offset: {sell_result.previous_offset:.4f} → "
            f"{sell_result.new_offset:.4f} ({sell_result.action}: {sell_result.reason})"
        )

    return SideAdaptationResult(buy=buy_result, sell=sell_result)
