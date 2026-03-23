"""323# Mixin: PreOrderAdjustmentsMixin -- 発注前の価格・offset 調整ヘルパー.

fill_cycle_executor.py からの God Object 分割 (323# 分割計画)。
offset 倍率適用・offset 変更後の価格再計算を担当。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    責務: 発注前の offset/price 調整計算のみ。
    FillRecord 構築 / SkipGate 評価 / 監視ロジックを追加しないこと。
"""

from __future__ import annotations


class PreOrderAdjustmentsMixin:
    """発注前の offset/price 調整ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: offset 倍率適用, offset 変更後の price 再計算
      NG: FillRecord 構築, SkipGate, 監視, ループ制御
    MAX LINES: 120
    ────────────────────────────────────────────────────
    323# fill_cycle_executor.py からの抽出
    """

    @staticmethod
    def _recalc_price_with_new_offset(
        side: str,
        order_price: float,
        spread_at_order: float | None,
        old_ratio: float,
        new_ratio: float,
    ) -> float:
        """304# DRY: offset 変更後の maker 価格再計算.

        474# 修正: half-spread (/ 2) バグを解消。
        ベース公式 sell = best_ask - spread * ratio = mid + spread*(0.5 - ratio) から
        mid を介さず直接差分で再計算:
          sell: new_price = price + spread * (old_ratio - new_ratio)
          buy:  new_price = price - spread * (old_ratio - new_ratio)
        spread 不明時は order_price をそのまま返す。
        """
        if spread_at_order is None or spread_at_order <= 0:
            return order_price
        delta = spread_at_order * (old_ratio - new_ratio)
        if side == "buy":
            return round(order_price - delta)
        else:
            return round(order_price + delta)

    @staticmethod
    def _apply_offset_multiplier(
        *,
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        offset_mult: float | None,
        aggressive_when_multiplier_gt_one: bool = False,
    ) -> tuple[float, float, float | None, float | None]:
        """offset 倍率を安全に適用し、更新後の価格・倍率を返す.

        `aggressive_when_multiplier_gt_one=False`:
          multiplier>1.0 で mid から遠ざける (195/196 の保守的発注)
        `aggressive_when_multiplier_gt_one=True`:
          multiplier>1.0 で mid に近づける (193 の EV 前向き調整)
        """
        if (
            offset_mult is None
            or offset_mult <= 0.0
            or spread_at_order is None
            or spread_at_order <= 0
            or order_price <= 0
        ):
            return order_price, effective_offset_ratio, None, None
        if offset_mult == 1.0:
            return order_price, effective_offset_ratio, None, None
        if not aggressive_when_multiplier_gt_one and offset_mult < 1.0:
            return order_price, effective_offset_ratio, None, None

        # 565# P3: Stage max_mult導入 (各段上限2.0)
        # 乗数チェーンの爆発を防ぐため、1段あたりの multiplier を最大2.0に制限
        if offset_mult > 2.0:
            offset_mult = 2.0

        old_offset = spread_at_order * effective_offset_ratio
        new_offset = old_offset * offset_mult
        delta = new_offset - old_offset
        if aggressive_when_multiplier_gt_one:
            if side == "buy":
                order_price = round(order_price + delta)
            else:
                order_price = round(order_price - delta)
        else:
            if side == "buy":
                order_price = round(order_price - delta)
            else:
                order_price = round(order_price + delta)
        return order_price, effective_offset_ratio * offset_mult, offset_mult, delta
