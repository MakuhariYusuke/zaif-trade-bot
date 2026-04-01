"""121# Side 選択モジュール.

FillTestRunner から buy/sell 交互ロジック + Smart Side を分離。
009# §4.2 / 054# S2 / 055# Fix を統合。

責務:
  - buy/sell 交互切替 (片側蓄積防止)
  - Smart Side: imbalance ベースの side 抑制/追従 (054# S2)
  - Rapid exit side 強制 (055# Fix)

市場理論的根拠:
  **Inventory Management Model** — Garman (1976) "Market Microstructure",
  Ho & Stoll (1981) "Optimal Dealer Pricing Under Transactions and Return
  Uncertainty".
  ディーラーの在庫ポジションは中立からの乖離が大きいほど
  リスクが増大する。buy/sell 交互で在庫中立性を維持し、
  Smart Side で偏りが大きいときに中立化方向へ side を誘導する。

  **在庫中立化の離散実装**: Stoll (1978) §3 が示す連続的
  reservation price 調整を、side 選択という離散的意思決定として
  実装。これは Avellaneda-Stoikov (2008) の optimal market maker が
  在庫に応じて bid/ask 非対称に注文を配置する考え方の簡略版。
"""""

from __future__ import annotations

import logging

from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class SideSelector:
    """009# §4.2: buy/sell 交互 + 054# S2 Smart Side.

    120# A5: inventory-aware side hint — 残高不足 side を一時的に回避。
    306# L2: Microprice side selection — 板の質的非対称性で side 決定。
    """

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        # start_side に応じて executed/attempted を設定 (交互ロジック用)
        if config.start_side == "sell":
            initial_side: str | None = "buy"  # → next() が "sell" を返す
        else:
            initial_side = None  # → next() が "buy" を返す
        self._last_executed_side: str | None = initial_side
        self._last_attempted_side: str | None = initial_side
        # 054# S2: 同一 side 連続カウンタ
        self._consecutive_same_side: int = 0
        # 054# S3: Early Exit — rapid exit
        self._rapid_exit_side: str | None = None
        # 120# A5: 残高不足 side の一時凍結 (inventory-aware)
        self._frozen_side: str | None = None
        self._frozen_remaining: int = 0

    @property
    def last_side(self) -> str | None:
        return self._last_executed_side

    @last_side.setter
    def last_side(self, value: str | None) -> None:
        self._last_executed_side = value
        self._last_attempted_side = value

    @property
    def last_executed_side(self) -> str | None:
        return self._last_executed_side

    @last_executed_side.setter
    def last_executed_side(self, value: str | None) -> None:
        self._last_executed_side = value

    @property
    def last_attempted_side(self) -> str | None:
        return self._last_attempted_side

    @last_attempted_side.setter
    def last_attempted_side(self, value: str | None) -> None:
        self._last_attempted_side = value

    @property
    def consecutive_same_side(self) -> int:
        return self._consecutive_same_side

    @property
    def rapid_exit_side(self) -> str | None:
        return self._rapid_exit_side

    @rapid_exit_side.setter
    def rapid_exit_side(self, value: str | None) -> None:
        self._rapid_exit_side = value

    def next(self, imbalance: float = 0.0, microprice_bias_bps: float = 0.0,
             *, spread_bps: float = 0.0, regime: str = "") -> str:
        """次の side を決定: 交互 or Smart Side or Microprice.

        120# A5: frozen_side に該当する side は自動的に反対に迂回。
        309# L2 (308# 修正): microprice_bias_bps > 0 → buy (safety), < 0 → sell (safety)。
        310# C: ガードレール — spread_bps / regime 条件付き有効化。
        Args:
            imbalance: 現在の板不均衡 (Smart Side 用)
            microprice_bias_bps: microprice vs mid の偏向 (bps, 306# L2)
            spread_bps: 現在のスプレッド (bps, 310# C ガードレール用)
            regime: 現在のレジーム文字列 (310# C ガードレール用)
        """
        # 055# Fix #1: S3 rapid exit で決定された side を最優先で返却
        if self._rapid_exit_side is not None:
            forced_side = self._rapid_exit_side
            self._rapid_exit_side = None
            logger.info(f"[early_exit] Rapid exit forcing side={forced_side}")
            return forced_side

        reference_side = self._last_executed_side
        base_side = "buy" if (reference_side is None or reference_side == "sell") else "sell"

        # 634# P1-3: ranging では buy 優先 (sell の無理な試行を減らしつつ profitable な buy を増やす)
        if regime == "ranging" and base_side == "sell":
            if self._consecutive_same_side < self._config.ranging_buy_priority_max_consecutive:
                base_side = "buy"
                logger.debug(f"[634#] regime=ranging prioritizing 'buy' over 'sell' (consecutive={self._consecutive_same_side})")

        # 120# A5: inventory-aware — frozen side は反対に迂回
        if self._frozen_side is not None and self._frozen_remaining > 0:
            if base_side == self._frozen_side:
                alt = "sell" if base_side == "buy" else "buy"
                self._frozen_remaining -= 1
                logger.debug(
                    f"[side_selector] Frozen {self._frozen_side} → "
                    f"diverting to {alt} (remaining={self._frozen_remaining})"
                )
                if self._frozen_remaining <= 0:
                    self._frozen_side = None
                return alt

        # 309# L2: Microprice side selection (308# 盲点1 修正: safety モード)
        # Glosten-Milgrom (1985): informed flow が支配する側に
        # maker が立つと逆選択コストが上昇する。
        # microprice > mid → 買い圧力 → buy に回る (safety: 厚い queue の後方)
        # microprice < mid → 売り圧力 → sell に回る (safety: 厚い queue の後方)
        # 旧実装 (306#) は liveness 優先で逆方向に送っていたが、
        # これは事実上の AS seeker であり理論的に倒錯していた。
        if self._config.microprice_side_enabled:
            # 310# C: ガードレール — 条件未達時はスキップ
            _guardrail_pass = True
            _min_sp = self._config.microprice_side_min_spread_bps
            if _min_sp > 0 and spread_bps < _min_sp:
                _guardrail_pass = False
                logger.debug(
                    f"[microprice_side] guardrail: spread={spread_bps:.1f}bps "
                    f"< min={_min_sp:.1f}bps — skipped"
                )
            _rg = self._config.microprice_side_regime_gate
            if _guardrail_pass and _rg and regime and regime not in _rg:
                _guardrail_pass = False
                logger.debug(
                    f"[microprice_side] guardrail: regime={regime} "
                    f"not in {_rg} — skipped"
                )
            if _guardrail_pass:
                threshold = self._config.microprice_side_threshold
                if microprice_bias_bps > threshold:
                    mp_side = "buy"
                    if mp_side != base_side:
                        logger.info(
                            f"[microprice_side] bias={microprice_bias_bps:+.2f}bps "
                            f"> {threshold} → buy (buy pressure, safety mode)"
                        )
                        return mp_side
                elif microprice_bias_bps < -threshold:
                    mp_side = "sell"
                    if mp_side != base_side:
                        logger.info(
                            f"[microprice_side] bias={microprice_bias_bps:+.2f}bps "
                            f"< -{threshold} → sell (sell pressure, safety mode)"
                        )
                        return mp_side

        if not self._config.smart_side_enabled:
            return base_side

        if self._config.smart_side_mode == "suppress":
            should_suppress = False
            if base_side == "buy" and imbalance < -self._config.imbalance_threshold:
                should_suppress = True
            elif base_side == "sell" and imbalance > self._config.imbalance_threshold:
                should_suppress = True

            if should_suppress:
                if self._consecutive_same_side >= self._config.smart_side_max_consecutive:
                    logger.debug(
                        f"[smart_side] Max consecutive ({self._consecutive_same_side}) reached, "
                        f"forcing {base_side}"
                    )
                    return base_side
                alt_side = self._last_executed_side or ("sell" if base_side == "buy" else "buy")
                logger.info(
                    f"[smart_side] Suppressing {base_side} (imb={imbalance:+.3f}), "
                    f"continuing {alt_side}"
                )
                return alt_side

        elif self._config.smart_side_mode == "follow":
            if abs(imbalance) > self._config.imbalance_threshold:
                follow_side = "buy" if imbalance > 0 else "sell"
                if (
                    follow_side == self._last_executed_side
                    and self._consecutive_same_side >= self._config.smart_side_max_consecutive
                ):
                    return base_side
                return follow_side

        return base_side

    def update_after_attempt(self, side: str) -> None:
        """試行時の side 状態を更新する."""
        if side == self._last_attempted_side:
            self._consecutive_same_side += 1
        else:
            self._consecutive_same_side = 0
        self._last_attempted_side = side
        logger.debug(
            "[side_selector] attempt updated: executed=%s attempted=%s consecutive=%d",
            self._last_executed_side,
            self._last_attempted_side,
            self._consecutive_same_side,
        )

    def update_after_decision(
        self,
        side: str,
        *,
        attempt_already_recorded: bool = False,
    ) -> None:
        """約定成功時の side 状態を更新する."""
        if not attempt_already_recorded:
            self.update_after_attempt(side)
        self._last_executed_side = side
        logger.debug(
            "[side_selector] execution updated: executed=%s attempted=%s consecutive=%d",
            self._last_executed_side,
            self._last_attempted_side,
            self._consecutive_same_side,
        )

    def restore_state(
        self,
        *,
        executed_side: str | None,
        attempted_side: str | None,
    ) -> None:
        """永続化済み state から selector を復元する."""
        self._last_executed_side = executed_side
        self._last_attempted_side = attempted_side if attempted_side is not None else executed_side

    def set_rapid_exit(self, current_side: str) -> None:
        """054# S3: early exit → 反対 side を設定."""
        self._rapid_exit_side = "sell" if current_side == "buy" else "buy"

    def freeze_side(self, side: str, cycles: int = 3) -> None:
        """120# A5: 残高不足 side を一時凍結 (inventory-aware).

        Args:
            side: 凍結する side ("buy" or "sell")
            cycles: 凍結サイクル数 (デフォルト 3)
        """
        self._frozen_side = side
        self._frozen_remaining = cycles
        logger.info(
            f"[side_selector] Freezing {side} for {cycles} cycles "
            f"(120# inventory-aware)"
        )

    def unfreeze_side(self) -> None:
        """120# A5: side 凍結を解除."""
        if self._frozen_side is not None:
            logger.debug(f"[side_selector] Unfreezing {self._frozen_side}")
            self._frozen_side = None
            self._frozen_remaining = 0
