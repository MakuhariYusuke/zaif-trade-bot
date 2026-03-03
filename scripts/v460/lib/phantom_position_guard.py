"""237# PhantomPositionGuard — status_unknown 後のファントムポジション検出.

232# §1.6 [HIGH] 対応: status_unknown + cancel失敗系で phantom position が
発生し得る問題への防御。

設計方針:
  - status_unknown 発生時に order_id/side/qty を退避
  - 次サイクル前に deferred re-check で注文状態を再照合
  - 残高差分判定でファントムポジションを検出
  - CRITICAL ログ + 安全側バイアスで後続サイクルを保護

238# セルフレビュー改善:
  - C-1: Protocol 型安全化 (getattr → OrderStatusLike.status 直接参照)
  - C-2: balance_btc snapshot の受け渡し経路追加
  - S-1: TTL 導入 — 300秒超過の stale エントリは自動パージ
  - S-2: phantom_side_veto — 検出側を一時拒否 (adverse selection 防御)
  - S-3: reconcile() の重複 CRITICAL ログ削除 (orchestrator に委譲)
  - S-4: resolved リスト簡素化

責務:
  - status_unknown イベントの記録
  - 遅延型の注文状態再照合
  - 残高ベースの乖離検出
  - ファントム検出時のメトリクス提供
  - 検出側の一時 veto (サイクル数ベース)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class _OrderStatusResult(Protocol):
    """注文ステータスの最小プロトコル (getattr 排除)."""

    @property
    def status(self) -> str: ...


@runtime_checkable
class _BalanceQueryable(Protocol):
    """残高クエリ用 adapter の最小インターフェース."""

    async def get_balance(self, currency: str) -> list: ...
    async def get_order_status(self, order_id: str) -> _OrderStatusResult | None: ...


@dataclass
class PendingReconciliation:
    """status_unknown 発生後の未照合レコード."""

    order_id: str
    side: str
    quantity: float
    price: float
    timestamp: float
    balance_snapshot_btc: float | None = None  # 発注前 BTC 残高
    balance_snapshot_jpy: float | None = None  # 発注前 JPY 残高


@dataclass
class PhantomDetection:
    """ファントムポジション検出結果."""

    order_id: str
    side: str
    quantity: float
    price: float
    detection_method: str  # "order_recheck" | "balance_delta" | "both"
    balance_delta_btc: float | None = None  # 実残高 - 期待残高


class PhantomPositionGuard:
    """237# status_unknown 後のファントムポジション検出ガード.

    ────────────────────────────────────────────────────
    責務: status_unknown 注文の遅延再照合 + 残高乖離検出
         + 検出時の同 side 一時拒否 (adverse selection 防御)
    非責務: 在庫管理, ロット計算, 取引実行
    ────────────────────────────────────────────────────

    市場理論的根拠:
      phantom fill は高ボラティリティ・高ティックフロー時に発生しやすく、
      強い逆選択シグナルである。検出後は同 side の短期 veto で
      逆ポジション蓄積リスクを軽減する (Avellaneda-Stoikov §3.2)。
    """

    # 最小再照合間隔 (秒) — API rate limit 保護
    _MIN_RECONCILE_INTERVAL_SEC: float = 5.0
    # 残高乖離の許容誤差 (BTC) — dust レベルの差分は無視
    _BALANCE_TOLERANCE_BTC: float = 0.0005
    # 238# S-1: 未照合エントリの最大保持時間 (秒)
    _MAX_PENDING_AGE_SEC: float = 300.0
    # 238# S-2: phantom 検出後の同 side veto サイクル数
    _PHANTOM_VETO_CYCLES: int = 3

    def __init__(self) -> None:
        self._pending: list[PendingReconciliation] = []
        self._phantom_count: int = 0
        self._total_reconciled: int = 0
        self._last_reconcile_time: float = 0.0
        # 直近の phantom 検出結果 (ログ/メトリクス用)
        self._last_detections: list[PhantomDetection] = []
        # 238# S-2: phantom 検出後の同 side 一時拒否 (side → 残りサイクル数)
        self._side_veto: dict[str, int] = {}

    @property
    def has_pending(self) -> bool:
        """未照合の status_unknown イベントがあるか."""
        return len(self._pending) > 0

    @property
    def pending_count(self) -> int:
        """未照合イベント数."""
        return len(self._pending)

    @property
    def phantom_detected_count(self) -> int:
        """累積ファントム検出数."""
        return self._phantom_count

    @property
    def total_reconciled(self) -> int:
        """累積照合数."""
        return self._total_reconciled

    def is_side_vetoed(self, side: str) -> bool:
        """238# S-2: phantom 検出後の同 side 一時拒否判定."""
        return self._side_veto.get(side, 0) > 0

    def tick_veto(self) -> None:
        """238# S-2: veto カウンタを 1 サイクル分デクリメント.

        サイクル開始時に呼ばれ、veto 期間が経過したら解除する。
        """
        expired = [s for s, c in self._side_veto.items() if c <= 1]
        for s in expired:
            del self._side_veto[s]
            logger.info(f"[238# phantom_guard] Side veto expired: {s}")
        for s in list(self._side_veto):
            self._side_veto[s] -= 1

    def register_unknown(
        self,
        order_id: str,
        side: str,
        quantity: float,
        price: float,
        *,
        balance_btc: float | None = None,
        balance_jpy: float | None = None,
    ) -> None:
        """status_unknown イベントを記録し、次サイクルでの再照合対象に追加.

        Args:
            order_id: 約定不明の注文 ID
            side: 注文サイド ("buy" / "sell")
            quantity: 注文数量 (BTC)
            price: 注文価格 (JPY)
            balance_btc: 発注前の BTC 残高 (取得済みの場合)
            balance_jpy: 発注前の JPY 残高 (取得済みの場合)
        """
        entry = PendingReconciliation(
            order_id=order_id,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=time.time(),
            balance_snapshot_btc=balance_btc,
            balance_snapshot_jpy=balance_jpy,
        )
        self._pending.append(entry)
        logger.warning(
            f"[237# phantom_guard] Registered status_unknown: "
            f"order={order_id}, side={side}, qty={quantity:.6f}, "
            f"price={price:.0f} — will reconcile before next cycle"
        )

    async def reconcile(self, adapter: _BalanceQueryable) -> list[PhantomDetection]:
        """未照合の注文を再確認し、ファントムポジションを検出.

        238# 改善:
          - S-1: TTL 超過エントリの自動パージ
          - S-2: 検出側に _PHANTOM_VETO_CYCLES サイクルの side veto を設定
          - S-3: CRITICAL ログは orchestrator に委譲 (ここでは INFO/WARNING のみ)
          - S-4: resolved リスト不要 → 直接クリア

        Args:
            adapter: 残高/注文ステータスクエリ可能なアダプタ

        Returns:
            検出されたファントムポジションのリスト (空=問題なし)
        """
        if not self._pending:
            return []

        now = time.time()

        # 238# S-1: TTL 超過エントリの自動パージ
        stale_count = 0
        fresh: list[PendingReconciliation] = []
        for entry in self._pending:
            if now - entry.timestamp > self._MAX_PENDING_AGE_SEC:
                stale_count += 1
                self._total_reconciled += 1  # stale も照合済みカウント
            else:
                fresh.append(entry)
        if stale_count > 0:
            logger.warning(
                f"[238# phantom_guard] Purged {stale_count} stale entries "
                f"(age > {self._MAX_PENDING_AGE_SEC:.0f}s)"
            )
        self._pending = fresh

        if not self._pending:
            return []

        if now - self._last_reconcile_time < self._MIN_RECONCILE_INTERVAL_SEC:
            logger.debug(
                "[237# phantom_guard] Reconcile skipped (rate limit): "
                f"next in {self._MIN_RECONCILE_INTERVAL_SEC - (now - self._last_reconcile_time):.1f}s"
            )
            return []

        self._last_reconcile_time = now
        detections: list[PhantomDetection] = []

        for entry in self._pending:
            detection = await self._reconcile_single(adapter, entry)
            if detection is not None:
                detections.append(detection)
                self._phantom_count += 1
                # 238# S-2: 検出側に veto を設定 (adverse selection 防御)
                self._side_veto[detection.side] = self._PHANTOM_VETO_CYCLES
                logger.warning(
                    f"[238# phantom_guard] Side veto set: "
                    f"{detection.side} blocked for {self._PHANTOM_VETO_CYCLES} cycles"
                )
            self._total_reconciled += 1

        # 238# S-4: 全エントリ照合済み — 直接クリア
        reconciled_count = len(self._pending)
        self._pending.clear()
        self._last_detections = detections

        if not detections:
            logger.info(
                f"[237# phantom_guard] Reconciliation complete: "
                f"{reconciled_count} entries checked, no phantom detected"
            )

        return detections

    async def _reconcile_single(
        self,
        adapter: _BalanceQueryable,
        entry: PendingReconciliation,
    ) -> PhantomDetection | None:
        """1件の status_unknown 注文を再照合.

        Phase 1: 注文ステータス再確認 (order_id が分かっている場合)
        Phase 2: 残高差分確認 (スナップショットがある場合)
        """
        order_filled = False
        balance_mismatch = False
        balance_delta: float | None = None

        # Phase 1: 注文ステータス再確認
        try:
            status = await adapter.get_order_status(entry.order_id)
            if status is not None:
                # 238# C-3: getattr → Protocol.status 直接参照 (型安全)
                _status_str = status.status.lower()
                if "filled" in _status_str or "completed" in _status_str:
                    order_filled = True
                    logger.warning(
                        f"[237# phantom_guard] Order {entry.order_id} "
                        f"confirmed FILLED on deferred recheck "
                        f"(status={_status_str})"
                    )
                elif "cancel" in _status_str or "reject" in _status_str:
                    # 確実にキャンセル済み → ファントムなし
                    logger.info(
                        f"[237# phantom_guard] Order {entry.order_id} "
                        f"confirmed cancelled (status={_status_str})"
                    )
                    return None
                else:
                    logger.info(
                        f"[237# phantom_guard] Order {entry.order_id} "
                        f"status={_status_str} (ambiguous)"
                    )
        except Exception as e:
            logger.warning(
                f"[237# phantom_guard] Order recheck failed for "
                f"{entry.order_id}: {e}"
            )

        # Phase 2: 残高差分確認
        if entry.balance_snapshot_btc is not None:
            try:
                btc_balances = await adapter.get_balance("BTC")
                current_btc = sum(b.free for b in btc_balances) if btc_balances else 0.0
                expected_btc = entry.balance_snapshot_btc
                # buy = BTC 増加, sell = BTC 減少
                if entry.side == "buy":
                    expected_delta = 0.0  # 未約定なら変化なし
                    actual_delta = current_btc - expected_btc
                    if actual_delta > self._BALANCE_TOLERANCE_BTC:
                        balance_mismatch = True
                        balance_delta = actual_delta
                        logger.warning(
                            f"[237# phantom_guard] Balance mismatch (buy): "
                            f"BTC delta={actual_delta:+.6f} "
                            f"(expected ~0, got +{actual_delta:.6f})"
                        )
                else:  # sell
                    actual_delta = expected_btc - current_btc
                    if actual_delta > self._BALANCE_TOLERANCE_BTC:
                        balance_mismatch = True
                        balance_delta = -actual_delta
                        logger.warning(
                            f"[237# phantom_guard] Balance mismatch (sell): "
                            f"BTC delta={-actual_delta:+.6f} "
                            f"(expected ~0, got -{actual_delta:.6f})"
                        )
            except Exception as e:
                logger.warning(
                    f"[237# phantom_guard] Balance check failed: {e}"
                )

        # 判定
        if order_filled and balance_mismatch:
            method = "both"
        elif order_filled:
            method = "order_recheck"
        elif balance_mismatch:
            method = "balance_delta"
        else:
            return None

        return PhantomDetection(
            order_id=entry.order_id,
            side=entry.side,
            quantity=entry.quantity,
            price=entry.price,
            detection_method=method,
            balance_delta_btc=balance_delta,
        )

    def clear(self) -> None:
        """全ての未照合エントリをクリア (テスト用)."""
        self._pending.clear()
        self._last_detections.clear()
        self._side_veto.clear()

    def get_metrics(self) -> dict[str, int | float]:
        """ガードの累積メトリクスを返す."""
        return {
            "pending_count": self.pending_count,
            "phantom_detected_count": self._phantom_count,
            "total_reconciled": self._total_reconciled,
            "side_veto_active": len(self._side_veto),
        }
