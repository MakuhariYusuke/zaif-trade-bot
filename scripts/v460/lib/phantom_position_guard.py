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

251# 三値化改善:
  - T-1: ReconcileResult 列挙型 (DETECTED / CLEAN / INCONCLUSIVE)
         API 例外時に pending を破棄せず INCONCLUSIVE として保持→再試行
  - T-2: reconcile_attempts カウンタ付き再試行上限
  - T-3: buy 側 JPY 残高照合 (BTC のみだった Phase 2 を双方向化)

  市場理論: API 障害時に「観測不能 ≠ clean」と区別することで
  Bayesian 事後確率の安易な 0 更新を防ぐ (Gelman et al., BDA3 §3.7)

責務:
  - status_unknown イベントの記録
  - 遅延型の注文状態再照合
  - 残高ベースの乖離検出 (BTC + JPY 双方向)
  - ファントム検出時のメトリクス提供
  - 検出側の一時 veto (サイクル数ベース)
  - INCONCLUSIVE 結果の再試行管理
"""

from __future__ import annotations

import enum
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


class ReconcileResult(enum.Enum):
    """251# 三値化: 照合結果.

    DETECTED    — ファントムポジション確認 (order filled or balance mismatch)
    CLEAN       — 正常 (order cancelled or no mismatch)
    INCONCLUSIVE — API 障害等で判定不能 → 再試行対象として保持
    """

    DETECTED = "detected"
    CLEAN = "clean"
    INCONCLUSIVE = "inconclusive"


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
    # 251# T-2: 照合試行回数 (INCONCLUSIVE 保持の上限管理)
    reconcile_attempts: int = 0


@dataclass
class PhantomDetection:
    """ファントムポジション検出結果."""

    order_id: str
    side: str
    quantity: float
    price: float
    detection_method: str  # "order_recheck" | "balance_delta" | "balance_delta_jpy" | "both"
    balance_delta_btc: float | None = None  # 実残高 - 期待残高 (BTC)
    # 251# T-3: buy 側 JPY 残高乖離
    balance_delta_jpy: float | None = None  # 実残高 - 期待残高 (JPY)


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
    # 251# T-2: INCONCLUSIVE 再試行上限 (超過 → stale と同様に強制パージ)
    _MAX_RECONCILE_ATTEMPTS: int = 3
    # 251# T-3: JPY 残高乖離の許容誤差 (JPY) — dust レベルの差分は無視
    _BALANCE_TOLERANCE_JPY: float = 50.0

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
        # 251# T-1: INCONCLUSIVE エントリは再試行のため保持
        inconclusive: list[PendingReconciliation] = []

        for entry in self._pending:
            entry.reconcile_attempts += 1
            result, detection = await self._reconcile_single(adapter, entry)

            if result == ReconcileResult.DETECTED and detection is not None:
                detections.append(detection)
                self._phantom_count += 1
                # 238# S-2: 検出側に veto を設定 (adverse selection 防御)
                self._side_veto[detection.side] = self._PHANTOM_VETO_CYCLES
                logger.warning(
                    f"[238# phantom_guard] Side veto set: "
                    f"{detection.side} blocked for {self._PHANTOM_VETO_CYCLES} cycles"
                )
                self._total_reconciled += 1
            elif result == ReconcileResult.INCONCLUSIVE:
                # 251# T-2: 再試行上限チェック
                if entry.reconcile_attempts >= self._MAX_RECONCILE_ATTEMPTS:
                    logger.warning(
                        f"[251# phantom_guard] INCONCLUSIVE entry exhausted retries "
                        f"({entry.reconcile_attempts}/{self._MAX_RECONCILE_ATTEMPTS}): "
                        f"order={entry.order_id} — force-purging"
                    )
                    self._total_reconciled += 1
                else:
                    logger.info(
                        f"[251# phantom_guard] INCONCLUSIVE: order={entry.order_id} "
                        f"(attempt {entry.reconcile_attempts}/{self._MAX_RECONCILE_ATTEMPTS}) "
                        f"— retaining for next cycle"
                    )
                    inconclusive.append(entry)
            else:
                # CLEAN
                self._total_reconciled += 1

        # 251# T-1: INCONCLUSIVE エントリのみ保持、残りはクリア
        reconciled_count = len(self._pending) - len(inconclusive)
        self._pending = inconclusive
        self._last_detections = detections

        if inconclusive:
            logger.info(
                f"[251# phantom_guard] Reconciliation: "
                f"{reconciled_count} resolved, {len(inconclusive)} inconclusive retained"
            )
        elif not detections:
            logger.info(
                f"[237# phantom_guard] Reconciliation complete: "
                f"{reconciled_count} entries checked, no phantom detected"
            )

        return detections

    async def _reconcile_single(
        self,
        adapter: _BalanceQueryable,
        entry: PendingReconciliation,
    ) -> tuple[ReconcileResult, PhantomDetection | None]:
        """1件の status_unknown 注文を再照合.

        251# 三値化: API 障害時は INCONCLUSIVE を返し pending 保持。
        「観測不能 ≠ clean」の原則に基づく (Bayesian 安全側推定)。

        Phase 1: 注文ステータス再確認 (order_id が分かっている場合)
        Phase 2a: BTC 残高差分確認 (スナップショットがある場合)
        Phase 2b: JPY 残高差分確認 (251# T-3: buy 側のみ)

        Returns:
            (ReconcileResult, PhantomDetection | None) のタプル
        """
        order_filled = False
        balance_mismatch = False
        balance_delta: float | None = None
        balance_delta_jpy: float | None = None
        # 251# T-1: API 障害フラグ (両 Phase とも失敗した場合 INCONCLUSIVE)
        phase1_api_failed = False
        phase2_api_failed = False

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
                    return ReconcileResult.CLEAN, None
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
            phase1_api_failed = True

        # Phase 2a: BTC 残高差分確認
        if entry.balance_snapshot_btc is not None:
            try:
                btc_balances = await adapter.get_balance("BTC")
                current_btc = sum(b.free for b in btc_balances) if btc_balances else 0.0
                expected_btc = entry.balance_snapshot_btc
                # buy = BTC 増加, sell = BTC 減少
                if entry.side == "buy":
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
                    f"[237# phantom_guard] Balance check (BTC) failed: {e}"
                )
                phase2_api_failed = True

        # Phase 2b: JPY 残高差分確認 (251# T-3: buy 側の支払い検証)
        # buy 約定時は JPY が減少するため、snapshot との差分で検出可能
        if entry.side == "buy" and entry.balance_snapshot_jpy is not None:
            try:
                jpy_balances = await adapter.get_balance("JPY")
                current_jpy = sum(b.free for b in jpy_balances) if jpy_balances else 0.0
                expected_jpy = entry.balance_snapshot_jpy
                # buy 約定 → JPY 減少 (qty * price 分の支払い)
                jpy_decrease = expected_jpy - current_jpy
                expected_cost = entry.quantity * entry.price
                if jpy_decrease > self._BALANCE_TOLERANCE_JPY and jpy_decrease > expected_cost * 0.5:
                    balance_mismatch = True
                    balance_delta_jpy = -jpy_decrease
                    logger.warning(
                        f"[251# phantom_guard] Balance mismatch (buy/JPY): "
                        f"JPY delta={-jpy_decrease:+.0f} "
                        f"(expected cost ~{expected_cost:.0f})"
                    )
            except Exception as e:
                logger.warning(
                    f"[251# phantom_guard] Balance check (JPY) failed: {e}"
                )
                # Phase 2b の失敗は Phase 2a と合算
                if entry.balance_snapshot_btc is None:
                    phase2_api_failed = True
        elif entry.balance_snapshot_btc is None:
            # スナップショットなし = Phase 2 スキップ (失敗ではない)
            pass

        # 251# T-1: 両 Phase とも API 障害 → INCONCLUSIVE
        _has_snapshot = (
            entry.balance_snapshot_btc is not None
            or entry.balance_snapshot_jpy is not None
        )
        if phase1_api_failed and (phase2_api_failed or not _has_snapshot):
            return ReconcileResult.INCONCLUSIVE, None

        # 判定
        if order_filled and balance_mismatch:
            method = "both"
        elif order_filled:
            method = "order_recheck"
        elif balance_mismatch:
            # 251# T-3: JPY のみの検出を区別
            if balance_delta is None and balance_delta_jpy is not None:
                method = "balance_delta_jpy"
            else:
                method = "balance_delta"
        else:
            return ReconcileResult.CLEAN, None

        return ReconcileResult.DETECTED, PhantomDetection(
            order_id=entry.order_id,
            side=entry.side,
            quantity=entry.quantity,
            price=entry.price,
            detection_method=method,
            balance_delta_btc=balance_delta,
            balance_delta_jpy=balance_delta_jpy,
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
