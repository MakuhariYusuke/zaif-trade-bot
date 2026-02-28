"""163# Mixin: FillRecordHelpersMixin -- skip record / lot / regime 共通ヘルパー.

FillTestRunner の run_single_cycle / run_continuous 双方から使われる共有メソッド。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: skip record 生成 + lot 算出 + regime 状態取得 + side 決定
    上記以外のメソッド追加は GOD OBJECT 化を招くため禁止。
    非同期メソッド / API 呼出し / ファイル I/O を追加しないこと。
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Final, Optional

from ztb.utils.git_utils import get_git_sha as _get_shared_git_sha

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from scripts.v460.lib.lot_manager import (
    compute_confidence_lot_factor,
    compute_effective_order_lot,
    resolve_regime_lot_multiplier,
    scale_lot_by_regime,
)
from ztb.metrics.fill_quality import FillRecord, load_fill_records_glob

if TYPE_CHECKING:
    from scripts.v460.lib.cancel_reasons import CancelReason
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.maker_price import MakerPriceCalculator
    from scripts.v460.lib.regime_detector import RegimeDetector
    from scripts.v460.lib.side_selector import SideSelector

logger = logging.getLogger(__name__)

_FILL_RECORD_FIELD_NAMES: Final[frozenset[str]] = frozenset(FillRecord.__dataclass_fields__.keys())
_SKIP_RECORD_RESERVED_FIELDS: Final[frozenset[str]] = frozenset({
    "cycle_id",
    "timestamp",
    "side",
    "order_price",
    "order_quantity",
    "cancelled",
    "cancel_reason",
    "run_id",
    "git_sha",
    "spread_at_order",
    "spread_offset_ratio",
    "regime",
    "balance_forced_switch",
    "ab_test_variant",
})


class FillRecordHelpersMixin:
    """skip record 生成 + ロット算出の共通ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: FillRecord 生成, regime/lot 算出, side 決定委譲
      NG: API 呼出し, 非同期処理, 状態永続化, ファイル I/O
    MAX LINES: 300 (超えたら分割を検討せよ)
    ────────────────────────────────────────────────────
    """

    # 106# R2: bps 換算定数 (1 bps = 1e-4)
    _BPS_FACTOR: int = 10_000

    # ──────────────────────────────────────────────────────────────────
    # 145# §9-#5/7: skip_record / cycle_id ヘルパ (DRY)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _new_cycle_id(prefix: str | None = None) -> str:
        """145# §9-#7: cycle_id 一元生成."""
        base = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return f"{prefix}_{base}" if prefix else base

    def _make_skip_record(
        self,
        *,
        timestamp: float | None = None,
        side: str,
        cancel_reason: CancelReason,
        cycle_id: str | None = None,
        order_quantity: float | None = None,
        order_price: float = 0.0,
        spread_at_order: float | None = None,
        spread_offset_ratio: float | None = None,
        regime: str | None = None,
        balance_forced_switch: bool = False,
        **extra: object,
    ) -> FillRecord:
        """145# §9-#5: skip/監査系 FillRecord 一元生成.

        run_id, git_sha, timestamp は自動設定。
        cancel_reason 文字列は cancel_reasons モジュールの定数を使うこと。
        """
        record = FillRecord(
            cycle_id=cycle_id or self._new_cycle_id(),
            timestamp=time.time() if timestamp is None else timestamp,
            side=side,
            order_price=order_price,
            order_quantity=order_quantity if order_quantity is not None else self._current_lot,
            cancelled=True,
            cancel_reason=cancel_reason,
            run_id=self._run_id,
            git_sha=self._git_sha,
            spread_at_order=spread_at_order,
            spread_offset_ratio=spread_offset_ratio,
            regime=regime,
            balance_forced_switch=balance_forced_switch,
            ab_test_variant=self.config.ab_test_variant or None,  # 158# P1-5
        )
        if not extra:
            return record

        duplicate_keys: list[str] = []
        unknown_keys: list[str] = []
        for key, value in extra.items():
            if key in _SKIP_RECORD_RESERVED_FIELDS:
                duplicate_keys.append(key)
                continue
            if key not in _FILL_RECORD_FIELD_NAMES:
                unknown_keys.append(key)
                continue
            setattr(record, key, value)

        if duplicate_keys:
            logger.debug(
                "FillRecordHelpersMixin._make_skip_record: duplicate extra keys ignored: %s",
                sorted(duplicate_keys),
            )
        if unknown_keys:
            logger.debug(
                "FillRecordHelpersMixin._make_skip_record: unknown extra keys ignored: %s",
                sorted(unknown_keys),
            )
        return record

    def _current_regime_value(self) -> str | None:
        """160# skip record 用: 現在の確定レジーム文字列.

        _make_skip_record の regime 引数に渡し、cancel 系レコードにも
        レジーム情報を伝搬する。regime=None → 43.9% だった問題を解消。
        """
        if self._regime_detector is None:
            return None
        return self._regime_detector.current_regime.value

    def _get_regime_state_fields(self) -> dict:
        """121# A4: regime state persistence — FillTestState に渡す regime 関連フィールド."""
        if self._regime_detector is None:
            return {}
        st = self._regime_detector.get_state()
        return {
            "regime_confirmed": st["confirmed"],
            "regime_stability": st["stability"],
            "regime_prices": st["prices"],
            "regime_raw_history": st["raw_history"],
        }

    def _regime_lot_multiplier(self) -> float:
        """145# §8-#1: 現在のレジームに対応するロット倍率を返す.

        倍率辞書が空 / レジーム未検出 / 該当なしの場合は 1.0 を返す.
        """
        multipliers = self.config.regime_lot_multipliers
        regime_value: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime
            regime_value = regime.value if regime is not None else None
        return resolve_regime_lot_multiplier(multipliers, regime_value=regime_value)

    def _regime_adjusted_lot(self) -> float:
        """143# R-1b: レジーム別ロット倍率を適用した注文ロットを返す.

        regime_lot_multipliers が空の場合は _current_lot をそのまま返す.
        倍率適用後も config.min_order_btc 以上を保証.
        """
        base_lot = self._current_lot
        mult = self._regime_lot_multiplier()
        if mult == 1.0:
            return base_lot
        # 倍率適用 + 安全クランプ (144# #2: config.min_order_btc に統一)
        min_lot = self.config.min_order_btc
        adjusted = scale_lot_by_regime(
            base_lot,
            multiplier=mult,
            min_lot=min_lot,
            max_lot=self.config.max_lot,
        )
        if adjusted != base_lot:
            logger.debug(
                f"[regime_lot] mult={mult:.2f} → lot adjusted: "
                f"{base_lot:.4f} × {mult:.2f} = {adjusted:.4f}"
            )
        return adjusted

    # ------------------------------------------------------------------
    # 151# P3-03: AS 確率連動ロットサイジング (confidence_lot)
    # ------------------------------------------------------------------

    def _confidence_lot_factor(
        self,
        as_prob: float | None,
        *,
        dust_sweep_active: bool = False,
    ) -> float:
        """151# P3-03: AS 確率に基づくロット倍率.

        §10 #5: dust_sweep_active 時は factor=1.0 (端数掃除を妨げない).
        §10 #3: sg.as_prob のみ使用 (mode=pnl は凍結).
        §10 #2: 戻り値は [0, 1] にクランプ (縮小専用の不変条件).

        Returns:
            1.0 (無効時 / 確率なし / dust_sweep) or [floor, 1.0] の倍率.
        """
        mode = self.config.confidence_lot_mode
        if self.config.enable_confidence_lot and mode == "pnl":
            # §10 #3/#6 + §13 #1: mode=pnl は凍結。__post_init__ で弾くが防御的二重ガード
            logger.warning("[confidence_lot] mode='pnl' is frozen; treating as 1.0")
        return compute_confidence_lot_factor(
            enabled=self.config.enable_confidence_lot,
            mode=mode,
            as_prob=as_prob,
            scale=self.config.confidence_lot_scale,
            floor=self.config.confidence_lot_floor,
            dust_sweep_active=dust_sweep_active,
        )

    def _effective_order_lot(
        self,
        regime_lot: float,
        as_prob: float | None = None,
        *,
        dust_sweep_active: bool = False,
    ) -> tuple[float, float]:
        """151# 統合ロット算出: regime_lot × confidence_factor.

        §10 #4: regime_lot は呼び出し元から1回算出して渡す (二重経路回避).

        Args:
            regime_lot: _regime_adjusted_lot() の結果.
            as_prob: SkipGate の AS 確率.
            dust_sweep_active: dust sweep アクティブフラグ.

        Returns:
            (effective_lot, confidence_factor) のタプル.
        """
        conf_factor = self._confidence_lot_factor(
            as_prob, dust_sweep_active=dust_sweep_active,
        )
        lot = compute_effective_order_lot(
            regime_lot=regime_lot,
            confidence_factor=conf_factor,
            min_lot=self.config.min_order_btc,
            max_lot=self.config.max_lot,
        )

        if conf_factor < 1.0:
            logger.debug(
                f"[confidence_lot] as_prob={as_prob}, factor={conf_factor:.2f} "
                f"→ lot={lot:.4f} (regime={regime_lot:.4f})"
            )
        return lot, conf_factor

    # 121# _last_side プロパティ: SideSelector に委譲 (後方互換)
    @property
    def _last_side(self) -> str | None:
        return self._side_selector.last_side

    @_last_side.setter
    def _last_side(self, value: str | None) -> None:
        self._side_selector.last_side = value

    @staticmethod
    def _get_git_sha() -> Optional[str]:
        """現在の git commit short hash を取得."""
        sha = _get_shared_git_sha(cwd=_PROJECT_ROOT)
        if sha == "unknown":
            return None
        return sha[:12]

    def resume_from_existing(self) -> list[FillRecord]:
        """既存 fill_records から状態を復元する (レジューム対応).

        中断→再開時に:
          - _cycle_count を復元
          - _last_side を復元 (片側蓄積防止)
          - 既存レコードを返す (結果集計用)
        """
        existing = load_fill_records_glob(str(self._results_dir))
        if not existing:
            return []

        self._cycle_count = len(existing)
        # 最後のレコードの side を復元
        last_record = existing[-1]
        self._last_side = last_record.side

        # 167# DL-4/P2: 末尾連続 skip カウンタを復元 (再起動耐性)
        # 175# 各カウンタを独立したループで計算 (交互出現時の過大計上を防止)
        _tss_count = 0  # trending_sell_skip
        for rec in reversed(existing):
            if rec.cancel_reason == "trending_sell_skip":
                _tss_count += 1
            else:
                break
        _bfs_count = 0  # balance_forced_skip
        for rec in reversed(existing):
            if rec.cancel_reason == "balance_forced_skip":
                _bfs_count += 1
            else:
                break
        if hasattr(self, "_trending_sell_skip_count"):
            self._trending_sell_skip_count = _tss_count
        if hasattr(self, "_balance_forced_skip_count"):
            self._balance_forced_skip_count = _bfs_count

        logger.info(
            f"Resumed from existing records: n={len(existing)}, "
            f"last_side={self._last_side}, cycle_count={self._cycle_count}, "
            f"trailing_tss={_tss_count}, trailing_bfs={_bfs_count}"
        )
        return existing

    def _next_side(self) -> str:
        """buy/sell を決定 — 121# SideSelector に委譲."""
        return self._side_selector.next(
            imbalance=self._maker_price._last_imbalance,
        )
