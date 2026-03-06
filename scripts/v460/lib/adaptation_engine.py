"""120# AdaptationEngine — パラメータ適応 & ロットサイズ自動適忚モジュール.

run_fill_test.py FillTestRunner からの God Object 分割:
- _try_auto_adapt (142L) → try_auto_adapt()
- _try_auto_lot_size (59L) → try_auto_lot_size()
- _build_adapt_kwargs (16L) → _build_adapt_kwargs()
- _build_lot_kwargs (22L) → _build_lot_kwargs()
- _update_dynamic_loss_cap (47L) → update_dynamic_loss_cap()
- streaming fill record 読み込みの TTL キャッシュ (メモリリーク修正)

市場理論的根拠:
  **Adaptive Market Hypothesis (AMH)** — Lo (2004) "The Adaptive Markets
  Hypothesis: Market Efficiency from an Evolutionary Perspective".
  市場の効率性は固定ではなく、参加者の学習と適応により時間変化する。
  本モジュールはこの AMH の主張を実装し、実測データから
  市場状況の変化を検出し、パラメータを適応的に調整する。

  **Kelly Criterion 統合** — Kelly (1956) "A New Interpretation of
  Information Rate".
  lot_sizer 経由で Kelly 推定値をロットの天井として使用。
  AMH + Kelly の組合せ: 市場状況に応じた情報の質に比例した
  最適ベットサイズを動的に計算する。

メモリリーク修正:
  _try_auto_adapt / _try_auto_lot_size が毎回全レコードをディスクからロードしていた
  (数千件 → メモリ圧迫)。
  _cached_records + _cache_ts で TTL ベースのキャッシュを導入し、
  同一適応サイクル内の二重読み込みを排除。

型安全: AdaptationResult NamedTuple、Final 定数。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final, NamedTuple, Protocol, Sequence

from scripts.v460.lib.regime_detector import RegimeDetectorLike


# ------------------------------------------------------------------
# 262# Protocol 定義 — type: ignore 排除
# ------------------------------------------------------------------
class _LossCapBalanceLike(Protocol):
    """残高エントリ — .currency / .total で通貨と合計残高を取得."""

    @property
    def currency(self) -> str: ...

    @property
    def total(self) -> float: ...


class LossCapAdapterProtocol(Protocol):
    """update_dynamic_loss_cap が adapter に要求する最小インタフェース.

    NOTE: balance_checker.BalanceAdapterProtocol とは異なる。
    - get_balance() は引数なし (全通貨一括) で .currency/.total を持つ。
    - BalanceAdapterProtocol は get_balance(currency: str) で .free を持つ。
    """

    async def get_current_price(self, symbol: str) -> float | None: ...

    async def get_balance(self) -> Sequence[_LossCapBalanceLike]: ...


class FastFillDefenseLike(Protocol):
    """try_auto_adapt が fast_fill_defense に要求する最小インタフェース."""

    def update_base_offsets(
        self,
        base: float,
        buy: float | None = None,
        sell: float | None = None,
    ) -> None: ...

from ztb.metrics.fill_quality import (
    FillRecord,
    compute_fill_metrics,
    iter_fill_records_glob,
    partition_clean_records,
)

from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)

from scripts.v460.lib.constants import BPS_FACTOR as _BPS_FACTOR

# 定数
_MIN_ORDER_BTC: Final[float] = 0.001
# レコードキャッシュ TTL (秒) — 同一サイクル内の二重ロードを防止
_RECORDS_CACHE_TTL_SEC: Final[float] = 10.0


class AdaptationResult(NamedTuple):
    """適応結果 (offset 変更情報)."""

    base_offset_changed: bool
    buy_offset_changed: bool
    sell_offset_changed: bool
    new_base_offset: float
    new_buy_offset: float | None
    new_sell_offset: float | None


class AdaptationEngine:
    """パラメータ適応 & ロットサイジング — FillTestRunner から分割.

    __slots__ でメモリフットプリントを制御。
    TTL ベースのレコードキャッシュでメモリリークを防止。
    """

    __slots__ = (
        "_config",
        "_yaml_cfg",
        "_results_dir",
        "_cached_records",
        "_cache_ts",
    )

    def __init__(
        self,
        config: FillTestConfig,
        yaml_cfg: dict,
        results_dir: Path,
    ) -> None:
        self._config = config
        self._yaml_cfg = yaml_cfg
        self._results_dir = results_dir
        # 120# メモリリーク修正: レコードキャッシュ (TTL ベース)
        self._cached_records: list[FillRecord] | None = None
        self._cache_ts: float = 0.0

    def _load_clean_records(self) -> list[FillRecord]:
        """clean レコードを TTL キャッシュ付きでロード (メモリリーク修正).

        同一適応サイクル内で _try_auto_adapt + _try_auto_lot_size が
        連続呼出しされる場合、ディスク I/O + メモリ重複を排除。
        """
        now = time.time()
        if self._cached_records is not None and (now - self._cache_ts) < self._config.records_cache_ttl_sec:
            return self._cached_records

        records, _q = partition_clean_records(
            iter_fill_records_glob(str(self._results_dir)),
        )
        self._cached_records = records
        self._cache_ts = now
        return records

    def invalidate_cache(self) -> None:
        """キャッシュ無効化 (新レコード保存後に呼ぶ)."""
        self._cached_records = None
        self._cache_ts = 0.0

    # ------------------------------------------------------------------
    # YAML 設定からの kwargs 構築
    # ------------------------------------------------------------------
    def _build_adapt_kwargs(self) -> dict:
        """YAML adaptation セクションから AdaptationConfig 用 kwargs を構築."""
        adapt_yaml = self._yaml_cfg.get("adaptation", {})
        kwargs: dict = {}
        key_map = {
            "min_fill_rate": "min_fill_rate",
            "max_as_ratio": "max_as_ratio",
            "step_ratio": "step_ratio",
            "min_offset_ratio": "min_offset_ratio",
            "max_offset_ratio": "max_offset_ratio",
            "min_samples": "min_samples",
        }
        for yaml_key, config_key in key_map.items():
            if yaml_key in adapt_yaml:
                kwargs[config_key] = adapt_yaml[yaml_key]
        return kwargs

    def _build_lot_kwargs(self) -> dict:
        """YAML lot_sizing セクションから LotSizingConfig 用 kwargs を構築."""
        lot_yaml = self._yaml_cfg.get("lot_sizing", {})
        safety_yaml = self._yaml_cfg.get("safety", {})
        kwargs: dict = {}
        key_map = {
            "min_lot": "min_lot",
            "lot_step": "lot_step",
            "min_fill_rate": "min_fill_rate",
            "max_as_ratio": "max_as_ratio",
            "min_recent_pnl_bps": "min_recent_pnl_bps",
            "min_samples": "min_samples",
        }
        for yaml_key, config_key in key_map.items():
            if yaml_key in lot_yaml:
                kwargs[config_key] = lot_yaml[yaml_key]
        # safety セクションから損失キャップ
        if "loss_cap_jpy" in safety_yaml:
            kwargs["loss_cap_jpy"] = safety_yaml["loss_cap_jpy"]
        if "loss_cap_warning_ratio" in safety_yaml:
            kwargs["loss_cap_warning_ratio"] = safety_yaml["loss_cap_warning_ratio"]
        # 131# D1: レジーム連動ガード
        if "regime_guard_enabled" in lot_yaml:
            kwargs["regime_guard_enabled"] = lot_yaml["regime_guard_enabled"]
        if "regime_hold_regimes" in lot_yaml:
            kwargs["regime_hold_regimes"] = tuple(lot_yaml["regime_hold_regimes"])
        if "regime_decrease_regimes" in lot_yaml:
            kwargs["regime_decrease_regimes"] = tuple(lot_yaml["regime_decrease_regimes"])
        return kwargs

    # ------------------------------------------------------------------
    # 方策 A: spread_offset_ratio 自動適応
    # ------------------------------------------------------------------
    def try_auto_adapt(
        self,
        total_count: int,
        filled_count: int,
        *,
        base_offset_ratio: float,
        base_offset_ratio_buy: float | None,
        base_offset_ratio_sell: float | None,
        regime_detector: RegimeDetectorLike | None = None,
        fast_fill_defense: FastFillDefenseLike | None = None,
    ) -> AdaptationResult:
        """032# P0: 方策 A — fill メトリクスに基づく spread_offset_ratio 自動適応.

        088# side 分離: buy/sell を独立に最適化。
        120# メモリリーク修正: TTL キャッシュ。

        Returns:
            AdaptationResult — 呼び出し側で offset 更新を反映する。
        """
        result = AdaptationResult(
            base_offset_changed=False,
            buy_offset_changed=False,
            sell_offset_changed=False,
            new_base_offset=base_offset_ratio,
            new_buy_offset=base_offset_ratio_buy,
            new_sell_offset=base_offset_ratio_sell,
        )
        cfg = self._config

        try:
            from scripts.v460.lib.param_adapter import (
                AdaptationConfig,
                compute_adaptation,
                compute_side_adaptation,
            )

            records = self._load_clean_records()
            # 096# rolling window
            if cfg.adapt_recency_window > 0 and len(records) > cfg.adapt_recency_window:
                records = records[-cfg.adapt_recency_window:]
            if len(records) < cfg.min_adapt_samples:
                return result

            # 088# side 分離
            buy_records = [r for r in records if r.side == "buy"]
            sell_records = [r for r in records if r.side == "sell"]

            min_side_samples = max(cfg.adapt_min_side_samples, cfg.min_adapt_samples // 2)

            # 258# F-3: RegimeDetectorLike Protocol — 直接アクセス
            regime_tag = (
                regime_detector.current_regime.value
                if regime_detector is not None
                else "n/a"
            )

            if len(buy_records) >= min_side_samples and len(sell_records) >= min_side_samples:
                buy_metrics = compute_fill_metrics(buy_records)
                sell_metrics = compute_fill_metrics(sell_records)

                buy_offset = base_offset_ratio
                if base_offset_ratio_buy is not None:
                    buy_offset = base_offset_ratio_buy
                sell_offset = base_offset_ratio_sell or base_offset_ratio

                buy_config = AdaptationConfig(
                    current_offset_ratio=buy_offset,
                    **self._build_adapt_kwargs(),
                )
                sell_config = AdaptationConfig(
                    current_offset_ratio=sell_offset,
                    **self._build_adapt_kwargs(),
                )

                side_result = compute_side_adaptation(
                    buy_fill_rate=buy_metrics.fill_rate_p90,
                    buy_as_ratio=buy_metrics.adverse_selection_ratio,
                    buy_sample_count=buy_metrics.total_orders,
                    sell_fill_rate=sell_metrics.fill_rate_p90,
                    sell_as_ratio=sell_metrics.adverse_selection_ratio,
                    sell_sample_count=sell_metrics.total_orders,
                    buy_config=buy_config,
                    sell_config=sell_config,
                    # 306# A1: EV-based adaptation
                    buy_avg_pnl_bps=buy_metrics.post_fill_30s_pnl_mean,
                    sell_avg_pnl_bps=sell_metrics.post_fill_30s_pnl_mean,
                )

                new_buy = base_offset_ratio_buy
                new_sell = base_offset_ratio_sell
                buy_changed = False
                sell_changed = False

                if side_result.buy.changed:
                    new_buy = side_result.buy.new_offset
                    buy_changed = True
                    logger.info(
                        f"[方策A] buy offset: {buy_offset:.4f} → {new_buy:.4f} "
                        f"({side_result.buy.action}) [regime={regime_tag}]"
                    )

                if side_result.sell.changed:
                    new_sell = side_result.sell.new_offset
                    sell_changed = True
                    logger.info(
                        f"[方策A] sell offset: {sell_offset:.4f} → {new_sell:.4f} "
                        f"({side_result.sell.action}) [regime={regime_tag}]"
                    )

                if side_result.any_changed and fast_fill_defense is not None:
                    fast_fill_defense.update_base_offsets(
                        base_offset_ratio,
                        new_buy,
                        new_sell,
                    )

                if not side_result.any_changed:
                    logger.debug(
                        f"[方策A] offset unchanged: "
                        f"buy={side_result.buy.reason}, sell={side_result.sell.reason}"
                    )

                result = AdaptationResult(
                    base_offset_changed=False,
                    buy_offset_changed=buy_changed,
                    sell_offset_changed=sell_changed,
                    new_base_offset=base_offset_ratio,
                    new_buy_offset=new_buy,
                    new_sell_offset=new_sell,
                )
            else:
                # side 別サンプル不足 → 全体で従来ロジック
                all_combined = buy_records + sell_records
                metrics = compute_fill_metrics(all_combined)
                del all_combined

                adapt_config = AdaptationConfig(
                    current_offset_ratio=base_offset_ratio,
                    **self._build_adapt_kwargs(),
                )
                adapt_result = compute_adaptation(
                    fill_rate=metrics.fill_rate_p90,
                    as_ratio=metrics.adverse_selection_ratio,
                    sample_count=metrics.total_orders,
                    config=adapt_config,
                )

                if adapt_result.changed:
                    new_base = adapt_result.new_offset
                    new_sell = base_offset_ratio_sell
                    if new_sell is not None and base_offset_ratio > 0:
                        ratio = new_base / base_offset_ratio
                        new_sell = min(
                            new_sell * ratio, cfg.max_offset_ratio,
                        )
                    logger.info(
                        f"[方策A] offset adapted (combined): "
                        f"{base_offset_ratio:.4f} → {new_base:.4f} "
                        f"({adapt_result.action}: {adapt_result.reason})"
                    )
                    if fast_fill_defense is not None:
                        fast_fill_defense.update_base_offsets(
                            new_base,
                            base_offset_ratio_buy,
                            new_sell,
                        )
                    result = AdaptationResult(
                        base_offset_changed=True,
                        buy_offset_changed=False,
                        sell_offset_changed=new_sell != base_offset_ratio_sell,
                        new_base_offset=new_base,
                        new_buy_offset=base_offset_ratio_buy,
                        new_sell_offset=new_sell,
                    )
                else:
                    logger.debug(f"[方策A] offset unchanged: {adapt_result.reason}")

        except Exception as e:
            logger.warning(f"[方策A] Auto-adapt failed (non-fatal): {e}")

        return result

    # ------------------------------------------------------------------
    # 方策 B: 動的ロットサイジング
    # ------------------------------------------------------------------
    def try_auto_lot_size(
        self,
        current_lot: float,
        *,
        regime_detector: RegimeDetectorLike | None = None,
    ) -> tuple[bool, float]:
        """033# 方策 B — fill メトリクスに基づくロットサイズ自動適応.

        120# メモリリーク修正: TTL キャッシュ。

        Returns:
            (changed, new_lot) タプル。
        """
        cfg = self._config
        try:
            from scripts.v460.lib.lot_sizer import (
                KellyEstimate,
                LotSizingConfig,
                compute_cumulative_pnl_jpy,
                compute_kelly_fraction,
                compute_lot_size,
                compute_recent_pnl_bps,
                kelly_recommended_lot,
            )

            records = self._load_clean_records()
            if len(records) < cfg.min_adapt_samples:
                return False, current_lot

            metrics = compute_fill_metrics(records)
            cum_pnl = compute_cumulative_pnl_jpy(records)
            recent_pnl = compute_recent_pnl_bps(
                records, window=cfg.recent_pnl_window,
            )

            lot_config = LotSizingConfig(
                current_lot=current_lot,
                max_lot=cfg.max_lot,
                **self._build_lot_kwargs(),
            )

            # 264# Kelly Criterion 推定
            kelly_est: KellyEstimate | None = None
            kelly_yaml = self._yaml_cfg.get("kelly", {})
            if kelly_yaml.get("enabled", False):
                kelly_est = compute_kelly_fraction(
                    records,
                    min_samples=kelly_yaml.get("min_win_samples", 30),
                    max_fraction=kelly_yaml.get("max_fraction", 0.25),
                    fractional=kelly_yaml.get("fraction", 0.5),
                )
                if kelly_est is not None:
                    equity_btc = kelly_yaml.get("equity_btc", 0.0)
                    if equity_btc > 0:
                        kelly_lot = kelly_recommended_lot(
                            kelly_est, equity_btc, lot_config,
                        )
                        kelly_est = KellyEstimate(
                            win_rate=kelly_est.win_rate,
                            win_loss_ratio=kelly_est.win_loss_ratio,
                            kelly_fraction=kelly_est.kelly_fraction,
                            fractional_kelly=kelly_est.fractional_kelly,
                            recommended_lot=kelly_lot,
                            sample_count=kelly_est.sample_count,
                            reason=kelly_est.reason,
                        )
                        logger.info(
                            f"[方策B] Kelly: lot={kelly_lot:.4f} BTC "
                            f"(f*={kelly_est.kelly_fraction:.4f}, "
                            f"frac={kelly_est.fractional_kelly:.4f}, "
                            f"equity={equity_btc:.4f} BTC)"
                        )

            # 131# D1: レジームタグを取得して compute_lot_size に渡す
            # 258# F-3: RegimeDetectorLike Protocol — 直接アクセス
            regime_tag = (
                regime_detector.current_regime.value
                if regime_detector is not None
                else "n/a"
            )
            lot_result = compute_lot_size(
                fill_rate=metrics.fill_rate_p90,
                as_ratio=metrics.adverse_selection_ratio,
                recent_pnl_bps=recent_pnl,
                cumulative_pnl_jpy=cum_pnl,
                sample_count=metrics.total_orders,
                config=lot_config,
                regime_tag=regime_tag,
                kelly_estimate=kelly_est,
            )

            if lot_result.changed:
                logger.info(
                    f"[方策B] lot adapted: {current_lot:.4f} → {lot_result.new_lot:.4f} BTC "
                    f"({lot_result.action}: {lot_result.reason}) [regime={regime_tag}]"
                )
                return True, lot_result.new_lot
            else:
                logger.debug(f"[方策B] lot unchanged: {lot_result.reason}")
                return False, current_lot

        except Exception as e:
            logger.warning(f"[方策B] Auto lot-size failed (non-fatal): {e}")
            return False, current_lot

    # ------------------------------------------------------------------
    # 動的 loss_cap
    # ------------------------------------------------------------------
    async def update_dynamic_loss_cap(
        self,
        adapter: LossCapAdapterProtocol,
        symbol: str,
    ) -> float | None:
        """041# 動的 loss_cap: API から口座残高を取得し、残高×比率でキャップを算出.

        Returns:
            新しい loss_cap_jpy。失敗時は None (フォールバック値を維持)。
        """
        cfg = self._config
        try:
            btc_price = await adapter.get_current_price(symbol)
            if btc_price is None:
                logger.warning(
                    "[loss_cap] BTC価格取得失敗 — フォールバック値を維持: "
                    f"{cfg.loss_cap_jpy:.0f} JPY"
                )
                return None

            balances = await adapter.get_balance()
            total_jpy = 0.0
            for b in balances:
                currency = b.currency.upper()
                if currency == "JPY":
                    total_jpy += b.total
                elif currency == "BTC":
                    total_jpy += b.total * btc_price

            if total_jpy <= 0:
                logger.warning(
                    "[loss_cap] 残高ゼロまたは取得不可 — フォールバック値を維持: "
                    f"{cfg.loss_cap_jpy:.0f} JPY"
                )
                return None

            new_cap = total_jpy * cfg.loss_cap_ratio
            new_cap = max(cfg.min_loss_cap_jpy, new_cap)
            old_cap = cfg.loss_cap_jpy
            cfg.loss_cap_jpy = new_cap
            logger.info(
                f"[loss_cap] 動的キャップ算出: 残高={total_jpy:.0f} JPY "
                f"× {cfg.loss_cap_ratio:.0%} = {new_cap:.0f} JPY "
                f"(旧: {old_cap:.0f} JPY)"
            )
            return new_cap
        except Exception as e:
            logger.warning(
                f"[loss_cap] 残高取得失敗 — フォールバック値を維持: "
                f"{cfg.loss_cap_jpy:.0f} JPY. error={e}"
            )
            return None
