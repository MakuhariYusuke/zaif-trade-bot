"""
G1.1-exec Fill Quality Metrics — 009# §4.3 準拠.

maker 注文の約定品質を評価する指標の算出と Gate 判定を提供する。

E1: fill_rate P90  — 日別 fill rate の 90th percentile
E2: cancel_ratio   — 未約定キャンセル率
E3: queue_wait_median_sec — 発注→約定 の待ち時間中央値
E4: post_fill_30s_pnl    — 約定後 30 秒の mid 価格変動平均 + 片側 t 検定
E5: adverse_selection_ratio — 約定後 30 秒で逆行した注文の割合
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class FillRecord:
    """1 回の maker 注文サイクルの結果.

    009# §4.2 FillRecord スキーマ準拠.
    """

    cycle_id: str
    timestamp: float  # t_submit (epoch)
    side: str  # 'buy' or 'sell'
    order_price: float  # 発注価格
    order_quantity: float  # 発注数量
    fill_price: Optional[float] = None  # 約定価格 (未約定は None)
    filled: bool = False
    cancelled: bool = False
    queue_wait_sec: float = 0.0  # 発注→約定 (or cancel) の秒数
    mid_at_fill: Optional[float] = None  # 約定時の mid price
    mid_30s_after: Optional[float] = None  # 約定 30 秒後の mid price
    post_fill_30s_pnl: Optional[float] = None  # 30 秒後 PnL (bps)
    adverse_selected: Optional[bool] = None  # 30 秒後に逆行したか (CM-3 deadzone 適用後)
    adverse_selected_raw: Optional[bool] = None  # 020# O5: 生の逆行判定 (deadzone 非適用)
    cancel_reason: Optional[str] = None  # CM-2: キャンセル理由 (api_error / timeout / post_only_reject)
    run_id: Optional[str] = None  # 020# O4: 実験ラン識別子
    git_sha: Optional[str] = None  # 020# O4: コミットハッシュ
    # 031# 追加フィールド
    spread_at_order: Optional[float] = None  # 発注時スプレッド (JPY)
    error_message: Optional[str] = None  # エラー詳細メッセージ
    spread_offset_ratio: Optional[float] = None  # 使用した spread_offset_ratio
    # 047# E3: multi-timeframe PnL 計測 (exit timing 最適化のデータ基盤)
    mid_60s_after: Optional[float] = None   # 約定 60 秒後の mid price
    mid_120s_after: Optional[float] = None  # 約定 120 秒後の mid price
    post_fill_60s_pnl: Optional[float] = None   # 60 秒後 PnL (bps)
    post_fill_120s_pnl: Optional[float] = None  # 120 秒後 PnL (bps)
    # 037# レジーム情報 (035# §7 Week 1)
    regime: Optional[str] = None  # FillTestRegime.value (trending/ranging/high_vol/unknown)
    regime_confidence: Optional[float] = None  # 0.0–1.0
    regime_stability: Optional[int] = None  # 連続同一レジーム数
    # 054# AS 予測データ基盤 — orderbook imbalance + spread + mid trend
    orderbook_imbalance: Optional[float] = None   # 板不均衡 [-1, +1] (+1=bid圧倒)
    bid_depth_total: Optional[float] = None       # bid 側合計数量 (BTC)
    ask_depth_total: Optional[float] = None       # ask 側合計数量 (BTC)
    mid_price_trend_5s: Optional[float] = None    # 直前 5s の mid 変化率 (bps)
    spread_bps: Optional[float] = None            # 発注時スプレッド (bps)
    effective_offset_used: Optional[float] = None # 実際に適用された offset 比率
    # 062# SkipGate ML 判定情報
    skip_gate_skipped: Optional[bool] = None      # SkipGate によるスキップ判定
    skip_gate_score: Optional[float] = None       # SkipGate 予測スコア (AS確率 or PnL予測値)
    skip_gate_reason: Optional[str] = None        # SkipGate 判定理由
    # 068# OB 品質 + SkipGate モデル使用ログ
    ob_quality_ok: Optional[bool] = None          # OB 特徴量が品質基準を満たしたか
    ob_age_ms: Optional[float] = None             # OB 取得からの経過ミリ秒
    skip_gate_model_used: Optional[str] = None    # "primary" or "fallback"
    # 084# SkipGate 可観測性改善: P(AS) と使用閾値を直接記録
    skip_gate_as_prob: Optional[float] = None     # AS 確率 (0.0-1.0), mode="as" 時のみ
    skip_gate_threshold_used: Optional[float] = None  # 実際に適用された閾値 (side別解決後)
    # 094# stale order cancel-replace 追跡
    reprice_count: int = 0                        # 1 サイクル内で再発注した回数
    # 100# P1-4: 実際の PnL 計測経過秒数 (early_exit で 30s 未満になる場合の記録)
    actual_measurement_sec: Optional[float] = None  # mid_30s_after の実計測秒数

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FillRecord:
        """Reconstruct from dict."""
        known_keys = set(cls.__dataclass_fields__.keys())
        unknown = set(d.keys()) - known_keys
        if unknown:
            logger.debug(f"FillRecord.from_dict: unknown fields ignored: {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known_keys})


@dataclass
class FillMetrics:
    """G1.1 Gate 指標の算出結果.

    009# §4.3 FillMetrics 準拠.
    """

    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    fill_rate_p90: float = 0.0  # E1
    cancel_ratio: float = 0.0  # E2
    queue_wait_median_sec: float = 0.0  # E3
    post_fill_30s_pnl_mean: float = 0.0  # E4
    post_fill_30s_pnl_pvalue: float = 1.0  # E4 片側 t 検定
    adverse_selection_ratio: float = 0.0  # E5 (deadzone 適用後)
    adverse_selection_ratio_raw: float = 0.0  # E5-raw (020# O5: 並行監視用)

    # 補助情報
    daily_fill_rates: list[float] = field(default_factory=list)
    measurement_days: int = 0
    sample_sufficient: bool = False  # 020# O1: n>=200 & 3暦日 達成フラグ
    # 047# Finding4: AS coverage (分母透明化)
    as_coverage: int = 0       # adverse_selected != None の件数
    as_raw_coverage: int = 0   # adverse_selected_raw != None の件数

    # 116# attempted ベース指標 (skip_gate 除外)
    attempted_orders: int = 0          # skip_gate 除外後のサイクル数
    skip_gate_count: int = 0           # skip_gate でスキップされた件数
    skip_gate_ratio: float = 0.0       # skip_gate_count / total_orders
    attempted_fill_rate: float = 0.0   # filled / attempted_orders
    attempted_cancel_ratio: float = 0.0  # (attempted - filled) / attempted
    overall_fill_rate: float = 0.0     # filled / total_orders (全体ベース)
    # 116# PnL CI (Kill Gate 複合条件用)
    post_fill_30s_pnl_ci_upper: float = 0.0  # 95% CI 上限

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return asdict(self)


# ======================================================================
# Metrics computation
# ======================================================================


def compute_fill_metrics(records: list[FillRecord]) -> FillMetrics:
    """FillRecord のリストから G1.1 Gate 指標を算出.

    009# §2.1 E1-E5 準拠.

    Returns:
        FillMetrics with all E1-E5 indicators computed.
    """
    if not records:
        return FillMetrics()

    total = len(records)
    filled = [r for r in records if r.filled]
    cancelled = [r for r in records if r.cancelled]

    # --- E1: fill_rate P90 (日別) ---
    daily_groups: dict[str, list[FillRecord]] = {}
    for r in records:
        day_key = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        daily_groups.setdefault(day_key, []).append(r)

    daily_fill_rates: list[float] = []
    for _day, day_records in sorted(daily_groups.items()):
        n_day = len(day_records)
        n_filled = sum(1 for r in day_records if r.filled)
        daily_fill_rates.append(n_filled / n_day if n_day > 0 else 0.0)

    fill_rate_p90 = float(np.percentile(daily_fill_rates, 10)) if daily_fill_rates else 0.0
    # NOTE: P90 = "90% of days have fill rate >= this value" = 10th percentile
    # (lower bound of the distribution)

    # --- E2: cancel_ratio ---
    cancel_ratio = len(cancelled) / total if total > 0 else 0.0

    # --- E3: queue_wait_median_sec (filled orders only) ---
    wait_times = [r.queue_wait_sec for r in filled if r.queue_wait_sec > 0]
    queue_wait_median = float(np.median(wait_times)) if wait_times else 0.0

    # --- E4: post_fill_30s_pnl ---
    pnl_values = [
        r.post_fill_30s_pnl for r in filled
        if r.post_fill_30s_pnl is not None
    ]
    if pnl_values:
        pnl_mean = float(np.mean(pnl_values))
        # 片側 t 検定: H0: mean >= 0, H1: mean < 0
        if len(pnl_values) >= 2:
            t_stat, two_sided_p = stats.ttest_1samp(pnl_values, 0.0)
            # 片側 (mean < 0 方向): t_stat < 0 なら p_one = two_sided_p / 2
            if t_stat < 0:
                pnl_pvalue = float(two_sided_p / 2)
            else:
                pnl_pvalue = 1.0 - float(two_sided_p / 2)
        else:
            pnl_pvalue = 1.0  # サンプル不足 → PASS 扱い
    else:
        pnl_mean = 0.0
        pnl_pvalue = 1.0

    # --- E5: adverse_selection_ratio ---
    adverse_records = [
        r for r in filled
        if r.adverse_selected is not None
    ]
    n_adverse = sum(1 for r in adverse_records if r.adverse_selected)
    adverse_ratio = n_adverse / len(adverse_records) if adverse_records else 0.0

    # --- E5-raw: 020# O5 — deadzone 非適用の生データ並行監視 ---
    adverse_raw_records = [
        r for r in filled
        if r.adverse_selected_raw is not None
    ]
    n_adverse_raw = sum(1 for r in adverse_raw_records if r.adverse_selected_raw)
    adverse_ratio_raw = n_adverse_raw / len(adverse_raw_records) if adverse_raw_records else adverse_ratio

    # --- 020# O1: サンプル充足判定 ---
    # 047# Finding3: 3日ではなく 7日を要求 (000# §3.3 準拠)
    sample_sufficient = total >= 200 and len(daily_fill_rates) >= 7

    # --- 116# attempted ベース指標 ---
    skip_gate_records = [
        r for r in records
        if getattr(r, "skip_gate_skipped", None) is True
        or getattr(r, "cancel_reason", None) == "skip_gate"
    ]
    skip_gate_count = len(skip_gate_records)
    attempted_orders = total - skip_gate_count
    skip_gate_ratio = skip_gate_count / total if total > 0 else 0.0
    attempted_fill_rate = len(filled) / attempted_orders if attempted_orders > 0 else 0.0
    attempted_cancel_ratio = (
        (attempted_orders - len(filled)) / attempted_orders
        if attempted_orders > 0
        else 0.0
    )
    overall_fill_rate = len(filled) / total if total > 0 else 0.0

    # --- 116# PnL CI 上限 (Kill Gate 複合条件用) ---
    pnl_ci_upper = 0.0
    if pnl_values and len(pnl_values) >= 2:
        se = float(np.std(pnl_values, ddof=1)) / np.sqrt(len(pnl_values))
        t_crit = float(stats.t.ppf(0.975, df=len(pnl_values) - 1))
        pnl_ci_upper = pnl_mean + t_crit * se

    return FillMetrics(
        total_orders=total,
        filled_orders=len(filled),
        cancelled_orders=len(cancelled),
        fill_rate_p90=fill_rate_p90,
        cancel_ratio=cancel_ratio,
        queue_wait_median_sec=queue_wait_median,
        post_fill_30s_pnl_mean=pnl_mean,
        post_fill_30s_pnl_pvalue=pnl_pvalue,
        adverse_selection_ratio=adverse_ratio,
        adverse_selection_ratio_raw=adverse_ratio_raw,
        daily_fill_rates=daily_fill_rates,
        measurement_days=len(daily_fill_rates),
        sample_sufficient=sample_sufficient,
        # 047# Finding4: AS coverage
        as_coverage=len(adverse_records),
        as_raw_coverage=len(adverse_raw_records),
        # 116# attempted
        attempted_orders=attempted_orders,
        skip_gate_count=skip_gate_count,
        skip_gate_ratio=skip_gate_ratio,
        attempted_fill_rate=attempted_fill_rate,
        attempted_cancel_ratio=attempted_cancel_ratio,
        overall_fill_rate=overall_fill_rate,
        post_fill_30s_pnl_ci_upper=pnl_ci_upper,
    )


# ======================================================================
# G1.1 Gate Judgment
# ======================================================================


def g1_1_judgment(
    metrics: FillMetrics,
    thresholds: dict,
    records: Optional[list[FillRecord]] = None,
) -> dict:
    """G1.1 Gate 合否判定.

    009# §2.1 / 000# §3.3 準拠.
    092# 追加: E6 (round-trip mean PnL), E7 (net inventory drift).

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_1_exec`` セクション.
        records: round-trip KPI 算出用の FillRecord リスト (省略時は E6/E7 スキップ).

    Returns:
        dict with gate_result, per-check details.
    """
    checks: dict[str, dict] = {}

    # E1: fill_rate P90
    min_fill = thresholds.get("min_fill_rate_p90", 0.90)
    checks["E1_fill_rate_p90"] = {
        "value": metrics.fill_rate_p90,
        "threshold": min_fill,
        "pass": metrics.fill_rate_p90 >= min_fill,
    }

    # E2: cancel_ratio
    max_cancel = thresholds.get("max_cancel_ratio", 0.30)
    checks["E2_cancel_ratio"] = {
        "value": metrics.cancel_ratio,
        "threshold": max_cancel,
        "pass": metrics.cancel_ratio <= max_cancel,
    }

    # E3: queue_wait_median_sec
    max_wait = thresholds.get("max_queue_wait_median_sec", 60)
    checks["E3_queue_wait_median"] = {
        "value": metrics.queue_wait_median_sec,
        "threshold": max_wait,
        "pass": metrics.queue_wait_median_sec <= max_wait,
    }

    # E4: post_fill_30s_pnl (§2.4 統計的補足)
    min_pnl = thresholds.get("min_post_fill_30s_pnl", 0.0)
    pnl_pass: bool
    if metrics.post_fill_30s_pnl_mean >= min_pnl:
        pnl_pass = True
    elif metrics.post_fill_30s_pnl_pvalue >= 0.05:
        pnl_pass = True  # 負だが統計的に有意でない → PASS
    else:
        pnl_pass = False  # 負かつ有意 → systemic adverse selection
    checks["E4_post_fill_pnl"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "threshold": min_pnl,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "pass": pnl_pass,
    }

    # E5: adverse_selection_ratio
    max_adverse = thresholds.get("max_adverse_selection_ratio", 0.20)
    checks["E5_adverse_selection"] = {
        "value": metrics.adverse_selection_ratio,
        "threshold": max_adverse,
        "pass": metrics.adverse_selection_ratio <= max_adverse,
    }

    # E5-raw: 020# O5 — deadzone 非適用の並行監視
    checks["E5_adverse_selection_raw"] = {
        "value": metrics.adverse_selection_ratio_raw,
        "threshold": max_adverse,
        "pass": metrics.adverse_selection_ratio_raw <= max_adverse,
        "informational": True,  # Gate 判定には影響しない (監視用)
    }

    # 092# E6: round-trip mean PnL (087# P1-1 / 083# §4.2-3)
    # mean が負でもテール損失管理の監視として重要
    if records is not None:
        filled_recs = [r for r in records if r.filled]
        if len(filled_recs) >= 2:
            rt_metrics, _ = compute_round_trip_metrics(records)
            if rt_metrics.total_pairs > 0:
                min_rt_pnl = thresholds.get("min_round_trip_pnl_mean", -2.0)
                checks["E6_round_trip_pnl"] = {
                    "value": rt_metrics.pnl_mean_bps,
                    "threshold": min_rt_pnl,
                    "pass": rt_metrics.pnl_mean_bps >= min_rt_pnl,
                    "pairs": rt_metrics.total_pairs,
                    "median": rt_metrics.pnl_median_bps,
                    "total_jpy": rt_metrics.pnl_total_jpy,
                    "informational": True,  # 当面は監視用、安定したら Gate 昇格
                }

                # E7: net inventory drift — 在庫偏り警告
                max_inventory = thresholds.get("max_net_inventory", 5)
                checks["E7_net_inventory"] = {
                    "value": abs(rt_metrics.net_inventory),
                    "threshold": max_inventory,
                    "pass": abs(rt_metrics.net_inventory) <= max_inventory,
                    "net_inventory": rt_metrics.net_inventory,
                    "unpaired_buys": rt_metrics.unpaired_buys,
                    "unpaired_sells": rt_metrics.unpaired_sells,
                    "informational": True,  # 監視用
                }

    # Gate 判定には informational=True のチェックは含めない
    gate_checks = {k: v for k, v in checks.items() if not v.get("informational")}
    all_pass = all(c["pass"] for c in gate_checks.values())

    # 047# Finding3: PROVISIONAL/INTERIM/FINAL の 3 段階判定
    # - PROVISIONAL: n<200 or days<3
    # - INTERIM: n>=200 & 3<=days<7 (暗定判定)
    # - FINAL: n>=200 & days>=7 (000# §3.3 準拠)
    if metrics.sample_sufficient:
        judgment_type = "FINAL"  # n>=200 & days>=7
    elif metrics.total_orders >= 200 and metrics.measurement_days >= 3:
        judgment_type = "INTERIM"  # 十分なサンプルだが 7 日未満
    else:
        judgment_type = "PROVISIONAL"

    return {
        "gate": "G1.1-exec",
        "gate_result": "PASS" if all_pass else "FAIL",
        "judgment_type": judgment_type,  # 020# O1 / 047# Finding3
        "sample_sufficient": metrics.sample_sufficient,
        "checks": checks,
        "metrics_summary": metrics.to_dict(),
    }


# ======================================================================
# 116# Two-Stage Gate Judgment (115# review)
# ======================================================================


def g1_1_quick_judgment(
    metrics: FillMetrics,
    thresholds: dict,
    cumulative_loss_jpy: float = 0.0,
) -> dict:
    """G1.1-quick (72h Kill Gate) 判定.

    116# 実装 / 115# レビュー反映.
    明らかに不成立な戦略を早期棄却する。

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_1_quick_exec`` セクション.
        cumulative_loss_jpy: fill_test の累積実損 (JPY, 正値=損失).

    Returns:
        dict with gate_result (PASS/FAIL/WATCH), per-check details.
    """
    checks: dict[str, dict] = {}

    # K1: attempted_fill_rate
    min_att_fill = thresholds.get("min_attempted_fill_rate", 0.60)
    checks["K1_attempted_fill_rate"] = {
        "value": metrics.attempted_fill_rate,
        "threshold": min_att_fill,
        "pass": metrics.attempted_fill_rate >= min_att_fill,
    }

    # K2: attempted_cancel_ratio
    max_att_cancel = thresholds.get("max_attempted_cancel_ratio", 0.40)
    checks["K2_attempted_cancel_ratio"] = {
        "value": metrics.attempted_cancel_ratio,
        "threshold": max_att_cancel,
        "pass": metrics.attempted_cancel_ratio <= max_att_cancel,
    }

    # K3: queue_wait_median
    max_wait = thresholds.get("max_queue_wait_median_sec", 120)
    checks["K3_queue_wait_median"] = {
        "value": metrics.queue_wait_median_sec,
        "threshold": max_wait,
        "pass": metrics.queue_wait_median_sec <= max_wait,
    }

    # K4: PnL 複合条件 — p < threshold かつ mean <= threshold で FAIL
    # 115# Q10.2(C): 単独 p 値判定は不十分。効果量条件を併設。
    pnl_kill_p = thresholds.get("pnl_kill_p_threshold", 0.02)
    pnl_kill_mean = thresholds.get("pnl_kill_mean_threshold", -0.8)
    pnl_is_significant = metrics.post_fill_30s_pnl_pvalue < pnl_kill_p
    pnl_is_large_loss = metrics.post_fill_30s_pnl_mean <= pnl_kill_mean
    # FAIL = 両条件同時成立
    k4_pass = not (pnl_is_significant and pnl_is_large_loss)
    checks["K4_pnl_kill"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "threshold_p": pnl_kill_p,
        "threshold_mean": pnl_kill_mean,
        "significant": pnl_is_significant,
        "large_loss": pnl_is_large_loss,
        "pass": k4_pass,
    }

    # K5: 累積実損
    max_loss = thresholds.get("max_cumulative_loss_jpy", 10000)
    checks["K5_cumulative_loss"] = {
        "value": cumulative_loss_jpy,
        "threshold": max_loss,
        "pass": cumulative_loss_jpy < max_loss,
    }

    # K6: skip_gate_ratio
    max_skip = thresholds.get("max_skip_gate_ratio", 0.25)
    checks["K6_skip_gate_ratio"] = {
        "value": metrics.skip_gate_ratio,
        "threshold": max_skip,
        "pass": metrics.skip_gate_ratio <= max_skip,
    }

    all_pass = all(c["pass"] for c in checks.values())

    # 115# Q10.4: Watch 層 — Kill には至らないが黄信号
    pnl_watch_p = thresholds.get("pnl_watch_p_threshold", 0.05)
    pnl_watch_mean = thresholds.get("pnl_watch_mean_threshold", -0.3)
    is_watch = (
        all_pass
        and metrics.post_fill_30s_pnl_pvalue < pnl_watch_p
        and metrics.post_fill_30s_pnl_mean < pnl_watch_mean
    )

    if not all_pass:
        gate_result = "FAIL"
    elif is_watch:
        gate_result = "WATCH"
    else:
        gate_result = "PASS"

    return {
        "gate": "G1.1-quick",
        "gate_result": gate_result,
        "checks": checks,
        "watch": is_watch,
        "watch_detail": {
            "pnl_mean": metrics.post_fill_30s_pnl_mean,
            "pnl_pvalue": metrics.post_fill_30s_pnl_pvalue,
            "watch_thresholds": {"p": pnl_watch_p, "mean": pnl_watch_mean},
        } if is_watch else None,
        "metrics_summary": metrics.to_dict(),
    }


def g1_2_full_judgment(
    metrics: FillMetrics,
    thresholds: dict,
) -> dict:
    """G1.2-full (168h Qualification Gate) 判定.

    116# 実装 / 115# レビュー反映.
    完全な品質適格性の確認。

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_2_full_exec`` セクション.

    Returns:
        dict with gate_result (PASS/FAIL), per-check details.
    """
    checks: dict[str, dict] = {}

    # F1: attempted_fill_rate
    min_att_fill = thresholds.get("min_attempted_fill_rate", 0.70)
    checks["F1_attempted_fill_rate"] = {
        "value": metrics.attempted_fill_rate,
        "threshold": min_att_fill,
        "pass": metrics.attempted_fill_rate >= min_att_fill,
    }

    # F1b: overall_fill_rate (115# Q10.2(A): SkipGate 過剰回避)
    min_overall_fill = thresholds.get("min_overall_fill_rate", 0.62)
    checks["F1b_overall_fill_rate"] = {
        "value": metrics.overall_fill_rate,
        "threshold": min_overall_fill,
        "pass": metrics.overall_fill_rate >= min_overall_fill,
    }

    # F2: attempted_cancel_ratio
    max_att_cancel = thresholds.get("max_attempted_cancel_ratio", 0.30)
    checks["F2_attempted_cancel_ratio"] = {
        "value": metrics.attempted_cancel_ratio,
        "threshold": max_att_cancel,
        "pass": metrics.attempted_cancel_ratio <= max_att_cancel,
    }

    # F3: queue_wait_median
    max_wait = thresholds.get("max_queue_wait_median_sec", 60)
    checks["F3_queue_wait_median"] = {
        "value": metrics.queue_wait_median_sec,
        "threshold": max_wait,
        "pass": metrics.queue_wait_median_sec <= max_wait,
    }

    # F4: PnL — 有意に負でないこと (原初 E4 維持)
    pnl_alpha = thresholds.get("pnl_alpha", 0.05)
    f4_pass: bool
    if metrics.post_fill_30s_pnl_mean >= 0:
        f4_pass = True
    elif metrics.post_fill_30s_pnl_pvalue >= pnl_alpha:
        f4_pass = True  # 負だが統計的に有意でない
    else:
        f4_pass = False
    checks["F4_pnl"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "ci_upper": metrics.post_fill_30s_pnl_ci_upper,
        "alpha": pnl_alpha,
        "pass": f4_pass,
    }

    # F5: AS_ratio (115# Q10.2(B): 35→30)
    max_as = thresholds.get("max_adverse_selection_ratio", 0.30)
    checks["F5_adverse_selection"] = {
        "value": metrics.adverse_selection_ratio,
        "threshold": max_as,
        "pass": metrics.adverse_selection_ratio <= max_as,
    }

    # F6: skip_gate_ratio
    max_skip = thresholds.get("max_skip_gate_ratio", 0.20)
    checks["F6_skip_gate_ratio"] = {
        "value": metrics.skip_gate_ratio,
        "threshold": max_skip,
        "pass": metrics.skip_gate_ratio <= max_skip,
    }

    # F7: calendar_days
    min_days = thresholds.get("min_calendar_days", 7)
    checks["F7_calendar_days"] = {
        "value": metrics.measurement_days,
        "threshold": min_days,
        "pass": metrics.measurement_days >= min_days,
    }

    # F8: n_attempted
    min_n = thresholds.get("min_attempted_samples", 500)
    checks["F8_n_attempted"] = {
        "value": metrics.attempted_orders,
        "threshold": min_n,
        "pass": metrics.attempted_orders >= min_n,
    }

    all_pass = all(c["pass"] for c in checks.values())

    return {
        "gate": "G1.2-full",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "metrics_summary": metrics.to_dict(),
    }


# ======================================================================
# I/O utilities
# ======================================================================


def save_fill_records(records: list[FillRecord], path: str | Path) -> None:
    """JSONL 形式で FillRecord を保存.

    032#17: バッチ全体を tempfile に書き出してから append する。
    SIGINT / ディスクフル時の不完全行混入を防止。
    """
    import os
    import tempfile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records]
    # Atomic batch: write to temp, fsync, then append to target
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.writelines(lines)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        with open(p, "a", encoding="utf-8") as f:
            with open(tmp_path, "r", encoding="utf-8") as tmp_r:
                f.write(tmp_r.read())
            f.flush()
            os.fsync(f.fileno())
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    logger.info(f"Saved {len(records)} fill records to {p}")


def load_fill_records(path: str | Path) -> list[FillRecord]:
    """JSONL ファイルから FillRecord を読み込み.

    032# #19: 破損行はスキップしてログ出力。
    101# §5: cycle_id による重複排除 (SIGINT 中断時の partial+emergency 重複対策)。
    """
    p = Path(path)
    if not p.exists():
        return []
    records: list[FillRecord] = []
    seen_ids: set[str] = set()
    skipped = 0
    duplicates = 0
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = FillRecord.from_dict(json.loads(line))
                if rec.cycle_id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(rec.cycle_id)
                records.append(rec)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                skipped += 1
                logger.warning(
                    f"Skipped corrupt line {line_no} in {p.name}: {e}"
                )
    if skipped:
        logger.warning(f"Total {skipped} corrupt lines skipped in {p.name}")
    if duplicates:
        logger.info(f"Deduplicated {duplicates} records in {p.name}")
    logger.info(f"Loaded {len(records)} fill records from {p}")
    return records


def load_fill_records_glob(directory: str | Path) -> list[FillRecord]:
    """ディレクトリ内の全 JSONL ファイルから FillRecord を読み込み.

    101# §5: cross-file 重複排除 (emergency dump との重複対策)."""
    d = Path(directory)
    records: list[FillRecord] = []
    seen_ids: set[str] = set()
    for p in sorted(d.glob("fill_records_*.jsonl")):
        file_records = load_fill_records(p)
        for r in file_records:
            if r.cycle_id not in seen_ids:
                seen_ids.add(r.cycle_id)
                records.append(r)
    # emergency dump ディレクトリも統合 (重複は自動排除)
    emergency_dir = d / "emergency"
    if emergency_dir.exists():
        for p in sorted(emergency_dir.glob("emergency_*.jsonl")):
            file_records = load_fill_records(p)
            for r in file_records:
                if r.cycle_id not in seen_ids:
                    seen_ids.add(r.cycle_id)
                    records.append(r)
    return records


def filter_clean_records(
    records: list[FillRecord],
    *,
    require_git_sha: bool = True,
) -> tuple[list[FillRecord], list[FillRecord]]:
    """046# clean/quarantine 分離 + 047# A5 拡張基準.

    以下のいずれかに該当するレコードを quarantine へ分類:
    - git_sha が blank/None (ゾンビプロセス由来)
    - run_id が blank/None (020# O4 以前の旧形式レコード)
    - 必須フィールド (side, order_price, order_quantity) が不正

    Args:
        records: 全 FillRecord リスト.
        require_git_sha: True の場合 git_sha が blank/None のレコードを quarantine.
            False の場合は全チェックをバイパスして全件 clean を返す.

    Returns:
        (clean, quarantine) のタプル.
    """
    if not require_git_sha:
        return records, []  # テスト用: 全件 clean (本番は常に True)

    clean: list[FillRecord] = []
    quarantine: list[FillRecord] = []
    for r in records:
        # 047# A5: 複合チェック — git_sha + run_id + 必須フィールド
        reason = _quarantine_reason(r)
        if reason:
            quarantine.append(r)
        else:
            clean.append(r)

    if quarantine:
        logger.info(
            f"[quarantine] {len(quarantine)}/{len(records)} records quarantined. "
            f"clean={len(clean)}"
        )
    return clean, quarantine


def _quarantine_reason(r: FillRecord) -> str | None:
    """047# A5: レコードの quarantine 理由を返す (None=clean)."""
    if not (r.git_sha and r.git_sha.strip()):
        return "blank_git_sha"
    if not (r.run_id and r.run_id.strip()):
        return "blank_run_id"
    if r.side not in ("buy", "sell"):
        return f"invalid_side={r.side}"
    if not r.order_price or r.order_price <= 0:
        return "invalid_order_price"
    if not r.order_quantity or r.order_quantity <= 0:
        return "invalid_order_quantity"
    return None


# ======================================================================
# 051# P2-2: Round-trip 評価 (buy→sell ペアリング)
# ======================================================================


@dataclass
class RoundTripRecord:
    """往復取引記録 (buy→sell / sell→buy 双方向対応).

    055# Fix: sell先行ペアも対称的に評価.
    """

    entry_record: FillRecord
    exit_record: FillRecord
    pnl_bps: float  # エントリー基準の損益 (bps)
    pnl_jpy: float  # 実損益 (JPY)
    hold_sec: float  # 保持時間 (秒)
    direction: str  # "buy_first" or "sell_first"

    # 後方互換: buy_record/sell_record プロパティ
    @property
    def buy_record(self) -> FillRecord:
        return self.entry_record if self.direction == "buy_first" else self.exit_record

    @property
    def sell_record(self) -> FillRecord:
        return self.exit_record if self.direction == "buy_first" else self.entry_record


@dataclass
class RoundTripMetrics:
    """Round-trip 集計指標.

    055# Fix: unpaired_sells / net_inventory 追加.
    """

    total_pairs: int = 0
    pnl_mean_bps: float = 0.0
    pnl_median_bps: float = 0.0
    pnl_std_bps: float = 0.0
    pnl_total_jpy: float = 0.0
    win_rate: float = 0.0  # pnl > 0 の割合
    hold_sec_median: float = 0.0
    unpaired_buys: int = 0  # ペアリング未完了の buy 件数
    unpaired_sells: int = 0  # 055# ペアリング未完了の sell 件数
    net_inventory: int = 0  # 055# 純在庫 (unpaired_buys - unpaired_sells)


def compute_round_trip_metrics(
    records: list[FillRecord],
) -> tuple[RoundTripMetrics, list[RoundTripRecord]]:
    """051# P2-2 → 055# Fix: 双方向 FIFO ペアリングで往復損益を算出.

    inventory-aware 方式:
    - net inventory を追跡し、buy/sell どちらが先でもペアリング.
    - buy 先行時: sell が来たら close. PnL = (sell - buy) / buy
    - sell 先行時: buy が来たら close. PnL = (sell - buy) / buy (空売り利益)

    Args:
        records: FillRecord リスト (時系列ソート済み想定).

    Returns:
        (RoundTripMetrics, list[RoundTripRecord]) タプル.
    """
    filled = [r for r in records if r.filled and r.fill_price is not None]
    filled.sort(key=lambda r: r.timestamp)

    pending_buys: list[FillRecord] = []
    pending_sells: list[FillRecord] = []
    trips: list[RoundTripRecord] = []

    for r in filled:
        if r.side == "buy":
            if pending_sells:
                # sell 先行 → buy で close
                sell_entry = pending_sells.pop(0)  # FIFO
                pnl_bps = (sell_entry.fill_price - r.fill_price) / r.fill_price * 10_000  # type: ignore[operator]
                qty = min(r.order_quantity, sell_entry.order_quantity)
                pnl_jpy = (sell_entry.fill_price - r.fill_price) * qty  # type: ignore[operator]
                hold_sec = r.timestamp - sell_entry.timestamp
                trips.append(RoundTripRecord(
                    entry_record=sell_entry,
                    exit_record=r,
                    pnl_bps=pnl_bps,
                    pnl_jpy=pnl_jpy,
                    hold_sec=hold_sec,
                    direction="sell_first",
                ))
            else:
                pending_buys.append(r)
        elif r.side == "sell":
            if pending_buys:
                # buy 先行 → sell で close
                buy_entry = pending_buys.pop(0)  # FIFO
                pnl_bps = (r.fill_price - buy_entry.fill_price) / buy_entry.fill_price * 10_000  # type: ignore[operator]
                qty = min(r.order_quantity, buy_entry.order_quantity)
                pnl_jpy = (r.fill_price - buy_entry.fill_price) * qty  # type: ignore[operator]
                hold_sec = r.timestamp - buy_entry.timestamp
                trips.append(RoundTripRecord(
                    entry_record=buy_entry,
                    exit_record=r,
                    pnl_bps=pnl_bps,
                    pnl_jpy=pnl_jpy,
                    hold_sec=hold_sec,
                    direction="buy_first",
                ))
            else:
                pending_sells.append(r)

    if not trips:
        return RoundTripMetrics(
            unpaired_buys=len(pending_buys),
            unpaired_sells=len(pending_sells),
            net_inventory=len(pending_buys) - len(pending_sells),
        ), []

    pnl_arr = [t.pnl_bps for t in trips]
    hold_arr = [t.hold_sec for t in trips]

    return RoundTripMetrics(
        total_pairs=len(trips),
        pnl_mean_bps=float(np.mean(pnl_arr)),
        pnl_median_bps=float(np.median(pnl_arr)),
        pnl_std_bps=float(np.std(pnl_arr)),
        pnl_total_jpy=sum(t.pnl_jpy for t in trips),
        win_rate=sum(1 for p in pnl_arr if p > 0) / len(pnl_arr),
        hold_sec_median=float(np.median(hold_arr)),
        unpaired_buys=len(pending_buys),
        unpaired_sells=len(pending_sells),
        net_inventory=len(pending_buys) - len(pending_sells),
    ), trips


# ======================================================================
# 051# P2-4: レジーム別メトリクス
# ======================================================================


@dataclass
class RegimeMetrics:
    """レジーム別の集計指標."""

    regime: str
    count: int = 0
    filled: int = 0
    fill_rate: float = 0.0
    pnl_mean_bps: float = 0.0
    as_ratio: float = 0.0
    queue_wait_median_sec: float = 0.0


def compute_regime_metrics(records: list[FillRecord]) -> list[RegimeMetrics]:
    """051# P2-4: レジーム別にメトリクスを算出.

    Args:
        records: FillRecord リスト.

    Returns:
        RegimeMetrics のリスト (レジーム名でソート).
    """
    from collections import defaultdict

    groups: dict[str, list[FillRecord]] = defaultdict(list)
    for r in records:
        regime = r.regime or "unknown"
        groups[regime].append(r)

    result: list[RegimeMetrics] = []
    for regime_name in sorted(groups.keys()):
        recs = groups[regime_name]
        filled_recs = [r for r in recs if r.filled]
        pnls = [
            r.post_fill_30s_pnl for r in filled_recs
            if r.post_fill_30s_pnl is not None
        ]
        as_recs = [r for r in filled_recs if r.adverse_selected is not None]
        waits = [r.queue_wait_sec for r in filled_recs if r.queue_wait_sec > 0]

        result.append(RegimeMetrics(
            regime=regime_name,
            count=len(recs),
            filled=len(filled_recs),
            fill_rate=len(filled_recs) / len(recs) if recs else 0.0,
            pnl_mean_bps=float(np.mean(pnls)) if pnls else 0.0,
            as_ratio=(
                sum(1 for r in as_recs if r.adverse_selected) / len(as_recs)
                if as_recs else 0.0
            ),
            queue_wait_median_sec=float(np.median(waits)) if waits else 0.0,
        ))
    return result


# ======================================================================
# 051# UTC 時間帯別分析
# ======================================================================


@dataclass
class HourlyMetrics:
    """UTC 時間帯別の集計指標."""

    utc_hour: int
    count: int = 0
    filled: int = 0
    pnl_mean_bps: float = 0.0
    as_ratio: float = 0.0


def compute_hourly_metrics(records: list[FillRecord]) -> list[HourlyMetrics]:
    """051# UTC 時間帯別にメトリクスを算出.

    time_filter 検証用: 各 UTC hour の AS/PnL を可視化.

    Args:
        records: FillRecord リスト.

    Returns:
        HourlyMetrics のリスト (utc_hour 昇順).
    """
    from collections import defaultdict

    groups: dict[int, list[FillRecord]] = defaultdict(list)
    for r in records:
        utc_hour = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).hour
        groups[utc_hour].append(r)

    result: list[HourlyMetrics] = []
    for hour in range(24):
        if hour not in groups:
            continue
        recs = groups[hour]
        filled_recs = [r for r in recs if r.filled]
        pnls = [
            r.post_fill_30s_pnl for r in filled_recs
            if r.post_fill_30s_pnl is not None
        ]
        as_recs = [r for r in filled_recs if r.adverse_selected is not None]

        result.append(HourlyMetrics(
            utc_hour=hour,
            count=len(recs),
            filled=len(filled_recs),
            pnl_mean_bps=float(np.mean(pnls)) if pnls else 0.0,
            as_ratio=(
                sum(1 for r in as_recs if r.adverse_selected) / len(as_recs)
                if as_recs else 0.0
            ),
        ))
    return result
