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
from datetime import datetime
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

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FillRecord:
        """Reconstruct from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
        day_key = datetime.utcfromtimestamp(r.timestamp).strftime("%Y-%m-%d")
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
    sample_sufficient = total >= 200 and len(daily_fill_rates) >= 3

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
    )


# ======================================================================
# G1.1 Gate Judgment
# ======================================================================


def g1_1_judgment(
    metrics: FillMetrics,
    thresholds: dict,
) -> dict:
    """G1.1 Gate 合否判定.

    009# §2.1 / 000# §3.3 準拠.

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_1_exec`` セクション.

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

    # Gate 判定には informational=True のチェックは含めない
    gate_checks = {k: v for k, v in checks.items() if not v.get("informational")}
    all_pass = all(c["pass"] for c in gate_checks.values())

    # 020# O1: サンプル要件不足の場合は暫定判定
    judgment_type = "FINAL" if metrics.sample_sufficient else "PROVISIONAL"

    return {
        "gate": "G1.1-exec",
        "gate_result": "PASS" if all_pass else "FAIL",
        "judgment_type": judgment_type,  # 020# O1
        "sample_sufficient": metrics.sample_sufficient,
        "checks": checks,
        "metrics_summary": metrics.to_dict(),
    }


# ======================================================================
# I/O utilities
# ======================================================================


def save_fill_records(records: list[FillRecord], path: str | Path) -> None:
    """JSONL 形式で FillRecord を保存."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} fill records to {p}")


def load_fill_records(path: str | Path) -> list[FillRecord]:
    """JSONL ファイルから FillRecord を読み込み."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[FillRecord] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(FillRecord.from_dict(json.loads(line)))
    logger.info(f"Loaded {len(records)} fill records from {p}")
    return records


def load_fill_records_glob(directory: str | Path) -> list[FillRecord]:
    """ディレクトリ内の全 JSONL ファイルから FillRecord を読み込み."""
    d = Path(directory)
    records: list[FillRecord] = []
    for p in sorted(d.glob("fill_records_*.jsonl")):
        records.extend(load_fill_records(p))
    return records
