"""672# 多角的深堀り分析 — 金融工学・情報理論・統計学・市場理論.

671# の 10 視座分析をさらに掘り下げ、データに嘘をつかせない厳密な検証。

使用理論:
  - 金融工学: Glosten-Milgrom AS推定、Realized spread 分解、最適クオート閾値
  - 情報理論: 相互情報量 (MI)、条件付きエントロピー、KL divergence
  - 統計学: Bootstrap CI、Permutation test、効果量 (Cohen's d)
  - 市場マイクロストラクチャ: Maker's edge decomposition、Fill probability model
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

# --- プロジェクトルートを path に追加 ---
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.v460.analysis.analysis_common import (
    DEFAULT_RESULTS_DIR,
    load_and_filter_records,
)
from ztb.utils.safety import safe_to_finite

# ======================================================================
# 定数
# ======================================================================
AS_THRESHOLD_BPS: Final[float] = -3.0
N_BOOTSTRAP: Final[int] = 10_000
SEED: Final[int] = 42

rng = np.random.default_rng(SEED)


# ======================================================================
# ユーティリティ
# ======================================================================

def _safe_float(record: dict, key: str, default: float = 0.0) -> float:
    v = record.get(key)
    if v is None:
        return default
    f = safe_to_finite(v)
    return float(f) if f is not None else default


def _pnl_30s(r: dict) -> float | None:
    v = r.get("post_fill_30s_pnl")
    if v is None:
        return None
    f = safe_to_finite(v)
    return float(f) if f is not None else None


def _pnl_120s(r: dict) -> float | None:
    v = r.get("post_fill_120s_pnl")
    if v is None:
        return None
    f = safe_to_finite(v)
    return float(f) if f is not None else None


def _pnl_ev(r: dict) -> float | None:
    v = r.get("ev_weighted_pnl")
    if v is None:
        return None
    f = safe_to_finite(v)
    return float(f) if f is not None else None


def _is_filled(r: dict) -> bool:
    return bool(r.get("filled", False))


def _is_as(r: dict) -> bool:
    """Adverse Selection 判定."""
    pnl = _pnl_30s(r)
    return pnl is not None and pnl < AS_THRESHOLD_BPS


# ======================================================================
# §1 Bootstrap 信頼区間
# ======================================================================

def bootstrap_ci(
    data: NDArray[np.float64],
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap による平均値の信頼区間.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, alpha * 100))
    hi = float(np.percentile(means, (1 - alpha) * 100))
    return (float(np.mean(data)), lo, hi)


def permutation_test(
    group_a: NDArray[np.float64],
    group_b: NDArray[np.float64],
    n_perm: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Permutation test for difference in means.

    Returns:
        (observed_diff, p_value)
    """
    obs_diff = float(np.mean(group_a) - np.mean(group_b))
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(perm_diff) >= abs(obs_diff):
            count += 1
    return obs_diff, count / n_perm


def cohens_d(
    group_a: NDArray[np.float64],
    group_b: NDArray[np.float64],
) -> float:
    """Cohen's d 効果量."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


# ======================================================================
# §2 情報理論: 相互情報量 (離散化ベース)
# ======================================================================

def _discretize(arr: NDArray[np.float64], n_bins: int = 10) -> NDArray[np.int64]:
    """等頻度ビン分割で離散化."""
    if len(arr) == 0:
        return np.array([], dtype=np.int64)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(arr, percentiles)
    # 重複エッジ対応
    edges = np.unique(edges)
    return np.digitize(arr, edges[1:-1]).astype(np.int64)


def mutual_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """離散化ベースの相互情報量 I(X;Y) (nats).

    相関が捉えない非線形依存もキャプチャする。
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0
    dx = _discretize(x, n_bins)
    dy = _discretize(y, n_bins)

    # Joint & marginal
    n = len(x)
    joint: dict[tuple[int, int], int] = {}
    mx: dict[int, int] = {}
    my: dict[int, int] = {}
    for i in range(n):
        xi, yi = int(dx[i]), int(dy[i])
        joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
        mx[xi] = mx.get(xi, 0) + 1
        my[yi] = my.get(yi, 0) + 1

    mi = 0.0
    for (xi, yi), c in joint.items():
        p_xy = c / n
        p_x = mx[xi] / n
        p_y = my[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    return float(mi)


def conditional_entropy(
    target: NDArray[np.float64],
    condition: NDArray[np.int64],
) -> float:
    """H(Target | Condition) — 条件付きエントロピー (nats)."""
    if len(target) == 0:
        return 0.0
    dt = _discretize(target, 10)
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(condition):
        groups.setdefault(int(c), []).append(int(dt[i]))
    n = len(target)
    h_cond = 0.0
    for c_val, t_vals in groups.items():
        p_c = len(t_vals) / n
        # H(T | C=c)
        counts: dict[int, int] = {}
        for tv in t_vals:
            counts[tv] = counts.get(tv, 0) + 1
        h_t_c = 0.0
        nc = len(t_vals)
        for cnt in counts.values():
            p = cnt / nc
            if p > 0:
                h_t_c -= p * np.log(p)
        h_cond += p_c * h_t_c
    return float(h_cond)


def entropy(arr: NDArray[np.float64], n_bins: int = 10) -> float:
    """H(X) — エントロピー (nats)."""
    if len(arr) == 0:
        return 0.0
    d = _discretize(arr, n_bins)
    n = len(d)
    counts: dict[int, int] = {}
    for v in d:
        counts[int(v)] = counts.get(int(v), 0) + 1
    h = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            h -= p * np.log(p)
    return float(h)


def kl_divergence(
    p_arr: NDArray[np.float64],
    q_arr: NDArray[np.float64],
    n_bins: int = 15,
) -> float:
    """KL(P || Q) — KL ダイバージェンス (nats).

    両分布を共通ビンで離散化し、Laplace smoothing 適用。
    """
    if len(p_arr) == 0 or len(q_arr) == 0:
        return 0.0
    all_data = np.concatenate([p_arr, q_arr])
    edges = np.percentile(all_data, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    bins_p = np.digitize(p_arr, edges[1:-1])
    bins_q = np.digitize(q_arr, edges[1:-1])
    n_actual_bins = len(edges) - 1

    # Laplace smoothed
    hist_p = np.ones(n_actual_bins + 1)  # +1 for overflow bin
    hist_q = np.ones(n_actual_bins + 1)
    for b in bins_p:
        hist_p[b] += 1
    for b in bins_q:
        hist_q[b] += 1
    p_dist = hist_p / hist_p.sum()
    q_dist = hist_q / hist_q.sum()

    kl = 0.0
    for i in range(len(p_dist)):
        if p_dist[i] > 0 and q_dist[i] > 0:
            kl += p_dist[i] * np.log(p_dist[i] / q_dist[i])
    return float(kl)


# ======================================================================
# §3 Glosten-Milgrom 推定
# ======================================================================

def estimate_adverse_selection_alpha(
    fills: list[dict],
) -> dict[str, float]:
    """Glosten-Milgrom モデルに基づく AS 確率推定.

    α = P(informed) を、mid price 変動の非対称性から推定。
    Informed trader のオーダーは mid を不利方向に動かし、
    uninformed は mid を動かさない（ランダム）。

    Returns:
        {"alpha": float, "informed_loss_avg": float,
         "uninformed_gain_avg": float, "implied_min_spread": float}
    """
    adverse_pnls: list[float] = []
    non_adverse_pnls: list[float] = []

    for r in fills:
        pnl = _pnl_30s(r)
        if pnl is None:
            continue
        if pnl < AS_THRESHOLD_BPS:
            adverse_pnls.append(pnl)
        else:
            non_adverse_pnls.append(pnl)

    n_total = len(adverse_pnls) + len(non_adverse_pnls)
    if n_total == 0:
        return {"alpha": 0.0, "informed_loss_avg": 0.0,
                "uninformed_gain_avg": 0.0, "implied_min_spread": 0.0}

    alpha = len(adverse_pnls) / n_total
    informed_loss = float(np.mean(adverse_pnls)) if adverse_pnls else 0.0
    uninformed_gain = float(np.mean(non_adverse_pnls)) if non_adverse_pnls else 0.0

    # Glosten-Milgrom: break-even half-spread = α × E[|V-P| | informed]
    # Simplified: implied_min_spread = α × |informed_loss| / (1 - α)
    implied = 0.0
    if alpha < 1.0 and informed_loss < 0:
        implied = alpha * abs(informed_loss) / (1 - alpha)

    return {
        "alpha": alpha,
        "informed_loss_avg": informed_loss,
        "uninformed_gain_avg": uninformed_gain,
        "implied_min_spread": implied,
    }


# ======================================================================
# §4 Realized Spread 分解
# ======================================================================

def realized_spread_decomposition(
    fills: list[dict],
) -> dict[str, float]:
    """スプレッド分解: quoted spread = realized spread + adverse selection cost.

    Mark-to-market at 30s:
      - Realized spread (maker revenue) = 2 × (fill_price - mid_30s) × side_sign
      - AS cost = quoted_half_spread - realized_half_spread

    全て bps 単位。
    """
    realized: list[float] = []
    quoted: list[float] = []

    for r in fills:
        spread_bps = _safe_float(r, "spread_bps")
        pnl_30 = _pnl_30s(r)
        if spread_bps <= 0 or pnl_30 is None:
            continue

        half_spread = spread_bps / 2
        # Realized = half_spread + pnl_30s
        # (pnl = fill_gain - mid_movement, so realized = half_spread + pnl)
        realized_hs = half_spread + pnl_30
        quoted.append(half_spread)
        realized.append(realized_hs)

    if not realized:
        return {"realized_hs_avg": 0.0, "quoted_hs_avg": 0.0,
                "as_cost_avg": 0.0, "as_cost_pct": 0.0}

    r_avg = float(np.mean(realized))
    q_avg = float(np.mean(quoted))
    as_cost = q_avg - r_avg

    return {
        "realized_hs_avg": r_avg,
        "quoted_hs_avg": q_avg,
        "as_cost_avg": as_cost,
        "as_cost_pct": (as_cost / q_avg * 100) if q_avg > 0 else 0.0,
    }


# ======================================================================
# §5 最適クオート閾値シミュレーション
# ======================================================================

def optimal_min_spread_simulation(
    all_records: list[dict],
) -> list[dict[str, float]]:
    """min_spread 閾値を変えた場合の期待収益をシミュレーション.

    全レコード (filled + NFQ) を使い、各 min_spread 閾値で:
      - fill 可能なレコード = NFQ のうち actual_spread >= 閾値のもの + filled
      - 期待 PnL をスプレッド帯域のhistorical PnLで近似

    Returns:
        各閾値での {"threshold_jpy", "fill_rate", "expected_avg30",
                    "expected_sum30", "n_fills"}
    """
    # まず filled records からスプレッド→PnL のマッピングを作成
    filled = [r for r in all_records if _is_filled(r)]
    nfq = [r for r in all_records if not _is_filled(r)
           and str(r.get("cancel_reason", "")).startswith("no_feasible")]

    # スプレッド→PnL テーブル (filled から)
    spread_pnl: list[tuple[float, float]] = []
    for r in filled:
        spread_jpy = _safe_float(r, "spread_at_order")
        pnl = _pnl_30s(r)
        if spread_jpy > 0 and pnl is not None:
            spread_pnl.append((spread_jpy, pnl))

    if not spread_pnl:
        return []

    # NFQ レコードのactual_spread
    nfq_spreads: list[float] = []
    for r in nfq:
        s = _safe_float(r, "nfq_actual_spread")
        if s <= 0:
            # Legacy: error_message からパース試行
            msg = str(r.get("error_message", ""))
            if "actual=" in msg:
                try:
                    s = float(msg.split("actual=")[1].split(",")[0].split(")")[0])
                except (ValueError, IndexError):
                    pass
        if s > 0:
            nfq_spreads.append(s)

    # スプレッド帯域ごとの平均PnL（200 JPY刻み）
    all_spreads = np.array([s for s, _ in spread_pnl])
    all_pnls = np.array([p for _, p in spread_pnl])

    bin_size = 200.0
    bins: dict[int, list[float]] = {}
    for s, p in spread_pnl:
        b = int(s // bin_size)
        bins.setdefault(b, []).append(p)

    def pnl_for_spread(s: float) -> float:
        b = int(s // bin_size)
        # 同ビン → 隣接ビン → 全体平均 のフォールバック
        for offset in [0, -1, 1, -2, 2]:
            if (b + offset) in bins:
                return float(np.mean(bins[b + offset]))
        return float(np.mean(all_pnls))

    # 閾値スイープ
    thresholds = list(range(500, 5001, 250))
    results = []
    total = len(filled) + len(nfq)

    for thresh in thresholds:
        # 現 filled のうち閾値以上のもの
        passed_filled = [(s, p) for s, p in spread_pnl if s >= thresh]
        # NFQ のうち閾値以下になれば通過するもの
        recovered_nfq = [s for s in nfq_spreads if s >= thresh]
        recovered_pnls = [pnl_for_spread(s) for s in recovered_nfq]

        all_passed_pnls = [p for _, p in passed_filled] + recovered_pnls
        n_fills = len(all_passed_pnls)
        fill_rate = n_fills / total if total > 0 else 0

        avg30 = float(np.mean(all_passed_pnls)) if all_passed_pnls else 0.0
        sum30 = float(np.sum(all_passed_pnls)) if all_passed_pnls else 0.0

        results.append({
            "threshold_jpy": float(thresh),
            "fill_rate": fill_rate,
            "expected_avg30": avg30,
            "expected_sum30": sum30,
            "n_fills": float(n_fills),
        })

    return results


# ======================================================================
# §6 Fill Probability Model
# ======================================================================

def fill_probability_by_offset(
    all_records: list[dict],
    n_bins: int = 8,
) -> list[dict[str, float]]:
    """Offset 帯域ごとの fill probability と avg PnL.

    マーケットメイカーの基本トレードオフ:
    offset ↑ → fill probability ↓, PnL per fill ↑（理論上）
    """
    data: list[tuple[float, bool, float]] = []
    for r in all_records:
        offset = _safe_float(r, "effective_offset_used")
        filled = _is_filled(r)
        pnl = _pnl_30s(r) if filled else None
        data.append((offset, filled, pnl if pnl is not None else 0.0))

    if not data:
        return []

    offsets = np.array([d[0] for d in data])
    edges = np.percentile(offsets, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)

    results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (offsets >= lo) & (offsets <= hi)
        else:
            mask = (offsets >= lo) & (offsets < hi)
        subset = [data[j] for j in range(len(data)) if mask[j]]
        if not subset:
            continue
        n_total = len(subset)
        n_filled = sum(1 for _, f, _ in subset if f)
        fill_prob = n_filled / n_total if n_total > 0 else 0
        pnls = [p for _, f, p in subset if f]
        avg_pnl = float(np.mean(pnls)) if pnls else 0.0
        # E[revenue] = fill_prob × avg_pnl
        expected_rev = fill_prob * avg_pnl
        results.append({
            "offset_lo": float(lo),
            "offset_hi": float(hi),
            "n": float(n_total),
            "fill_prob": fill_prob,
            "avg_pnl30": avg_pnl,
            "expected_revenue": expected_rev,
        })

    return results


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:  # noqa: C901
    records = load_and_filter_records(
        DEFAULT_RESULTS_DIR,
        date_from="2026-03-20",
        exit_on_empty=True,
    )
    filled = [r for r in records if _is_filled(r)]
    nfq = [r for r in records if not _is_filled(r)
           and str(r.get("cancel_reason", "")).startswith("no_feasible")]

    print("=" * 72)
    print("672# 多角的深堀り分析")
    print(f"  Total records: {len(records)}, Filled: {len(filled)}, NFQ: {len(nfq)}")
    print("=" * 72)

    # ---- §1 Bootstrap CI ----
    print("\n" + "─" * 72)
    print("§1 Bootstrap 信頼区間 (95% CI, n_boot=10000)")
    print("─" * 72)

    pnl30_all = np.array([_pnl_30s(r) for r in filled if _pnl_30s(r) is not None])
    pnl120_all = np.array([_pnl_120s(r) for r in filled if _pnl_120s(r) is not None])

    pnl30_as = np.array([_pnl_30s(r) for r in filled
                         if _pnl_30s(r) is not None and _is_as(r)])
    pnl30_non_as = np.array([_pnl_30s(r) for r in filled
                             if _pnl_30s(r) is not None and not _is_as(r)])

    for label, data in [
        ("全 fill PnL30", pnl30_all),
        ("全 fill PnL120", pnl120_all),
        ("AS fill PnL30", pnl30_as),
        ("Non-AS fill PnL30", pnl30_non_as),
    ]:
        mean, lo, hi = bootstrap_ci(data)
        contains_zero = "✓" if lo <= 0 <= hi else "✗"
        print(f"  {label:20s}: mean={mean:+.3f} bps  "
              f"CI=[{lo:+.3f}, {hi:+.3f}]  "
              f"n={len(data)}  0∈CI: {contains_zero}")

    # ---- §1b Permutation test: AS vs Non-AS ----
    print("\n  Permutation test (AS vs Non-AS, PnL30):")
    if len(pnl30_as) > 0 and len(pnl30_non_as) > 0:
        diff, p_val = permutation_test(pnl30_as, pnl30_non_as)
        d = cohens_d(pnl30_as, pnl30_non_as)
        print(f"    diff(AS - NonAS) = {diff:+.3f} bps")
        print(f"    p-value = {p_val:.4f}")
        print(f"    Cohen's d = {d:.3f} "
              f"({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")

    # ---- §2 情報理論 ----
    print("\n" + "─" * 72)
    print("§2 情報理論分析")
    print("─" * 72)

    # H(PnL30)
    h_pnl = entropy(pnl30_all)
    print(f"\n  H(PnL30) = {h_pnl:.4f} nats")

    # H(PnL30 | AS)
    as_labels = np.array([1 if _is_as(r) else 0 for r in filled
                          if _pnl_30s(r) is not None], dtype=np.int64)
    h_cond = conditional_entropy(pnl30_all, as_labels)
    print(f"  H(PnL30 | AS) = {h_cond:.4f} nats")
    print(f"  I(PnL30 ; AS) = {h_pnl - h_cond:.4f} nats  "
          f"(AS が解消する不確実性: {(h_pnl - h_cond) / h_pnl * 100:.1f}%)")

    # MI: 各 feature vs PnL30
    print("\n  相互情報量 I(Feature ; PnL30):")
    features_to_test = [
        ("skip_gate_score", "SkipGate score"),
        ("effective_offset_used", "effective_offset"),
        ("spread_bps", "spread_bps"),
        ("cross_venue_lead_lag_spread_bps", "cv_lead_lag"),
        ("vg_vpin", "vg_vpin"),
        ("orderbook_imbalance", "OB imbalance"),
        ("regime_confidence", "regime_confidence"),
    ]

    pnl_for_mi = []
    feature_arrays: dict[str, list[float]] = {f: [] for f, _ in features_to_test}

    for r in filled:
        pnl = _pnl_30s(r)
        if pnl is None:
            continue
        all_valid = True
        vals: dict[str, float] = {}
        for fkey, _ in features_to_test:
            v = _safe_float(r, fkey, default=float("nan"))
            if np.isnan(v):
                all_valid = False
                break
            vals[fkey] = v
        if not all_valid:
            continue
        pnl_for_mi.append(pnl)
        for fkey, _ in features_to_test:
            feature_arrays[fkey].append(vals[fkey])

    pnl_mi_arr = np.array(pnl_for_mi)
    for fkey, flabel in features_to_test:
        f_arr = np.array(feature_arrays[fkey])
        if len(f_arr) == 0:
            print(f"    {flabel:25s}: (no data)")
            continue
        mi = mutual_information(f_arr, pnl_mi_arr)
        corr = float(np.corrcoef(f_arr, pnl_mi_arr)[0, 1]) if len(f_arr) > 1 else 0
        print(f"    {flabel:25s}: MI={mi:.4f} nats  ρ={corr:+.4f}  "
              f"(MI/H(PnL)={mi / h_pnl * 100:.1f}%)")

    # KL divergence: AS vs Non-AS の PnL 分布
    print("\n  KL divergence:")
    if len(pnl30_as) > 0 and len(pnl30_non_as) > 0:
        kl_as_nonas = kl_divergence(pnl30_as, pnl30_non_as)
        kl_nonas_as = kl_divergence(pnl30_non_as, pnl30_as)
        print(f"    KL(AS || NonAS) = {kl_as_nonas:.4f} nats")
        print(f"    KL(NonAS || AS) = {kl_nonas_as:.4f} nats")
        print(f"    → 非対称性: AS 分布は NonAS とかなり"
              f"{'異なる' if kl_as_nonas > 0.5 else '近い'}")

    # ---- §3 Glosten-Milgrom ----
    print("\n" + "─" * 72)
    print("§3 Glosten-Milgrom AS 推定")
    print("─" * 72)

    gm = estimate_adverse_selection_alpha(filled)
    print(f"  α (informed rate)   = {gm['alpha']:.3f} ({gm['alpha'] * 100:.1f}%)")
    print(f"  E[loss|informed]    = {gm['informed_loss_avg']:+.2f} bps")
    print(f"  E[gain|uninformed]  = {gm['uninformed_gain_avg']:+.2f} bps")
    print(f"  Implied min HS      = {gm['implied_min_spread']:.2f} bps")
    print(f"  → Break-even条件: half-spread > {gm['implied_min_spread']:.2f} bps "
          f"が必要")

    # 帯域別 GM
    print("\n  スプレッド帯域別 α:")
    spread_bands = [(0, 1500), (1500, 2500), (2500, 3500), (3500, 99999)]
    for lo, hi in spread_bands:
        band_fills = [r for r in filled
                      if lo <= _safe_float(r, "spread_at_order") < hi]
        if not band_fills:
            continue
        gm_band = estimate_adverse_selection_alpha(band_fills)
        label = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
        print(f"    {label:>10s} JPY: α={gm_band['alpha']:.3f}  "
              f"loss={gm_band['informed_loss_avg']:+.1f}  "
              f"gain={gm_band['uninformed_gain_avg']:+.2f}  "
              f"impl_hs={gm_band['implied_min_spread']:.2f} bps  "
              f"n={len(band_fills)}")

    # ---- §4 Realized Spread 分解 ----
    print("\n" + "─" * 72)
    print("§4 Realized Spread 分解")
    print("─" * 72)

    rsd = realized_spread_decomposition(filled)
    print(f"  Quoted half-spread avg   = {rsd['quoted_hs_avg']:.2f} bps")
    print(f"  Realized half-spread avg = {rsd['realized_hs_avg']:.2f} bps")
    print(f"  AS cost avg              = {rsd['as_cost_avg']:.2f} bps "
          f"({rsd['as_cost_pct']:.1f}% of quoted)")
    margin = rsd["realized_hs_avg"]
    print(f"  → Maker net margin       = {margin:+.2f} bps/fill")

    # 帯域別
    print("\n  スプレッド帯域別 Realized Spread:")
    for lo, hi in spread_bands:
        band_fills = [r for r in filled
                      if lo <= _safe_float(r, "spread_at_order") < hi]
        if not band_fills:
            continue
        rsd_band = realized_spread_decomposition(band_fills)
        label = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
        print(f"    {label:>10s} JPY: quoted_hs={rsd_band['quoted_hs_avg']:.2f}  "
              f"realized_hs={rsd_band['realized_hs_avg']:.2f}  "
              f"AS_cost={rsd_band['as_cost_avg']:.2f} "
              f"({rsd_band['as_cost_pct']:.0f}%)  n={len(band_fills)}")

    # ---- §5 最適 min_spread シミュレーション ----
    print("\n" + "─" * 72)
    print("§5 最適 min_spread 閾値シミュレーション")
    print("─" * 72)

    opt_results = optimal_min_spread_simulation(records)
    if opt_results:
        print(f"  {'閾値(JPY)':>10s} {'fill率':>8s} {'avg30':>8s} "
              f"{'sum30':>10s} {'n_fills':>8s}")
        best_sum = max(opt_results, key=lambda x: x["expected_sum30"])
        best_avg = max(opt_results, key=lambda x: x["expected_avg30"])
        for r in opt_results:
            marker = ""
            if r["threshold_jpy"] == best_sum["threshold_jpy"]:
                marker += " ★sum"
            if r["threshold_jpy"] == best_avg["threshold_jpy"]:
                marker += " ★avg"
            print(f"  {r['threshold_jpy']:>10.0f} {r['fill_rate']:>7.1%} "
                  f"{r['expected_avg30']:>+8.2f} {r['expected_sum30']:>+10.1f} "
                  f"{r['n_fills']:>8.0f}{marker}")

    # ---- §6 Fill Probability Model ----
    print("\n" + "─" * 72)
    print("§6 Fill Probability Model (Offset vs Fill Rate vs PnL)")
    print("─" * 72)

    fp_results = fill_probability_by_offset(records)
    if fp_results:
        print(f"  {'offset帯':>16s} {'n':>6s} {'fill確率':>8s} "
              f"{'avg30':>8s} {'E[rev]':>8s}")
        for r in fp_results:
            print(f"  [{r['offset_lo']:.2f}, {r['offset_hi']:.2f}] "
                  f"{r['n']:>6.0f} {r['fill_prob']:>7.1%} "
                  f"{r['avg_pnl30']:>+8.2f} {r['expected_revenue']:>+8.4f}")

    # ---- §7 Bootstrap CI by spread band + NFQ counterfactual ----
    print("\n" + "─" * 72)
    print("§7 スプレッド帯域別 Bootstrap CI")
    print("─" * 72)

    for lo, hi in spread_bands:
        band_fills = [r for r in filled
                      if lo <= _safe_float(r, "spread_at_order") < hi]
        pnls = np.array([_pnl_30s(r) for r in band_fills
                         if _pnl_30s(r) is not None])
        if len(pnls) == 0:
            continue
        label = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
        mean, ci_lo, ci_hi = bootstrap_ci(pnls)
        contains_zero = "✓" if ci_lo <= 0 <= ci_hi else "✗"
        print(f"  {label:>10s} JPY: mean={mean:+.3f} "
              f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"n={len(pnls)}  0∈CI: {contains_zero}")

    # ---- §8 時間帯別 AS 率 ----
    print("\n" + "─" * 72)
    print("§8 時間帯別 Adverse Selection 率")
    print("─" * 72)

    hour_bins: dict[int, list[bool]] = {}
    for r in filled:
        ts = r.get("timestamp")
        if ts is None:
            continue
        try:
            from datetime import datetime, timezone
            ts_f = float(ts)  # type: ignore[arg-type]
            dt = datetime.fromtimestamp(ts_f, tz=timezone.utc)
            h = dt.hour
        except (ValueError, TypeError, OSError):
            continue
        hour_bins.setdefault(h, []).append(_is_as(r))

    if hour_bins:
        print(f"  {'Hour(UTC)':>10s} {'n':>6s} {'AS率':>8s} {'AS件':>6s} {'bar'}")
        for h in sorted(hour_bins.keys()):
            vals = hour_bins[h]
            n = len(vals)
            as_n = sum(vals)
            rate = as_n / n if n > 0 else 0
            bar = "█" * int(rate * 40)
            print(f"  {h:>10d} {n:>6d} {rate:>7.1%} {as_n:>6d} {bar}")

    print("\n" + "=" * 72)
    print("分析完了")
    print("=" * 72)


if __name__ == "__main__":
    main()
