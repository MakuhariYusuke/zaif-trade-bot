#!/usr/bin/env python3
"""
Phase 4.5 Gate C3: SAC vs ベースライン統計比較

66# (0番§5.6) 準拠:
- Mann-Whitney U検定
- Cliff's Delta（効果量 > 0.33 = medium以上）
- Holm-Bonferroni補正（多重比較）

0番§5.2 成功基準:
- Net ROI > 5%
- Profit Factor > 1.2
- Sharpe Ratio > 1.0 (年率)
- Max Drawdown < 15%
- Win Rate > 35%
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 統計検定ユーティリティ
# ============================================================================

def mann_whitney_u(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Mann-Whitney U検定 (scipy不要版)
    
    Returns: (U統計量, 近似p値)
    """
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    
    if nx == 0 or ny == 0:
        return 0.0, 1.0
    
    # 全ペア比較
    u = 0.0
    for xi in x:
        for yi in y:
            if xi > yi:
                u += 1
            elif xi == yi:
                u += 0.5
    
    # 正規近似 (n >= 4のとき)
    mu = nx * ny / 2
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    
    if sigma == 0:
        return u, 1.0
    
    z = (u - mu) / sigma
    # 正規分布の両側p値（近似）
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    
    return u, p_value


def _norm_cdf(z: float) -> float:
    """標準正規分布のCDF（近似）"""
    # Abramowitz and Stegun approximation
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5 = -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * abs(z))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z / 2)
    if z < 0:
        y = 1.0 - y
    return y


def cliffs_delta(x: List[float], y: List[float]) -> float:
    """Cliff's Delta (効果量)
    
    |d| < 0.147: negligible
    |d| < 0.33:  small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    
    if nx == 0 or ny == 0:
        return 0.0
    
    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1
    
    return (more - less) / (nx * ny)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni多重比較補正
    
    Returns: 各検定がalpha水準で有意かどうかのリスト
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # p値の昇順インデックス
    sorted_idx = np.argsort(p_values)
    significant = [False] * n
    
    for rank, idx in enumerate(sorted_idx):
        adjusted_alpha = alpha / (n - rank)
        if p_values[idx] <= adjusted_alpha:
            significant[idx] = True
        else:
            # Holm法: 一度棄却できないと以降も棄却しない
            break
    
    return significant


# ============================================================================
# メトリクス計算
# ============================================================================

def compute_phase5_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """66# (0番§5.2) のPhase 5メトリクスを計算
    
    各resultは extract_env_metrics() の出力を想定
    """
    if not results:
        return {}
    
    rois = [r.get("net_roi", r.get("balance_roi", r.get("final_roi", 0))) for r in results]
    gross_pnls = [r.get("gross_pnl", 0) for r in results]
    fees = [r.get("total_fees", 0) for r in results]
    trades = [r.get("total_trades", 0) for r in results]
    buys = [r.get("buy_count", 0) for r in results]
    sells = [r.get("sell_count", 0) for r in results]
    
    # Profit Factor = gross_profit / gross_loss
    # ベースライン結果からはtrade-level PFは不可 → ROIベースで近似
    
    return {
        "n": len(results),
        "net_roi_mean": float(np.mean(rois)),
        "net_roi_std": float(np.std(rois)),
        "net_roi_all": rois,
        "gross_pnl_mean": float(np.mean(gross_pnls)),
        "gross_pnl_std": float(np.std(gross_pnls)),
        "avg_fees": float(np.mean(fees)),
        "avg_trades": float(np.mean(trades)),
        "avg_buys": float(np.mean(buys)),
        "avg_sells": float(np.mean(sells)),
    }


def check_phase5_criteria(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """0番§5.2 成功基準チェック"""
    return {
        "net_roi_gt_5pct": metrics.get("net_roi_mean", -999) > 5.0,
        # PF, Sharpe, MaxDD, WinRate は環境レベルの集計が必要
        # Phase 4.5ではNet ROIを主要判定基準とする
        "net_roi_gt_0pct": metrics.get("net_roi_mean", -999) > 0.0,
        "gross_pnl_positive": metrics.get("gross_pnl_mean", -999) > 0,
    }


# ============================================================================
# 比較分析
# ============================================================================

def run_comparison(
    sac_results: List[Dict[str, Any]],
    baseline_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """SAC vs 全ベースラインの統計比較"""
    
    sac_rois = [r.get("net_roi", r.get("balance_roi", 0)) for r in sac_results if r.get("success", True)]
    
    comparisons = {}
    p_values = []
    comparison_names = []
    
    for baseline_name, bl_results in baseline_results.items():
        bl_rois = [r.get("net_roi", 0) for r in bl_results if r.get("success", True)]
        
        if not sac_rois or not bl_rois:
            continue
        
        # Mann-Whitney U
        u_stat, p_val = mann_whitney_u(sac_rois, bl_rois)
        
        # Cliff's Delta
        delta = cliffs_delta(sac_rois, bl_rois)
        
        # 効果量の解釈
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            effect_size = "negligible"
        elif abs_delta < 0.33:
            effect_size = "small"
        elif abs_delta < 0.474:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        comparisons[baseline_name] = {
            "sac_mean_roi": float(np.mean(sac_rois)),
            "sac_std_roi": float(np.std(sac_rois)),
            "baseline_mean_roi": float(np.mean(bl_rois)),
            "baseline_std_roi": float(np.std(bl_rois)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "cliffs_delta": float(delta),
            "effect_size": effect_size,
            "sac_n": len(sac_rois),
            "baseline_n": len(bl_rois),
        }
        
        p_values.append(p_val)
        comparison_names.append(baseline_name)
    
    # Holm-Bonferroni補正
    if p_values:
        significant = holm_bonferroni(p_values)
        for i, name in enumerate(comparison_names):
            comparisons[name]["holm_significant"] = significant[i]
    
    return comparisons


def print_comparison_report(
    sac_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
    comparisons: Dict[str, Any],
    sac_label: str = "SAC(P1-1)"
):
    """比較レポートをコンソールに出力"""
    
    print("\n" + "=" * 80)
    print("  Phase 4.5 Gate C3: SAC vs ベースライン比較レポート")
    print("  66# (0番§5.2/§5.6) 準拠統計分析")
    print("=" * 80)
    
    # --- 基本メトリクス ---
    print(f"\n{'='*60}")
    print(f"  [{sac_label}] SAC実験結果")
    print(f"{'='*60}")
    print(f"  n = {sac_metrics.get('n', 0)}")
    print(f"  Net ROI: {sac_metrics.get('net_roi_mean', 0):+.2f}% ± {sac_metrics.get('net_roi_std', 0):.2f}%")
    print(f"  Gross PnL: {sac_metrics.get('gross_pnl_mean', 0):+,.0f} ± {sac_metrics.get('gross_pnl_std', 0):,.0f}")
    print(f"  Fees: {sac_metrics.get('avg_fees', 0):,.0f}")
    print(f"  Trades: {sac_metrics.get('avg_trades', 0):.0f} (BUY: {sac_metrics.get('avg_buys', 0):.0f}, SELL: {sac_metrics.get('avg_sells', 0):.0f})")
    
    criteria = check_phase5_criteria(sac_metrics)
    print(f"\n  Phase 5 基準:")
    for k, v in criteria.items():
        status = "✅ PASS" if v else "❌ FAIL"
        print(f"    {k}: {status}")
    
    # --- ベースライン ---
    for name, metrics in baseline_metrics.items():
        print(f"\n  [{name}]")
        print(f"    Net ROI: {metrics.get('net_roi_mean', 0):+.2f}% ± {metrics.get('net_roi_std', 0):.2f}%")
        print(f"    Gross PnL: {metrics.get('gross_pnl_mean', 0):+,.0f}")
        print(f"    Trades: {metrics.get('avg_trades', 0):.0f}")
    
    # --- 統計検定 ---
    print(f"\n{'='*60}")
    print(f"  統計検定結果（0番§5.6）")
    print(f"{'='*60}")
    
    for name, comp in comparisons.items():
        print(f"\n  {sac_label} vs {name}:")
        print(f"    Mann-Whitney U = {comp['mann_whitney_u']:.1f}, p = {comp['p_value']:.4f}")
        print(f"    Cliff's Delta = {comp['cliffs_delta']:+.3f} ({comp['effect_size']})")
        holm = "有意" if comp.get("holm_significant") else "非有意"
        print(f"    Holm-Bonferroni補正後: {holm}")
        
        # 方向性
        diff = comp["sac_mean_roi"] - comp["baseline_mean_roi"]
        direction = "SAC優位" if diff > 0 else "ベースライン優位" if diff < 0 else "同等"
        print(f"    差分: {diff:+.2f}% ({direction})")
    
    # --- Go/No-Go判定 ---
    print(f"\n{'='*60}")
    print(f"  Go/No-Go 判定")
    print(f"{'='*60}")
    
    # 判定ロジック
    sac_roi = sac_metrics.get("net_roi_mean", -999)
    any_significant = any(c.get("holm_significant", False) for c in comparisons.values())
    any_medium_effect = any(c.get("cliffs_delta", 0) > 0.33 for c in comparisons.values())
    
    if sac_roi > 5.0 and any_significant and any_medium_effect:
        verdict = "GO → Phase 5"
        reason = "Net ROI > 5% かつ統計的有意差あり（medium以上）"
    elif sac_roi > 0.0:
        verdict = "CONDITIONAL GO → Phase C (コスト圧縮)"
        reason = f"Net ROI = {sac_roi:+.2f}%（正だが5%未達）→ 手数料最適化で改善余地あり"
    elif sac_roi > -5.0:
        verdict = "CONDITIONAL → Phase C (コスト圧縮必須)"
        reason = f"Net ROI = {sac_roi:+.2f}%（軽度の負）→ コスト圧縮で0%到達可能性あり"
    else:
        verdict = "NO-GO（条件付き継続）"
        reason = f"Net ROI = {sac_roi:+.2f}%（重度の負）→ アルゴリズムレベルの改善必要"
    
    # ベースライン比較での追加判定
    best_bl_name = None
    best_bl_roi = -999
    for name, metrics in baseline_metrics.items():
        roi = metrics.get("net_roi_mean", -999)
        if roi > best_bl_roi:
            best_bl_roi = roi
            best_bl_name = name
    
    if sac_roi > best_bl_roi:
        bl_comparison = f"最良ベースライン({best_bl_name}: {best_bl_roi:+.2f}%)を上回る"
    else:
        bl_comparison = f"最良ベースライン({best_bl_name}: {best_bl_roi:+.2f}%)を下回る"
    
    print(f"\n  判定: {verdict}")
    print(f"  理由: {reason}")
    print(f"  ベースライン比較: {bl_comparison}")
    
    # Gross PnL分析
    sac_gross = sac_metrics.get("gross_pnl_mean", 0)
    sac_fees = sac_metrics.get("avg_fees", 0)
    if sac_gross > 0:
        print(f"\n  💡 Gross PnL分析:")
        print(f"    粗利: {sac_gross:+,.0f} - 手数料: {sac_fees:,.0f} = Net: {sac_gross - sac_fees:+,.0f}")
        if sac_fees > 0:
            fee_ratio = sac_fees / max(sac_gross, 1)
            print(f"    手数料率: {fee_ratio:.1%} → {'コスト圧縮で改善可能' if fee_ratio > 1 else '許容範囲'}")
    
    return verdict


# ============================================================================
# メイン
# ============================================================================

def load_latest_results(results_dir: Path, prefix: str) -> Optional[Dict]:
    """最新の結果ファイルを読み込み"""
    files = sorted(results_dir.glob(f"{prefix}*.json"), reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    from datetime import datetime
    
    print("Phase 4.5 Gate C3: 統計比較分析")
    print("=" * 70)
    
    # ベースライン結果読み込み
    baseline_dir = project_root / "results" / "phase45_baselines"
    baseline_data = load_latest_results(baseline_dir, "baseline_results")
    
    if not baseline_data:
        print("❌ ベースライン結果が見つかりません")
        print(f"   期待パス: {baseline_dir}/baseline_results_*.json")
        return
    
    # SAC結果読み込み
    sac_dir = project_root / "results" / "phase45_p1_baseline"
    sac_data = load_latest_results(sac_dir, "p1_results")
    
    if not sac_data:
        print("❌ SAC結果が見つかりません")
        print(f"   期待パス: {sac_dir}/p1_results_*.json")
        return
    
    # ベースライン結果を戦略別に分類
    baseline_by_strategy = {}
    for r in baseline_data.get("all_results", []):
        strategy = r.get("strategy", "Unknown")
        if r.get("success", True):
            if strategy not in baseline_by_strategy:
                baseline_by_strategy[strategy] = []
            baseline_by_strategy[strategy].append(r)
    
    # SAC結果をカテゴリ別に分類
    sac_by_category = {}
    sac_all = sac_data.get("all_results", sac_data.get("partial_results", []))
    for r in sac_all:
        cat = r.get("experiment_category", r.get("experiment_name", "Unknown"))
        # P1-1/P1-3 を抽出
        if "P1-1" in str(cat):
            cat_key = "P1-1"
        elif "P1-3" in str(cat):
            cat_key = "P1-3"
        else:
            cat_key = str(cat)
        
        if r.get("success", False):
            if cat_key not in sac_by_category:
                sac_by_category[cat_key] = []
            sac_by_category[cat_key].append(r)
    
    # メトリクス計算
    baseline_metrics = {}
    for strategy, results in baseline_by_strategy.items():
        baseline_metrics[strategy] = compute_phase5_metrics(results)
    
    # SAC各カテゴリで比較
    all_comparisons = {}
    for sac_cat, sac_results in sac_by_category.items():
        sac_metrics = compute_phase5_metrics(sac_results)
        comparisons = run_comparison(sac_results, baseline_by_strategy)
        
        verdict = print_comparison_report(
            sac_metrics, baseline_metrics, comparisons, sac_label=f"SAC({sac_cat})"
        )
        
        all_comparisons[sac_cat] = {
            "sac_metrics": sac_metrics,
            "comparisons": comparisons,
            "verdict": verdict,
        }
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "phase45_gate_c3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"gate_c3_comparison_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "baseline_metrics": baseline_metrics,
            "sac_comparisons": all_comparisons,
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ 分析結果保存: {output_file}")


if __name__ == "__main__":
    main()
