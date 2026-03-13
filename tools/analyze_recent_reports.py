"""最近のトレーニングレポートを分析するスクリプト"""
from typing import TypedDict
from pathlib import Path

from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.trading.environment.components.rewards.utils import RewardUtils
from ztb.utils.safety import ensure_dict, safe_to_float


class RecentReportAnalysis(TypedDict):
    file: str
    reward: float
    buy: float
    sell: float
    hold: float
    buy_sell_diff: float


def analyze_reports(limit: int = 20) -> list[RecentReportAnalysis]:
    """最近のレポートを分析"""
    reports = get_recent_training_reports(limit=limit, reports_dir=Path("reports"))
    
    results: list[RecentReportAnalysis] = []
    for report_path in reports:
        try:
            data = load_training_report(report_path)
            if data is None:
                raise ValueError("Could not load JSON payload")
            
            stats = ensure_dict(data.get('training_stats'))
            actions = extract_action_distribution_from_payload(data)
            
            reward = safe_to_float(stats.get('final_reward'), 0.0)
            
            results.append({
                'file': report_path.name,
                'reward': reward,
                'buy': actions.get('BUY', 0),
                'sell': actions.get('SELL', 0),
                'hold': actions.get('HOLD', 0),
                'buy_sell_diff': RewardUtils.calculate_buy_sell_diff(
                    actions.get('BUY', 0), actions.get('SELL', 0)
                ),
            })
        except Exception as e:
            print(f"Error processing {report_path}: {e}")
    
    return results

def print_analysis(results: list[RecentReportAnalysis]) -> None:
    """分析結果を表示"""
    if not results:
        print("\nNo valid reports found")
        return

    print("\n" + "="*80)
    print("最近のトレーニングレポート分析")
    print("="*80)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['file'][:50]}")
        print(f"   Reward: {r['reward']:.2f}")
        print(f"   BUY: {r['buy']:.1%}, SELL: {r['sell']:.1%}, HOLD: {r['hold']:.1%}")
        print(f"   BUY-SELL差: {r['buy_sell_diff']:.1%}")
    
    # 統計サマリー
    print("\n" + "="*80)
    print("統計サマリー")
    print("="*80)
    
    avg_reward = sum(r['reward'] for r in results) / len(results)
    avg_buy = sum(r['buy'] for r in results) / len(results)
    avg_sell = sum(r['sell'] for r in results) / len(results)
    avg_hold = sum(r['hold'] for r in results) / len(results)
    avg_diff = sum(r['buy_sell_diff'] for r in results) / len(results)
    
    print(f"\n平均報酬: {avg_reward:.2f}")
    print(f"平均BUY: {avg_buy:.1%}")
    print(f"平均SELL: {avg_sell:.1%}")
    print(f"平均HOLD: {avg_hold:.1%}")
    print(f"平均BUY-SELL差: {avg_diff:.1%}")
    
    # 極端なケースの分析
    extreme_cases = [r for r in results if r['buy_sell_diff'] > 0.5]
    if extreme_cases:
        print(f"\n⚠️  極端なバイアス（BUY-SELL差>50%）: {len(extreme_cases)}件")
        for case in extreme_cases[:3]:
            print(f"   - Reward: {case['reward']:.2f}, BUY: {case['buy']:.1%}, SELL: {case['sell']:.1%}")

if __name__ == '__main__':
    results = analyze_reports(limit=20)
    print_analysis(results)
