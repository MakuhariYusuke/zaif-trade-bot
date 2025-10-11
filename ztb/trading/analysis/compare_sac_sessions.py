"""
SAC訓練ログ比較スクリプト
v395a vs v395c の比較
統計的検定付き (t検定 + p平均法)

特徴:
- TensorBoardログからメトリクスを抽出
- セッション間の統計的比較
- t検定による個別メトリクスの有意差検定
- p平均法による総合的有意性評価
- 効果量の計算

p平均法 (p-mean method) について:
p平均法は、複数の統計検定のp値を統合して総合的な有意性を評価する手法です。
このスクリプトでは以下の方法を実装しています：

1. 算術平均 (arithmetic mean):
   - 複数のp値を単純に平均する
   - 直感的で理解しやすい
   - 極端なp値の影響を受けやすい

2. 幾何平均 (geometric mean):
   - p値の対数を平均し、指数関数で戻す
   - 極端な値の影響を緩和
   - 0に近いp値を適切に扱える

使用例:
- 複数のメトリクス（ent_coef, critic_loss, actor_loss）のp値を統合
- 全体としてセッション間に有意な差があるかを評価
- p_mean < 0.05 で全体として有意と判断

参考文献:
- https://note.com/a_small_hamster/n/n718dbe6bfe9e
- https://note.com/btcml/n/n0d9575882640
"""
import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import json
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats

# 定数定義
DEFAULT_METRICS = ['train/ent_coef', 'train/critic_loss', 'train/actor_loss']
DEFAULT_OUTPUT_FILE = "sac_comparison.json"
DEFAULT_LOG_DIR = "checkpoints/sac_session"
DEFAULT_SESSION_IDS = [5, 6]  # SAC_5 = v395a, SAC_6 = v395c

def p_mean_method(p_values: List[float], method: str = 'arithmetic') -> float:
    """
    p平均法による総合p値の計算

    p平均法は、複数の独立した統計検定のp値を統合し、
    全体として統計的有意性があるかを評価する手法です。

    Args:
        p_values: p値のリスト（0.0 ~ 1.0の範囲）
        method: 平均化手法
            - 'arithmetic': 算術平均（単純平均）
            - 'geometric': 幾何平均（対数変換後平均）

    Returns:
        総合p値（0.0 ~ 1.0）

    算術平均の特徴:
        - 直感的で理解しやすい
        - 全てのp値に等しい重み付け
        - 極端なp値（0.99など）の影響を受けやすい

    幾何平均の特徴:
        - 極端なp値の影響を緩和
        - 0に非常に近いp値を適切に扱える
        - 統計学的によりロバスト

    使用例:
        # 3つのメトリクスのp値統合
        p_values = [0.03, 0.07, 0.02]  # 個別のt検定結果
        combined_p = p_mean_method(p_values, 'geometric')
        significant = combined_p < 0.05

    注意事項:
        - p値が完全に独立であることを仮定
        - 相関のある検定では結果が保守的になる可能性
        - 解釈時は個別の検定結果も確認すること
    """
    if not p_values:
        return 1.0

    p_array = np.array(p_values)

    if method == 'arithmetic':
        # 算術平均
        return float(np.mean(p_array))
    elif method == 'geometric':
        # 幾何平均 (0を避けるため小さな値を加算)
        p_array = np.clip(p_array, 1e-10, 1.0)
        return float(np.exp(np.mean(np.log(p_array))))
    else:
        raise ValueError(f"Unknown method: {method}")

def perform_statistical_tests(session_data: Dict[int, Dict[str, List[float]]],
                            metrics: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    セッション間の統計的検定を実行

    Welchのt検定（等分散を仮定しない）を使用して、
    各メトリクスにおけるセッション間の差の統計的有意性を評価します。

    Args:
        session_data: セッションごとのメトリクス生データ
            形式: {session_id: {metric_name: [value1, value2, ...]}}
        metrics: 検定対象のメトリクス名リスト

    Returns:
        検定結果の辞書
        形式: {
            metric_name: {
                't_statistic': float,      # t統計量
                'p_value': float,          # p値
                'significant': bool,       # p < 0.05かどうか
                'session_a_mean': float,   # セッションAの平均値
                'session_b_mean': float,   # セッションBの平均値
                'effect_size': float       # 効果量（標準化された差）
            }
        }

    効果量の解釈:
        - 0.2: 小さな効果
        - 0.5: 中程度の効果
        - 0.8: 大きな効果

    注意事項:
        - データが正規分布に従うことを仮定
        - サンプルサイズが小さい場合、検定力が低下
        - 複数の検定を行う場合、p値の補正を検討
    """
    results: Dict[str, Dict[str, Any]] = {}
    session_ids = list(session_data.keys())

    if len(session_ids) != 2:
        print("⚠️ Statistical tests require exactly 2 sessions")
        return results

    session_a, session_b = session_ids

    for metric in metrics:
        if metric not in session_data[session_a] or metric not in session_data[session_b]:
            continue

        values_a = session_data[session_a][metric]
        values_b = session_data[session_b][metric]

        if len(values_a) < 2 or len(values_b) < 2:
            print(f"⚠️ Insufficient data for {metric} statistical test")
            continue

        # t検定実行
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # 戻り値がnumpyのスカラー型の場合、floatに変換
        t_stat = float(t_stat) if hasattr(t_stat, '__float__') else t_stat
        p_value = float(p_value) if hasattr(p_value, '__float__') else p_value

        # 有意性判断 (p < 0.05)
        significant = p_value < 0.05

        results[metric] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'session_a_mean': np.mean(values_a),
            'session_b_mean': np.mean(values_b),
            'effect_size': abs(np.mean(values_a) - np.mean(values_b)) / np.std(values_a + values_b)
        }

    return results

def compare_sac_sessions(log_dir: str, session_ids: List[int]) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, Any]]:
    """
    複数のSACセッションを比較

    Args:
        log_dir: TensorBoardログが保存されているディレクトリパス
        session_ids: 比較するセッションIDのリスト

    Returns:
        Tuple of (session_stats, statistical_results)
        session_stats: 各セッションのメトリクス統計
        statistical_results: 統計的検定結果とp平均法結果
    """
    print(f"📊 Comparing SAC sessions: {session_ids}")
    print("=" * 80)
    
    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    session_raw_data: Dict[int, Dict[str, List[float]]] = {}  # 統計的検定用の生データ
    
    for session_id in session_ids:
        session_path = os.path.join(log_dir, f"SAC_{session_id}")
        
        if not os.path.exists(session_path):
            print(f"⚠️ Session SAC_{session_id} not found at {session_path}")
            continue
        
        print(f"\n📁 Analyzing SAC_{session_id}...")
        
        ea = event_accumulator.EventAccumulator(session_path)
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        if not isinstance(scalar_tags, list):
            print(f"⚠️ Invalid scalar_tags format for session {session_id}")
            continue
        session_results = {}
        session_raw_data[session_id] = {}
        
        metrics = DEFAULT_METRICS
        
        for tag in metrics:
            if tag in scalar_tags:
                events = ea.Scalars(tag)
                values = [e.value for e in events]
                
                if values:
                    session_results[tag] = {
                        'initial': values[0],
                        'final': values[-1],
                        'max': max(values),
                        'min': min(values),
                        'mean': sum(values) / len(values),
                    }
                    # 統計的検定用の生データを保存
                    session_raw_data[session_id][tag] = values
        
        results[session_id] = session_results
    
    # Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    
    for metric in ['train/ent_coef', 'train/critic_loss', 'train/actor_loss']:
        print(f"\n🔍 {metric}:")
        print("-" * 80)
        
        for session_id in session_ids:
            if session_id in results and metric in results[session_id]:
                data = results[session_id][metric]
                print(f"  SAC_{session_id}:")
                print(f"    Initial: {data['initial']:.4e}")
                print(f"    Final:   {data['final']:.4e}")
                print(f"    Max:     {data['max']:.4e}")
                print(f"    Min:     {data['min']:.4e}")
                print(f"    Mean:    {data['mean']:.4e}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS")
    print("=" * 80)
    
    # 統計的検定実行
    stat_results = {}
    p_mean_arithmetic = None
    p_mean_geometric = None
    
    if len(session_raw_data) == 2:
        print("\n" + "=" * 80)
        print("📊 STATISTICAL TESTS (t-test)")
        print("=" * 80)
        
        stat_results = perform_statistical_tests(session_raw_data, DEFAULT_METRICS)
        
        for metric, test_result in stat_results.items():
            print(f"\n🔬 {metric}:")
            print(f"  t-statistic: {test_result['t_statistic']:.4f}")
            print(f"  p-value: {test_result['p_value']:.4f}")
            print(f"  Significant (p<0.05): {'✅ Yes' if test_result['significant'] else '❌ No'}")
            print(f"  Effect size: {test_result['effect_size']:.4f}")
            print(f"  Session A mean: {test_result['session_a_mean']:.6f}")
            print(f"  Session B mean: {test_result['session_b_mean']:.6f}")
        
        # p平均法
        if stat_results:
            p_values = [result['p_value'] for result in stat_results.values()]
            p_mean_arithmetic = p_mean_method(p_values, 'arithmetic')
            p_mean_geometric = p_mean_method(p_values, 'geometric')
            
            print(f"\n🔬 P-MEAN METHOD:")
            print(f"  Arithmetic mean p-value: {p_mean_arithmetic:.4f}")
            print(f"  Geometric mean p-value: {p_mean_geometric:.4f}")
            print(f"  Overall significant (p<0.05): {'✅ Yes' if p_mean_arithmetic < 0.05 else '❌ No'}")
    
    # Compare ent_coef
    if 'train/ent_coef' in results.get(session_ids[0], {}) and 'train/ent_coef' in results.get(session_ids[1], {}):
        ent_coef_5 = results[session_ids[0]]['train/ent_coef']['final']
        ent_coef_6 = results[session_ids[1]]['train/ent_coef']['final']
        
        print(f"\n📈 Entropy Coefficient (ent_coef):")
        print(f"  SAC_5 (v395a): {ent_coef_5:.2f}")
        print(f"  SAC_6 (v395c): {ent_coef_6:.2f}")
        
        if ent_coef_6 < ent_coef_5:
            improvement = (ent_coef_5 - ent_coef_6) / ent_coef_5 * 100
            print(f"  ✅ Improvement: -{improvement:.1f}%")
        else:
            print(f"  ⚠️  Increased: +{(ent_coef_6 - ent_coef_5) / ent_coef_5 * 100:.1f}%")
    
    # Compare critic_loss
    if 'train/critic_loss' in results.get(session_ids[0], {}) and 'train/critic_loss' in results.get(session_ids[1], {}):
        critic_5 = results[session_ids[0]]['train/critic_loss']['final']
        critic_6 = results[session_ids[1]]['train/critic_loss']['final']
        
        print(f"\n📉 Critic Loss:")
        print(f"  SAC_5 (v395a): {critic_5:.2e}")
        print(f"  SAC_6 (v395c): {critic_6:.2e}")
        
        if critic_6 < critic_5:
            improvement = (critic_5 - critic_6) / critic_5 * 100
            print(f"  ✅ Improvement: -{improvement:.1f}%")
        else:
            print(f"  ❌ Increased: +{(critic_6 - critic_5) / critic_5 * 100:.1f}%")
    
    # Save comparison
    output_file = DEFAULT_OUTPUT_FILE
    output_data = {
        'session_stats': results,
        'statistical_tests': stat_results,
        'p_mean_results': {
            'arithmetic': p_mean_arithmetic,
            'geometric': p_mean_geometric
        }
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Comparison saved to: {output_file}")
    print("=" * 80)
    
    statistical_results = {
        'tests': stat_results,
        'p_mean': {
            'arithmetic': p_mean_arithmetic,
            'geometric': p_mean_geometric
        }
    }
    
    return results, statistical_results

if __name__ == "__main__":
    log_dir = DEFAULT_LOG_DIR
    session_ids = DEFAULT_SESSION_IDS
    stats, statistical_results = compare_sac_sessions(log_dir, session_ids)
    print(f"\n📈 Session comparison completed. Statistical significance: {statistical_results}")
