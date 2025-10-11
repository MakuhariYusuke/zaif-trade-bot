"""
アクション分布問題診断ツール

学習結果のアクション分布を分析し、問題の原因を特定します。
ユーザーから情報を収集して詳細な診断を行います。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def print_header():
    """ヘッダーを表示"""
    print("="*70)
    print("🔍 アクション分布問題診断ツール")
    print("="*70)
    print()


def collect_action_distribution() -> Optional[Dict[str, float]]:
    """アクション分布データを収集"""
    print("📊 学習結果のアクション分布を入力してください:")
    print("   (学習ログやTensorBoardから確認できる最終的な分布)")
    print()
    
    try:
        hold = float(input("HOLD率 (0.0-1.0): "))
        buy = float(input("BUY率  (0.0-1.0): "))
        sell = float(input("SELL率 (0.0-1.0): "))
        
        total = hold + buy + sell
        if abs(total - 1.0) > 0.01:
            print(f"⚠️ 警告: 合計が {total:.3f} です（1.0であるべき）")
            normalize = input("正規化しますか？ (y/n): ").lower() == 'y'
            if normalize:
                hold /= total
                buy /= total
                sell /= total
                print(f"✓ 正規化後: HOLD={hold:.3f}, BUY={buy:.3f}, SELL={sell:.3f}")
        
        return {
            'hold': hold,
            'buy': buy,
            'sell': sell
        }
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 入力がキャンセルされました")
        return None


def diagnose_distribution(dist: Dict[str, float]) -> Dict[str, any]:
    """アクション分布を診断"""
    hold = dist['hold']
    buy = dist['buy']
    sell = dist['sell']
    
    print("\n" + "="*70)
    print("📊 現在のアクション分布:")
    print("="*70)
    print(f"  HOLD: {hold*100:6.2f}%")
    print(f"  BUY:  {buy*100:6.2f}%")
    print(f"  SELL: {sell*100:6.2f}%")
    print()
    
    # 理想的な分布
    ideal = 1.0 / 3.0
    tolerance_warning = 0.10  # ±10%
    tolerance_critical = 0.20  # ±20%
    
    issues = []
    severity = []
    root_causes = []
    recommendations = []
    
    # === SELL bias 診断 ===
    if sell > (ideal + tolerance_critical):
        issues.append(f"🔴 深刻なSELL bias: {sell*100:.1f}% (理想: 33.3%)")
        severity.append('CRITICAL')
        root_causes.extend([
            "報酬関数がSELLアクションを過度に優遇している可能性",
            "Lagrange制約が機能していない",
            "PAN (Per-Action Normalization)が無効または不十分",
            "SELL時の利益計算に問題がある可能性"
        ])
        recommendations.extend([
            "1. Lagrange制約を強化:",
            "   - lagrange_eta: 0.05 → 0.1",
            "   - lagrange_lambda_max: 2.0 → 3.0",
            "2. PAN (Per-Action Normalization)を有効化:",
            "   - enable_pan: true",
            "3. エントロピー係数を増加:",
            "   - ent_coef: 0.1 → 0.2",
            "4. SELL報酬にペナルティを追加:",
            "   - profit_bonus_multipliers: [1.0, 1.0, 0.8] (SELL=0.8)"
        ])
    elif sell > (ideal + tolerance_warning):
        issues.append(f"🟡 軽度のSELL bias: {sell*100:.1f}% (理想: 33.3%)")
        severity.append('WARNING')
        root_causes.extend([
            "Lagrange制約の調整不足",
            "エントロピー係数が低い"
        ])
        recommendations.extend([
            "1. Lagrange制約を微調整:",
            "   - lagrange_eta: 0.05 → 0.07",
            "2. エントロピー係数を増加:",
            "   - ent_coef: 0.1 → 0.15"
        ])
    
    # === HOLD bias 診断 ===
    if hold > (ideal + tolerance_critical):
        issues.append(f"🔴 深刻なHOLD bias: {hold*100:.1f}% (理想: 33.3%)")
        severity.append('CRITICAL')
        root_causes.extend([
            "アクションペナルティが過度に大きい",
            "取引コストが高すぎる",
            "min_holding_periodが厳しすぎる",
            "エントロピー係数が低すぎる"
        ])
        recommendations.extend([
            "1. アクションペナルティを削減:",
            "   - action_penalty_scale: 0.01 → 0.005",
            "   - trade_frequency_penalty: 0.01 → 0.005",
            "2. エントロピー係数を増加:",
            "   - ent_coef: 0.1 → 0.3",
            "3. min_holding_periodを緩和:",
            "   - 現在値を確認し、5以下に設定",
            "4. 取引コストを確認:",
            "   - transaction_cost: 0.001が適切か確認"
        ])
    elif hold > (ideal + tolerance_warning):
        issues.append(f"🟡 軽度のHOLD bias: {hold*100:.1f}% (理想: 33.3%)")
        severity.append('WARNING')
        root_causes.extend([
            "保守的な報酬設定",
            "エントロピー係数の不足"
        ])
        recommendations.extend([
            "1. エントロピー係数を増加:",
            "   - ent_coef: 0.1 → 0.15",
            "2. アクションペナルティを微調整:",
            "   - action_penalty_scale: 0.01 → 0.008"
        ])
    
    # === BUY bias 診断 ===
    if buy > (ideal + tolerance_critical):
        issues.append(f"🔴 深刻なBUY bias: {buy*100:.1f}% (理想: 33.3%)")
        severity.append('CRITICAL')
        root_causes.extend([
            "報酬関数がBUYアクションを過度に優遇している",
            "データセットにロングポジション優位のバイアス",
            "SELLペナルティが過度に大きい"
        ])
        recommendations.extend([
            "1. BUY報酬にペナルティを追加:",
            "   - profit_bonus_multipliers: [1.0, 0.8, 1.0] (BUY=0.8)",
            "2. データセットを確認:",
            "   - トレンド方向のバランスを確認",
            "3. アクション分布を強制:",
            "   - curriculum_stage: 'forced_balance'",
            "   - enable_stratified_sampling: true"
        ])
    elif buy > (ideal + tolerance_warning):
        issues.append(f"🟡 軽度のBUY bias: {buy*100:.1f}% (理想: 33.3%)")
        severity.append('WARNING')
        root_causes.extend([
            "報酬関数の微調整不足"
        ])
        recommendations.extend([
            "1. profit_bonus_multipliersを調整:",
            "   - [1.0, 0.9, 1.0] (BUY=0.9)"
        ])
    
    # === 不均衡比率の診断 ===
    max_rate = max(hold, buy, sell)
    min_rate = min(hold, buy, sell)
    imbalance_ratio = max_rate / (min_rate + 1e-6)
    
    if imbalance_ratio > 10.0:
        issues.append(f"🔴 極端なアクション不均衡: {imbalance_ratio:.1f}倍の差")
        severity.append('CRITICAL')
        root_causes.extend([
            "根本的な報酬設計の問題",
            "環境設定の深刻なミス"
        ])
        recommendations.extend([
            "1. カリキュラム学習を強制:",
            "   - curriculum_stage: 'forced_balance'",
            "2. 層別サンプリングを有効化:",
            "   - enable_stratified_sampling: true",
            "3. 全てのdiversity enforcement機能を有効化:",
            "   - enable_forced_diversity: true",
            "   - enable_pan: true",
            "   - enable_probes: true",
            "   - enable_lagrange: true"
        ])
    elif imbalance_ratio > 5.0:
        issues.append(f"🟡 大きなアクション不均衡: {imbalance_ratio:.1f}倍の差")
        severity.append('WARNING')
    
    # === エントロピーの推定 ===
    # H = -Σ p_i * log(p_i)
    import math
    entropy = 0.0
    for rate in [hold, buy, sell]:
        if rate > 1e-10:
            entropy -= rate * math.log(rate)
    
    max_entropy = math.log(3)  # ≈ 1.099
    entropy_ratio = entropy / max_entropy
    
    print(f"  エントロピー: {entropy:.4f} / {max_entropy:.4f} ({entropy_ratio*100:.1f}%)")
    
    if entropy_ratio < 0.7:
        issues.append(f"🟡 低いアクションエントロピー: {entropy:.3f} (最大: {max_entropy:.3f})")
        root_causes.append("探索が不足している")
        recommendations.append("エントロピー係数を増加: ent_coef → 0.2以上")
    
    return {
        'distribution': dist,
        'issues': issues,
        'severity': severity,
        'root_causes': root_causes,
        'recommendations': recommendations,
        'entropy': entropy,
        'entropy_ratio': entropy_ratio,
        'imbalance_ratio': imbalance_ratio
    }


def print_diagnosis(diagnosis: Dict[str, any]):
    """診断結果を表示"""
    if not diagnosis['issues']:
        print("✅ アクション分布は概ね均衡しています！")
        return
    
    print("="*70)
    print("⚠️ 検出された問題:")
    print("="*70)
    for i, (issue, sev) in enumerate(zip(diagnosis['issues'], diagnosis['severity']), 1):
        print(f"{i}. {issue} [{sev}]")
    
    print("\n" + "="*70)
    print("🔍 考えられる根本原因:")
    print("="*70)
    for i, cause in enumerate(diagnosis['root_causes'], 1):
        print(f"{i}. {cause}")
    
    print("\n" + "="*70)
    print("🔧 推奨される対処法:")
    print("="*70)
    for rec in diagnosis['recommendations']:
        print(f"  {rec}")


def generate_config_fix(diagnosis: Dict[str, any]) -> Dict[str, any]:
    """診断結果から設定修正案を生成"""
    dist = diagnosis['distribution']
    
    fix = {
        "_comment": "アクション分布問題の修正案",
        "_current_distribution": {
            "hold": f"{dist['hold']*100:.2f}%",
            "buy": f"{dist['buy']*100:.2f}%",
            "sell": f"{dist['sell']*100:.2f}%"
        }
    }
    
    # SELL bias対策
    if dist['sell'] > 0.45:  # 45%以上
        fix.update({
            "lagrange_eta": 0.1,
            "lagrange_lambda_max": 3.0,
            "enable_pan": True,
            "enable_probes": True,
            "ent_coef": 0.2,
            "profit_bonus_multipliers": [1.0, 1.0, 0.8]
        })
    elif dist['sell'] > 0.40:
        fix.update({
            "lagrange_eta": 0.07,
            "lagrange_lambda_max": 2.5,
            "enable_pan": True,
            "ent_coef": 0.15
        })
    
    # HOLD bias対策
    if dist['hold'] > 0.50:
        fix.update({
            "ent_coef": 0.3,
            "action_penalty_scale": 0.005,
            "trade_frequency_penalty": 0.005
        })
    elif dist['hold'] > 0.45:
        fix.update({
            "ent_coef": 0.15,
            "action_penalty_scale": 0.008
        })
    
    # BUY bias対策
    if dist['buy'] > 0.45:
        fix.update({
            "profit_bonus_multipliers": [1.0, 0.8, 1.0],
            "curriculum_stage": "forced_balance",
            "enable_stratified_sampling": True
        })
    
    # 極端な不均衡
    if diagnosis['imbalance_ratio'] > 10.0:
        fix.update({
            "curriculum_stage": "forced_balance",
            "enable_stratified_sampling": True,
            "enable_forced_diversity": True,
            "enable_pan": True,
            "enable_lagrange": True,
            "ent_coef": 0.3
        })
    
    return fix


def main():
    """メイン実行"""
    print_header()
    
    # アクション分布の収集
    dist = collect_action_distribution()
    if dist is None:
        return
    
    # 診断実行
    diagnosis = diagnose_distribution(dist)
    
    # 結果表示
    print_diagnosis(diagnosis)
    
    # 設定修正案を生成
    if diagnosis['issues']:
        print("\n" + "="*70)
        print("📝 推奨設定変更（ppo_balanced_mem_optimized.jsonに適用）:")
        print("="*70)
        
        fix = generate_config_fix(diagnosis)
        print(json.dumps(fix, indent=2, ensure_ascii=False))
        
        # 保存
        output_file = Path("action_distribution_diagnosis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'diagnosis': {
                    'distribution': diagnosis['distribution'],
                    'issues': diagnosis['issues'],
                    'root_causes': diagnosis['root_causes'],
                    'entropy': diagnosis['entropy'],
                    'imbalance_ratio': diagnosis['imbalance_ratio']
                },
                'recommended_fix': fix
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 診断結果を保存しました: {output_file}")
    
    print("\n" + "="*70)
    print("追加情報の確認事項:")
    print("="*70)
    print("1. TensorBoardで詳細なメトリクスを確認:")
    print("   tensorboard --logdir tensorboard")
    print()
    print("2. 学習ログで以下を確認:")
    print("   - Lagrange λの推移")
    print("   - SELL rate errorの推移")
    print("   - Advantageの各アクション別値")
    print()
    print("3. 設定ファイルで現在の値を確認:")
    print("   - lagrange_eta, lagrange_lambda_max")
    print("   - enable_pan, enable_probes")
    print("   - ent_coef")
    print("   - profit_bonus_multipliers")


if __name__ == "__main__":
    main()
