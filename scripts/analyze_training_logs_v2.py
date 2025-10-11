"""
学習ログ分析ツール v2 - アクション分布問題の詳細診断

学習ログファイルから自動的にアクション分布を抽出し、詳細な分析を行います。
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Any
import json
import math


class SellRateInfo(TypedDict, total=False):
    """SELL rate and constraint information extracted from logs."""
    sell_rate: float
    lambda_value: float
    constraint_active: bool
    timestamp: str


class SellAvoidanceDiagnosis(TypedDict, total=False):
    """Detailed diagnosis of SELL avoidance problems."""
    sell_rate: float
    lambda_val: float
    constraint_active: bool
    estimated_distribution: Dict[str, float]
    entropy: float
    issues: List[str]
    root_causes: List[str]
    recommendations: List[str]
    severity: str


class EmergencyConfig(TypedDict, total=False):
    """Emergency configuration for fixing SELL avoidance."""
    _comment: str
    _problem: str
    _section_1: str
    lagrange_r_target: float
    lagrange_tolerance: float
    lagrange_eta: float
    lagrange_lambda_max: float
    lagrange_warmup_steps: int
    _section_2: str
    reward_settings: Dict[str, Any]
    _section_3: str
    ent_coef: float
    enable_forced_diversity: bool
    enable_stratified_sampling: bool
    curriculum_stage: str
    _section_4: str
    enable_pan: bool
    enable_probes: bool
    enable_lagrange: bool
    _validation: Dict[str, str]


def find_latest_log_file() -> Optional[Path]:
    """最新のログファイルを見つける"""
    log_patterns = [
        "training*.log",
        "*.log",
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(Path(".").glob(pattern))
    
    if not log_files:
        return None
    
    # 最新のファイルを取得
    latest = max(log_files, key=lambda f: f.stat().st_mtime)
    return latest


def parse_sell_rate_from_text(text: str) -> Optional[SellRateInfo]:
    """テキストからSELL rate情報を抽出"""
    results: SellRateInfo = {}
    
    # SELL Rate (avg): 1.6% のパターン
    sell_rate_match = re.search(r'SELL Rate \(avg\):\s*([\d.]+)%', text)
    if sell_rate_match:
        results['sell_rate'] = float(sell_rate_match.group(1)) / 100.0
    
    # Lambda (final): 2.000000 のパターン
    lambda_match = re.search(r'Lambda \(final\):\s*([\d.]+)', text)
    if lambda_match:
        results['lambda_value'] = float(lambda_match.group(1))
    
    # Constraint Active: True のパターン
    constraint_match = re.search(r'Constraint Active:\s*(\w+)', text)
    if constraint_match:
        results['constraint_active'] = constraint_match.group(1) == 'True'
    
    return results if results else None


def estimate_action_distribution(sell_rate: float) -> Dict[str, float]:
    """
    SELL rateから全体のアクション分布を推定
    
    SELL率が極端に低い場合、HOLDとBUYで残りを分配していると仮定
    """
    # 残りをHOLDとBUYで分配（おそらくHOLD偏重）
    remaining = 1.0 - sell_rate
    
    # 推定ロジック:
    # SELL率が低い場合、通常HOLDが多くなる傾向がある
    # BUYとSELLのバランスが取れていないと、HOLDで待機するしかない
    
    # 仮定: 残りの70%がHOLD、30%がBUY
    hold_rate = remaining * 0.7
    buy_rate = remaining * 0.3
    
    return {
        'hold': hold_rate,
        'buy': buy_rate,
        'sell': sell_rate
    }


def diagnose_sell_avoidance(sell_rate: float, lambda_val: float, constraint_active: bool) -> SellAvoidanceDiagnosis:
    """SELL回避問題の詳細診断"""
    
    print("="*70)
    print("🔍 SELL回避問題の詳細診断")
    print("="*70)
    print(f"  SELL Rate: {sell_rate*100:.2f}% (目標: 33%)")
    print(f"  Lambda: {lambda_val:.6f}")
    print(f"  Constraint Active: {constraint_active}")
    print()
    
    issues = []
    root_causes = []
    recommendations = []
    severity = 'CRITICAL'
    
    # === 重症度判定 ===
    if sell_rate < 0.05:  # 5%未満
        issues.append(f"🔴 極めて深刻なSELL回避: {sell_rate*100:.2f}% (目標: 33%)")
        severity = 'CRITICAL'
        
        root_causes.extend([
            "❌ Lagrange制約が完全に機能していない",
            "❌ Lambdaが上限に張り付いている (2.0/2.0)",
            "❌ SELL時の報酬が著しくマイナス（大きなペナルティ）",
            "❌ Action maskingでSELLが過度にブロックされている可能性",
            "❌ 報酬関数の設計に根本的な問題がある"
        ])
        
    elif sell_rate < 0.15:  # 15%未満
        issues.append(f"🔴 深刻なSELL回避: {sell_rate*100:.2f}% (目標: 33%)")
        severity = 'CRITICAL'
        
        root_causes.extend([
            "⚠️ Lagrange制約が弱すぎる",
            "⚠️ Lambda上限が低すぎる",
            "⚠️ SELL時のペナルティが大きい"
        ])
    
    # === Lambda分析 ===
    if lambda_val >= 1.9:  # 上限近く
        issues.append(f"🔴 Lagrange制約が上限に到達: λ={lambda_val:.3f}")
        root_causes.append("⚠️ Lambda上限 (2.0) が低すぎる - 制約が飽和している")
        recommendations.append("1. Lambda上限を大幅に引き上げる: lambda_max 2.0 → 10.0 以上")
    
    # === 推定アクション分布 ===
    estimated_dist = estimate_action_distribution(sell_rate)
    print("📊 推定アクション分布:")
    print(f"  HOLD: {estimated_dist['hold']*100:.2f}% (推定)")
    print(f"  BUY:  {estimated_dist['buy']*100:.2f}% (推定)")
    print(f"  SELL: {estimated_dist['sell']*100:.2f}% (実測)")
    print()
    
    # === エントロピー計算 ===
    entropy = 0.0
    for rate in estimated_dist.values():
        if rate > 1e-10:
            entropy -= rate * math.log(rate)
    max_entropy = math.log(3)  # ≈ 1.099
    
    print(f"  推定エントロピー: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")
    print()
    
    if entropy < max_entropy * 0.5:
        issues.append(f"🔴 極めて低いアクションエントロピー: {entropy:.3f}")
        root_causes.append("❌ 探索が完全に停止している")
    
    # === 根本原因の詳細分析 ===
    print("="*70)
    print("🔍 根本原因の詳細分析:")
    print("="*70)
    
    # 原因1: Lagrange制約の設計問題
    print("\n【原因1】Lagrange制約の設計問題")
    print(f"  - 現在のλ上限: 2.0 → 完全に飽和")
    print(f"  - 現在のη（学習率）: 0.05 → 遅すぎる可能性")
    print(f"  - 制約の強さが不十分でSELL誘導できていない")
    
    # 原因2: 報酬関数の問題
    print("\n【原因2】報酬関数の問題")
    print("  SELLアクション時に以下のペナルティが累積している可能性:")
    print("  - action_penalty_scale: 0.01")
    print("  - trade_frequency_penalty: 0.01")
    print("  - consecutive_trade_penalty: 0.05")
    print("  - transaction_cost: 0.001")
    print("  → これらが合わさってSELLが不利になっている")
    
    # 原因3: Action masking
    print("\n【原因3】Action Masking の影響")
    print("  - min_holding_period: 設定値不明（デフォルト5?）")
    print("  - ポジション保持期間中はSELLがブロックされる")
    print("  - 過度に制限的な可能性")
    
    # 原因4: データセット
    print("\n【原因4】データセット/環境の問題")
    print("  - データセットの価格変動パターンがSELL不利？")
    print("  - 報酬計算でSELL時のPnLが常にマイナス？")
    
    # === 推奨事項 ===
    print("\n" + "="*70)
    print("🔧 推奨される対処法（優先度順）:")
    print("="*70)
    
    recommendations.extend([
        "",
        "【最優先】Lagrange制約の大幅強化:",
        "  1. Lambda上限を10倍に: lagrange_lambda_max: 2.0 → 20.0",
        "  2. 学習率を倍増: lagrange_eta: 0.05 → 0.1",
        "  3. ターゲットSELL率を明示: lagrange_r_target: 0.33",
        "",
        "【高優先】報酬関数の調整:",
        "  4. SELLボーナスを追加: profit_bonus_multipliers: [1.0, 1.0, 2.0]",
        "     （SELL時の報酬を2倍にしてインセンティブ強化）",
        "  5. アクションペナルティを削減: action_penalty_scale: 0.01 → 0.001",
        "  6. 取引ペナルティを無効化（一時的）:",
        "     - trade_frequency_penalty: 0.01 → 0.0",
        "     - consecutive_trade_penalty: 0.05 → 0.0",
        "",
        "【中優先】多様性強制の追加:",
        "  7. エントロピー係数を大幅増加: ent_coef: 0.1 → 0.5",
        "  8. カリキュラム学習を強制: curriculum_stage: 'forced_balance'",
        "  9. 層別サンプリング有効化: enable_stratified_sampling: true",
        "",
        "【調査】環境設定の確認:",
        "  10. min_holding_periodを確認・削減（5 → 1）",
        "  11. データセットの価格変動パターンを確認",
        "  12. SELL時の実際の報酬値をデバッグ出力で確認"
    ])
    
    for rec in recommendations:
        print(rec)
    
    return {
        'sell_rate': sell_rate,
        'lambda_val': lambda_val,
        'constraint_active': constraint_active,
        'estimated_distribution': estimated_dist,
        'entropy': entropy,
        'issues': issues,
        'root_causes': root_causes,
        'recommendations': recommendations,
        'severity': severity
    }


def generate_emergency_config() -> EmergencyConfig:
    """緊急修正用の設定を生成"""
    return {
        "_comment": "🚨 SELL回避問題の緊急修正設定",
        "_problem": "SELL rate 1.6% → 極めて深刻なSELL回避",
        
        "_section_1": "=== Lagrange制約の大幅強化 ===",
        "lagrange_r_target": 0.33,
        "lagrange_tolerance": 0.05,
        "lagrange_eta": 0.1,
        "lagrange_lambda_max": 20.0,
        "lagrange_warmup_steps": 500,
        
        "_section_2": "=== 報酬関数の調整（SELLインセンティブ強化） ===",
        "reward_settings": {
            "profit_bonus_multipliers": [1.0, 1.0, 2.0],
            "action_penalty_scale": 0.001,
            "trade_frequency_penalty": 0.0,
            "consecutive_trade_penalty": 0.0,
            "trade_cooldown_penalty": 0.0
        },
        
        "_section_3": "=== 多様性の強制 ===",
        "ent_coef": 0.5,
        "enable_forced_diversity": True,
        "enable_stratified_sampling": True,
        "curriculum_stage": "forced_balance",
        
        "_section_4": "=== その他の設定 ===",
        "enable_pan": True,
        "enable_probes": True,
        "enable_lagrange": True,
        
        "_validation": {
            "expected_sell_rate": ">15% in 10k steps",
            "expected_lambda": "<10.0 (not saturated)",
            "expected_entropy": ">0.8"
        }
    }


def main():
    """メイン実行"""
    print("="*70)
    print("📊 学習ログ分析ツール v2 - SELL回避問題診断")
    print("="*70)
    print()
    
    # コマンドライン引数またはログテキストから情報を取得
    if len(sys.argv) > 1:
        # ファイルパスが指定された場合
        log_file = Path(sys.argv[1])
        if not log_file.exists():
            print(f"❌ ログファイルが見つかりません: {log_file}")
            return
        log_text = log_file.read_text(encoding='utf-8')
    else:
        # 標準入力から読み取る（ログテキストを貼り付け）
        print("ログテキストを貼り付けてください（Ctrl+Dで終了）:")
        print("-" * 70)
        log_text = sys.stdin.read()
    
    # ログから情報を抽出
    info = parse_sell_rate_from_text(log_text)
    
    if not info or 'sell_rate' not in info:
        print("❌ ログからSELL rate情報を抽出できませんでした")
        print()
        print("期待されるログフォーマット:")
        print("  SELL Rate (avg): X.X%")
        print("  Lambda (final): X.XXXXXX")
        print("  Constraint Active: True/False")
        return
    
    print("✅ ログ解析成功:")
    print(f"  - SELL Rate: {info['sell_rate']*100:.2f}%")
    print(f"  - Lambda: {info.get('lambda', 'N/A')}")
    print(f"  - Constraint Active: {info.get('constraint_active', 'N/A')}")
    print()
    
    # 詳細診断
    diagnosis = diagnose_sell_avoidance(
        info['sell_rate'],
        info.get('lambda', 0.0),
        info.get('constraint_active', False)
    )
    
    # 緊急修正設定を生成
    print("\n" + "="*70)
    print("📝 緊急修正設定（ppo_balanced_mem_optimized.jsonに適用）:")
    print("="*70)
    
    emergency_config = generate_emergency_config()
    print(json.dumps(emergency_config, indent=2, ensure_ascii=False))
    
    # 結果を保存
    output_file = Path("sell_avoidance_diagnosis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'log_info': info,
            'diagnosis': {
                'sell_rate': diagnosis.get('sell_rate'),
                'lambda': diagnosis.get('lambda_val'),
                'estimated_distribution': diagnosis.get('estimated_distribution'),
                'entropy': diagnosis.get('entropy'),
                'issues': diagnosis.get('issues'),
                'root_causes': diagnosis.get('root_causes'),
                'severity': diagnosis.get('severity')
            },
            'emergency_config': emergency_config,
            'recommendations': diagnosis.get('recommendations')
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 診断結果を保存しました: {output_file}")
    
    # 次のステップ
    print("\n" + "="*70)
    print("📋 次のステップ:")
    print("="*70)
    print("1. 上記の緊急修正設定をppo_balanced_mem_optimized.jsonに適用")
    print("2. 短い学習セッション（5000-10000 steps）で効果を確認")
    print("3. SELL rateが15%以上に改善されたか確認")
    print("4. Lambda値が上限に張り付いていないか確認")
    print()
    print("デバッグ出力の追加も推奨:")
    print("- SELL時の実際の報酬値")
    print("- アクション別のadvantage値")
    print("- min_holding_periodによるブロック頻度")


if __name__ == "__main__":
    main()
