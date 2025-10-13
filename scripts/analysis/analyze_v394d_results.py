"""
v394d訓練結果の詳細分析
100,352 timestepsで完了したが、HOLD 89%に収束
"""

import json


def analyze_v394d_results() -> None:
    """v394d訓練結果の詳細分析"""
    
    print("="*80)
    print("v394d (Aggressive) Training Analysis")
    print("="*80)
    print()
    
    # 訓練設定
    print("📋 Training Configuration:")
    print("  - Total timesteps: 100,352 ✅")
    print("  - Learning rate: 0.007503")
    print("  - Batch size: 256")
    print("  - ent_coef: 0.01")
    print()
    
    # 報酬設定
    print("💰 Reward Settings (Aggressive):")
    print("  - HOLD penalty: 0.1 (5x)")
    print("  - Consecutive HOLD penalty: 0.05 (5x)")
    print("  - Successful trade bonus: 5.0 (5x)")
    print("  - Profit multiplier: 10.0 (2x)")
    print("  - Trading frequency bonus: 0.3 (2x)")
    print()
    
    # Action分布の推移
    print("📊 Action Distribution Over Time:")
    print()
    data = [
        (2048, 128, 67, 61),
        (5120, 191, 34, 31),
        (10240, 232, 11, 13),
        (95232, 240, 9, 7),
        (96256, 232, 10, 14),
        (97280, 229, 15, 12),
        (98304, 235, 12, 9),
        (99328, 234, 9, 13),
        (100352, 228, 12, 16),
    ]
    
    print(f"{'Timesteps':>10} | {'HOLD':>4} | {'BUY':>4} | {'SELL':>4} | {'HOLD%':>6} | {'BUY+SELL%':>10} | Entropy")
    print("-" * 80)
    
    for timesteps, hold, buy, sell in data:
        total = hold + buy + sell
        hold_pct = (hold / total) * 100
        trade_pct = ((buy + sell) / total) * 100
        
        # エントロピー推定（簡易）
        if timesteps >= 95232:
            entropy = "0.61"
        elif timesteps >= 10240:
            entropy = "~0.65"
        elif timesteps >= 5120:
            entropy = "~0.90"
        else:
            entropy = "~1.00"
        
        icon = "✅" if hold_pct < 60 else "⚠️" if hold_pct < 80 else "🚨"
        print(f"{timesteps:>10,} | {hold:>4} | {buy:>4} | {sell:>4} | {hold_pct:>5.1f}% | {trade_pct:>9.1f}% | {entropy:>7} {icon}")
    
    print()
    print("="*80)
    print("🔍 Key Findings:")
    print("="*80)
    print()
    
    print("1. ✅ 初期改善は顕著")
    print("   - 2,048 steps: HOLD 50% (128/256)")
    print("   - これは全v394シリーズで最良")
    print()
    
    print("2. 🚨 しかし急速に悪化")
    print("   - 5,120 steps: HOLD 75% (191/256)")
    print("   - 10,240 steps: HOLD 91% (232/256)")
    print("   - 95,232 steps: HOLD 94% (240/256)")
    print("   - 100,352 steps: HOLD 89% (228/256) ← 最終")
    print()
    
    print("3. ⚠️ エントロピー減少")
    print("   - 初期: ~1.00-1.07")
    print("   - 最終: 0.61")
    print("   - ent_coef=0.01では不十分")
    print()
    
    print("4. 🎯 報酬シェーピングの限界")
    print("   - HOLD罰則5倍でも効果は一時的")
    print("   - 取引報酬5倍でも不十分")
    print("   - 根本的なアプローチが必要")
    print()
    
    print("="*80)
    print("💡 Next Steps:")
    print("="*80)
    print()
    
    print("Strategy A: 超高エントロピー版（v394f）")
    print("  - ent_coef: 0.01 → 0.2 (20倍)")
    print("  - 探索を強制的に維持")
    print("  - HOLD収束を最大限抑制")
    print()
    
    print("Strategy B: Stochastic推論での評価")
    print("  - deterministic=False で推論")
    print("  - 初期Action分布（HOLD 50%）が使えるか確認")
    print("  - early stopping（2,048 steps時点）も検討")
    print()
    
    print("Strategy C: カリキュラム学習")
    print("  - 段階的にHOLD罰則を強化")
    print("  - エントロピーを動的に調整")
    print()
    
    print("Strategy D: 異なるアプローチ")
    print("  - Soft Actor-Critic (SAC) を試す")
    print("  - 最大エントロピーRLで探索維持")
    print()
    
    print("="*80)
    print("🎬 Immediate Action:")
    print("  1. v394f作成（ent_coef=0.2）")
    print("  2. v394d Stochasticバックテスト（deterministic=False）")
    print("  3. 結果比較と最終判断")
    print("="*80)


if __name__ == "__main__":
    analyze_v394d_results()
