"""
v394d vs v394f 完全比較分析
ent_coef 20倍でも結果はほぼ同じという衝撃的な事実
"""

def analyze_v394_comparison() -> None:
    """v394d vs v394f の詳細比較"""
    
    print("="*80)
    print("🚨 v394d vs v394f 完全比較分析")
    print("="*80)
    print()
    
    # 設定比較
    print("📋 Configuration Comparison:")
    print()
    print(f"{'Setting':<30} | {'v394d':<15} | {'v394f':<15} | {'Difference'}")
    print("-" * 80)
    print(f"{'ent_coef':<30} | {'0.01':<15} | {'0.2':<15} | 20x 🔥")
    print(f"{'HOLD penalty':<30} | {'0.1':<15} | {'0.1':<15} | Same")
    print(f"{'Trade bonus':<30} | {'5.0':<15} | {'5.0':<15} | Same")
    print(f"{'Profit multiplier':<30} | {'10.0':<15} | {'10.0':<15} | Same")
    print()
    
    # 最終結果比較
    print("="*80)
    print("🔍 Final Results Comparison (100,352 timesteps)")
    print("="*80)
    print()
    
    v394d_data = {
        "HOLD": 228,
        "BUY": 12,
        "SELL": 16,
        "entropy": 0.610
    }
    
    v394f_data = {
        "HOLD": 228,
        "BUY": 15,
        "SELL": 13,
        "entropy": 0.597
    }
    
    total = 256
    
    print(f"{'Metric':<30} | {'v394d':<20} | {'v394f':<20} | {'Winner'}")
    print("-" * 80)
    
    for action in ["HOLD", "BUY", "SELL"]:
        d_count = v394d_data[action]
        f_count = v394f_data[action]
        d_pct = (d_count / total) * 100
        f_pct = (f_count / total) * 100
        
        if action == "HOLD":
            winner = "v394f ✅" if f_count < d_count else "Tie 🟰" if f_count == d_count else "v394d ✅"
        else:
            winner = "v394f ✅" if f_count > d_count else "Tie 🟰" if f_count == d_count else "v394d ✅"
        
        print(f"{action:<30} | {d_count:>3} ({d_pct:>5.1f}%){'':<8} | {f_count:>3} ({f_pct:>5.1f}%){'':<8} | {winner}")
    
    print(f"{'Entropy':<30} | {v394d_data['entropy']:<20.3f} | {v394f_data['entropy']:<20.3f} | {'Tie 🟰'}")
    print()
    
    # 時系列比較
    print("="*80)
    print("📊 Action Distribution Over Time")
    print("="*80)
    print()
    
    print("v394d (ent_coef=0.01):")
    print(f"{'Timesteps':>10} | {'HOLD':>4} | {'BUY':>4} | {'SELL':>4} | {'HOLD%':>6}")
    print("-" * 50)
    d_timeline = [
        (2048, 128, 67, 61),
        (10240, 232, 11, 13),
        (100352, 228, 12, 16),
    ]
    for ts, h, b, s in d_timeline:
        total = h + b + s
        print(f"{ts:>10,} | {h:>4} | {b:>4} | {s:>4} | {(h/total)*100:>5.1f}%")
    
    print()
    print("v394f (ent_coef=0.2):")
    print(f"{'Timesteps':>10} | {'HOLD':>4} | {'BUY':>4} | {'SELL':>4} | {'HOLD%':>6}")
    print("-" * 50)
    f_timeline = [
        # (2048, ?, ?, ?),  # データなし
        (94208, 234, 12, 10),
        (95232, 223, 17, 16),
        (98304, 223, 16, 17),
        (100352, 228, 15, 13),
    ]
    for ts, h, b, s in f_timeline:
        total = h + b + s
        print(f"{ts:>10,} | {h:>4} | {b:>4} | {s:>4} | {(h/total)*100:>5.1f}%")
    
    print()
    print("="*80)
    print("🚨 衝撃的な結論")
    print("="*80)
    print()
    
    print("1. ent_coef を20倍にしても結果はほぼ同じ")
    print("   - v394d: HOLD 89.1%, entropy 0.610")
    print("   - v394f: HOLD 89.1%, entropy 0.597")
    print("   - 差: ほぼゼロ！")
    print()
    
    print("2. エントロピー係数の効果が限定的")
    print("   - ent_coef 0.01 → 0.2 (20倍)")
    print("   - エントロピー 0.610 → 0.597 (ほぼ同じ)")
    print("   - HOLD比率 89.1% → 89.1% (完全同一)")
    print()
    
    print("3. 根本原因はent_coefではない")
    print("   - 報酬シグナルそのものが弱い")
    print("   - HOLDが「最適解」として学習されている")
    print("   - 取引コストがHOLD罰則を上回っている可能性")
    print()
    
    print("="*80)
    print("💡 次の戦略")
    print("="*80)
    print()
    
    print("Strategy A: Stochastic推論での評価 (最優先)")
    print("  - deterministic=False で両バージョンをテスト")
    print("  - 訓練時の多様性が実際に使えるか確認")
    print("  - 初期チェックポイント（2k steps）も評価")
    print()
    
    print("Strategy B: 報酬関数の根本的見直し")
    print("  - 取引コストを報酬計算から除外")
    print("  - HOLD罰則をさらに強化（0.5-1.0）")
    print("  - 成功取引報酬を10-20倍に")
    print()
    
    print("Strategy C: 異なるアルゴリズム")
    print("  - Soft Actor-Critic (SAC): 最大エントロピーRL")
    print("  - TD3: 連続行動空間で探索改善")
    print("  - A2C: より単純なベースライン")
    print()
    
    print("Strategy D: Early Stopping")
    print("  - 2,048 steps時点のモデルを使用")
    print("  - v394d: HOLD 50% (最良)")
    print("  - それ以上訓練すると悪化する")
    print()
    
    print("="*80)
    print("🎬 Immediate Actions:")
    print("="*80)
    print("  1. Stochasticバックテスト（v394d & v394f）")
    print("  2. 初期チェックポイント評価（2k steps）")
    print("  3. 報酬関数の抜本的見直し")
    print("  4. 代替アルゴリズム（SAC等）の検討")
    print("="*80)


if __name__ == "__main__":
    analyze_v394_comparison()
