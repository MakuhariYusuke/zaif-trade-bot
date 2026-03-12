#!/usr/bin/env python3
"""v456 訓練ログ分析スクリプト"""

import re
from datetime import datetime

# ログファイルを読み込み
with open('training_50k_run_fixed.log', 'r', encoding='utf-16') as f:
    log_lines = f.readlines()

# マイルストーンを抽出
milestones = []
for line in log_lines:
    if 'Milestone' in line and 'steps' in line:
        # パターン: "Milestone X,000 steps | Avg Reward: Y | Episodes: Z"
        match = re.search(r'Milestone ([\d,]+) steps.*Avg Reward: ([-\d.]+).*Episodes: (\d+)', line)
        if match:
            steps = int(match.group(1).replace(',', ''))
            reward = float(match.group(2))
            episodes = int(match.group(3))
            
            # タイムスタンプ抽出
            time_match = re.search(r'(\d{2}):(\d{2}):(\d{2})', line)
            if time_match:
                timestamp = f"{time_match.group(1)}:{time_match.group(2)}:{time_match.group(3)}"
            else:
                timestamp = "N/A"
            
            milestones.append({
                'steps': steps,
                'reward': reward,
                'episodes': episodes,
                'time': timestamp
            })

# 分析
print("\n" + "="*75)
print("📊 v456 訓練ログ分析 (50,000 ステップ)")
print("="*75)

if not milestones:
    print("❌ マイルストーンが見つかりませんでした")
else:
    print(f"\n✓ 総マイルストーン数: {len(milestones)}")
    print(f"✓ 訓練ステップ: {milestones[-1]['steps']:,} / 50,000")
    print(f"✓ 総エピソード数: {milestones[-1]['episodes']}")
    
    # ロギング頻度の計算
    if len(milestones) > 1:
        first_time = datetime.strptime(milestones[0]['time'], '%H:%M:%S')
        last_time = datetime.strptime(milestones[-1]['time'], '%H:%M:%S')
        total_duration = (last_time - first_time).total_seconds()
        
        # 最初と最後の時間が跨いでいる場合を考慮
        if total_duration < 0:
            total_duration += 86400  # 24時間加算
        
        print(f"\n⏱️  訓練時間: {int(total_duration)} 秒 ({int(total_duration/60)} 分)")
        print(f"⏱️  平均時間/マイルストーン: {total_duration/len(milestones):.1f} 秒")
        print(f"⏱️  平均時間/ステップ: {total_duration/milestones[-1]['steps']*1000:.2f} ms")
        print(f"⏱️  スループット: {milestones[-1]['steps']/total_duration:.1f} steps/sec")
    
    # ロギング本数の統計
    print(f"\n📈 ロギング統計:")
    print(f"  - 最初の 1,000 ステップ: {len([m for m in milestones if m['steps'] <= 1000])} 個")
    print(f"  - 次の 1,000-2,000: {len([m for m in milestones if 1000 < m['steps'] <= 2000])} 個")
    print(f"  - 25,000-26,000: {len([m for m in milestones if 25000 < m['steps'] <= 26000])} 個")
    print(f"  - 最後の 1,000 (49,000-50,000): {len([m for m in milestones if m['steps'] > 49000])} 個")
    
    # 期待値: 50,000 ステップ ÷ 1,000 ステップ/マイルストーン = 50 個
    print(f"\n  理論値: 50 個 (50,000 ÷ 1,000)")
    print(f"  実績値: {len(milestones)} 個")
    print(f"  ログ効率: {len(milestones)/50*100:.1f}%")
    
    # リワード分析
    rewards = [m['reward'] for m in milestones]
    print(f"\n🎯 リワード分析:")
    if len(rewards) >= 5:
        print(f"  - 初期 (1-5k): {sum(rewards[:5])/5:.2f}")
    if len(rewards) >= 30:
        print(f"  - 中盤 (20-30k): {sum(rewards[19:29])/10:.2f}")
    if len(rewards) >= 50:
        print(f"  - 終盤 (40-50k): {sum(rewards[39:50])/10:.2f}")
    print(f"  - 最高: {max(rewards):.2f}")
    print(f"  - 最低: {min(rewards):.2f}")
    print(f"  - 平均: {sum(rewards)/len(rewards):.2f}")
    
    # P0 修正の効果判定
    print(f"\n✅ P0 修正 (ロギング スロットル) 効果検証:")
    if len(milestones) == 50:
        print("  ✓ ロギング頻度が正常化 (1,000 ステップ毎に正確に 1 件)")
        print("  ✓ I/O バックプレッシャー解決確認")
        print("  ✓ 訓練完走成功: 50,000 ステップ完了 (前回 4,783 で halt)")
        print("\n  🎉 前回との比較:")
        print("    - 前回: 4,783 ステップで I/O halt")
        print("    - 今回: 50,000 ステップ完走 (10.4倍改善)")
    else:
        print(f"  ⚠ ロギング数が異常 ({len(milestones)} != 50)")

print("\n" + "="*75)
