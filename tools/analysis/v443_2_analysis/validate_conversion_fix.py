import json

import numpy as np""""""

報酬変換ロジックの修正効果検証スクリプト報酬変換ロジックの修正効果検証スクリプト

def old_conversion(reward):

    return (reward + 10) * 10""""""



def new_conversion(reward):

    return reward * 0.1

import numpy as npimport
import numpy as np

# バックテスト結果から報酬データを取得

with open('backtest_results/backtest_results_sac_v427_hybrid_20251026_063723.json', 'r') as f:import pandas as pd

    data = json.load(f)

def old_conversion(reward):import matplotlib.pyplot as plt

reward_min = data['reward_stats']['min']

reward_max = data['reward_stats']['max']    """旧変換ロジック"""from scipy import stats



print("=== 報酬変換ロジック修正効果分析 ===")    return (reward + 10) * 10

print(f"報酬範囲: [{reward_min:.2f}, {reward_max:.2f}]")

def old_conversion(reward):

# テスト報酬範囲を生成

test_rewards = np.linspace(reward_min, reward_max, 20)def new_conversion(reward):    """旧変換ロジック"""



old_changes = [old_conversion(r) for r in test_rewards]    """新変換ロジック"""    return (reward + 10) * 10

new_changes = [new_conversion(r) for r in test_rewards]

    return reward * 0.1

old_corr = np.corrcoef(test_rewards, old_changes)[0, 1]

new_corr = np.corrcoef(test_rewards, new_changes)[0, 1]def new_conversion(reward):



print("def analyze_conversion_effect():    """新変換ロジック"""

相関係数比較:")

print(f"  旧変換: {old_corr:.3f}")    """変換ロジックの効果を分析"""    return reward * 0.1

print(f"  新変換: {new_corr:.3f}")



print("

スケーリング範囲:")    # バックテスト結果から報酬データを取得def analyze_conversion_effect():

print(f"  旧変換: [{min(old_changes):.1f}, {max(old_changes):.1f}]")

print(f"  新変換: [{min(new_changes):.1f}, {max(new_changes):.1f}]")    try:    """変換ロジックの効果を分析"""



print("        with open('backtest_results/backtest_results_sac_v427_hybrid_20251026_063723.json', 'r') as f:

サンプル変換:")

for r in [-12.35, 0, 26.65]:            import json    # バックテスト結果から報酬データを取得

    old_pc = old_conversion(r)

    new_pc = new_conversion(r)            data = json.load(f)    try:

    print(f"Reward {r:6.2f}: 旧 {old_pc:7.1f} -> 新 {new_pc:5.1f}")
        with open('backtest_results/backtest_results_sac_v427_hybrid_20251026_063723.json', 'r') as f:

        reward_mean = data['reward_stats']['mean']            import json

        reward_std = data['reward_stats']['std']            data = json.load(f)

        reward_min = data['reward_stats']['min']

        reward_max = data['reward_stats']['max']        reward_mean = data['reward_stats']['mean']

        reward_std = data['reward_stats']['std']

        print("=== 報酬変換ロジック修正効果分析 ===")        reward_min = data['reward_stats']['min']

        print(f"報酬統計: 平均={reward_mean:.2f}, 標準偏差={reward_std:.2f}, 範囲=[{reward_min:.2f}, {reward_max:.2f}]")        reward_max = data['reward_stats']['max']



        # テスト報酬範囲を生成        print("=== 報酬変換ロジック修正効果分析 ===")

        test_rewards = np.linspace(reward_min, reward_max, 50)        print(f"報酬統計: 平均={reward_mean:.2f}, 標準偏差={reward_std:.2f}, 範囲=[{reward_min:.2f}, {reward_max:.2f}]")



        # 旧変換と新変換の比較        # テスト報酬範囲を生成

        old_changes = [old_conversion(r) for r in test_rewards]        test_rewards = np.linspace(reward_min, reward_max, 50)

        new_changes = [new_conversion(r) for r in test_rewards]

        # 旧変換と新変換の比較

        # 相関係数の計算（理論的な比較）        old_changes = [old_conversion(r) for r in test_rewards]

        old_corr = np.corrcoef(test_rewards, old_changes)[0, 1]        new_changes = [new_conversion(r) for r in test_rewards]

        new_corr = np.corrcoef(test_rewards, new_changes)[0, 1]

        # 相関係数の計算（理論的な比較）

        print("\n相関係数比較:")        # 理想的には、報酬とポートフォリオ変化が正の相関を持つべき

        print(f"  旧変換: {old_corr:.3f}")        old_corr = np.corrcoef(test_rewards, old_changes)[0, 1]

        print(f"  新変換: {new_corr:.3f}")        new_corr = np.corrcoef(test_rewards, new_changes)[0, 1]

        print(f"  改善度: {((new_corr - old_corr) / abs(old_corr) * 100):.1f}%")

        print("

        # スケーリング効果の比較相関係数比較:")

        print("\nスケーリング範囲:")        print(f"  旧変換: {old_corr:.3f}")

        print(f"  旧変換: [{min(old_changes):.1f}, {max(old_changes):.1f}]")        print(f"  新変換: {new_corr:.3f}")

        print(f"  新変換: [{min(new_changes):.1f}, {max(new_changes):.1f}]")        print(f"  改善度: {((new_corr - old_corr) / abs(old_corr) * 100):.1f}%")



        # ゼロクロスポイントの確認        # スケーリング効果の比較

        zero_cross_old = -10        print("

        zero_cross_new = 0スケーリング範囲:")

        print(f"  旧変換: [{min(old_changes):.1f}, {max(old_changes):.1f}]")

        print("\nゼロクロスポイント:")        print(f"  新変換: [{min(new_changes):.1f}, {max(new_changes):.1f}]")

        print(f"  旧変換: reward = {zero_cross_old} で portfolio_change = 0")

        print(f"  新変換: reward = {zero_cross_new} で portfolio_change = 0")        # ゼロクロスポイントの確認

        zero_cross_old = -10  # (0 + 10) * 10 = 0のとき reward = -10

        # 推奨事項        zero_cross_new = 0    # reward * 0.1 = 0のとき reward = 0

        print("\n=== 修正効果の評価 ===")

        if new_corr > old_corr:        print("

            print("✓ 相関係数が改善されました")ゼロクロスポイント:")

        else:        print(f"  旧変換: reward = {zero_cross_old} で portfolio_change = 0")

            print("✗ 相関係数が悪化しました")        print(f"  新変換: reward = {zero_cross_new} で portfolio_change = 0")



        if max(new_changes) - min(new_changes) < max(old_changes) - min(old_changes):        # 推奨事項

            print("✓ スケーリング範囲が適切になりました")        print("

        else:=== 修正効果の評価 ===")

            print("✗ スケーリング範囲に問題があります")        if new_corr > old_corr:

            print("✓ 相関係数が改善されました")

        print("\n=== 理論的根拠 ===")        else:

        print("1. Pendulum環境では報酬が高いほど良い行動")            print("✗ 相関係数が悪化しました")

        print("2. 新変換では報酬とポートフォリオ変化が線形関係")

        print("3. 小さなスケーリングにより過度な変動を防ぐ")        if max(new_changes) - min(new_changes) < max(old_changes) - min(old_changes):

        print("4. ゼロクロスが報酬0で自然")            print("✓ スケーリング範囲が適切になりました")

        else:

    except FileNotFoundError:            print("✗ スケーリング範囲に問題があります")

        print("バックテスト結果ファイルが見つかりません")

        # サンプルデータでデモ        print("\n=== 理論的根拠 ===")

        print("\nサンプルデータでのデモ:")        print("1. Pendulum環境では報酬が高いほど良い行動")

        sample_rewards = np.array([-16, -8, 0, 8, 16])        print("2. 新変換では報酬とポートフォリオ変化が線形関係")

        print("3. 小さなスケーリングにより過度な変動を防ぐ")

        for r in sample_rewards:        print("4. ゼロクロスが報酬0で自然")

            old_pc = old_conversion(r)

            new_pc = new_conversion(r)    except FileNotFoundError:

            print(f"Reward {r:4.0f}: 旧 {old_pc:6.1f} -> 新 {new_pc:5.1f}")        print("バックテスト結果ファイルが見つかりません")

        # サンプルデータでデモ

if __name__ == "__main__":        print("\nサンプルデータでのデモ:")

    analyze_conversion_effect()        sample_rewards = np.array([-16, -8, 0, 8, 16])

        for r in sample_rewards:
            old_pc = old_conversion(r)
            new_pc = new_conversion(r)
            print(f"Reward {r:4.0f}: 旧 {old_pc:6.1f} -> 新 {new_pc:5.1f}")

if __name__ == "__main__":
    analyze_conversion_effect()
