# Balance Penalty Fix - Asymmetric Targets Solution

## 問題の発見

2000ステップ訓練後でも SELL: 66.6% のまま改善なし。

**根本原因**: 前の修正でも対称性が残っていた
```python
# 旧修正（まだ対称）:
total_deviation = deviation_buy + deviation_sell + deviation_hold + buy_sell_imbalance * 0.5

# 計算結果:
# ALL_SELL: 1.333 + 0.5 = 1.833
# ALL_BUY:  1.333 + 0.5 = 1.833  ← 同じペナルティ！
```

## 真の解決策: 非対称ターゲット比率

**新実装（ztb/trading/environment/components/reward_calculator.py lines 240-268）**:

```python
# 非対称なターゲット: BUY 優遇, SELL 抑制
buy_target = 0.4      # BUY を 40% を目標
sell_target = 0.25    # SELL を 25% に抑制
hold_target = 0.35    # HOLD は 35%

# ペナルティ計算:
total_deviation = |buy_ratio - 0.4| + |sell_ratio - 0.25| + |hold_ratio - 0.35|
balance_penalty = total_deviation * balance_penalty_scale (200.0)
```

### ペナルティの非対称性

| シナリオ | 計算 | ペナルティ |
|---------|------|-----------|
| **ALL_SELL** | \|0-0.4\| + \|1-0.25\| + \|0-0.35\| = 0.4 + 0.75 + 0.35 = 1.5 | **1.5 × 200 = 300.0** |
| **ALL_BUY** | \|1-0.4\| + \|0-0.25\| + \|0-0.35\| = 0.6 + 0.25 + 0.35 = 1.2 | **1.2 × 200 = 240.0** |
| **理想（目標）** | \|0.4-0.4\| + \|0.25-0.25\| + \|0.35-0.35\| = 0.0 | **0.0** |

**ペナルティ差: 60.0** → ALL_SELL は ALL_BUY より 60 だけ高いコストを支払う

## 効果

1. **非対称性の確立**: ALL_SELL と ALL_BUY で異なるペナルティ
2. **BUY のインセンティブ**: BUY を選ぶことでペナルティが 60 低下
3. **SELL の抑止**: SELL を選ぶことでペナルティが 60 高上昇
4. **アクション ボーナスとの協力**:
   - BUY アクション ボーナス: +10.0 → ボーナス + ペナルティ軽減 = 強力
   - SELL アクション ボーナス: +5.0 → 弱いボーナスなのに高ペナルティ = 選ばない

## 実装ファイル

- **ztb/trading/environment/components/reward_calculator.py** (lines 240-268)
  - 非対称ターゲット比率の実装
  - ログ出力で `targets=[BUY:0.400, SELL:0.250, HOLD:0.350]` を表示

## テスト対象

設定ファイル: `config/sac_v444_asymmetric_targets_2k_test.json`

2000ステップ訓練で以下を検証:
- SELL が 66.6% から低下するか
- BUY が増加するか
- HOLD の変化

期待結果: BUY > SELL （逆転）

