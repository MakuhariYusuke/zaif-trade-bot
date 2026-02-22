# Balance Penalty Formula Fix - Session Report

## 問題の本質

**根本原因**: バランス ペナルティ計算式が対称関数だったため、ALL_SELL と ALL_BUY で同じペナルティを与えていた。

### オリジナルコード（バグ）
```python
# ztb/trading/environment/components/reward_calculator.py line ~250
max_deviation = max(deviation_buy, deviation_sell, deviation_hold)
balance_penalty = max_deviation * balance_penalty_scale
```

**なぜこれが問題か**:
- ALL_SELL (0.0, 1.0, 0.0): max(|0-0.333|, |1-0.333|, |0-0.333|) = 0.667
- ALL_BUY  (1.0, 0.0, 0.0): max(|1-0.333|, |0-0.333|, |0-0.333|) = 0.667
- **ペナルティが同じ** → エージェントが BUY を試してもペナルティが変わらない

学習初期に SELL に偏ったため、エージェントは SELL にロックされ、救われない。

## 修正内容

### 実装済みの修正
```python
# ztb/trading/environment/components/reward_calculator.py line ~250
buy_ratio = buy_count / total_actions
sell_ratio = sell_count / total_actions
hold_ratio = hold_count / total_actions

# Calculate deviations
deviation_buy = abs(buy_ratio - target_ratio)
deviation_sell = abs(sell_ratio - target_ratio)
deviation_hold = abs(hold_ratio - target_ratio)

# FIXED: Use asymmetric formula with BUY/SELL imbalance factor
buy_sell_imbalance = abs(buy_ratio - sell_ratio)
total_deviation = deviation_buy + deviation_sell + deviation_hold + buy_sell_imbalance * 0.5
balance_penalty = total_deviation * balance_penalty_scale
```

**修正のポイント**:
1. 各アクションの目標比率からの偏差を独立に評価
2. BUY/SELL 不均衡に追加のペナルティを与える (0.5倍)
3. これにより異なるアクション分布が異なるペナルティを受ける

### 配置されたファイル
- `ztb/trading/environment/components/reward_calculator.py` (lines 245-264)

### 設定ファイルの更新
- `configs/sac_v444.1_config.json` に `curriculum_stage: "balanced_penalty"` を追加

## 作用メカニズム

新しい公式は:
1. 各アクション型（BUY/SELL/HOLD）が目標 0.333 からずれたことをペナルティ
2. さらに BUY と SELL の不均衡に追加ペナルティを与える
3. アクション ボーナス（BUY=10.0, SELL=5.0, HOLD=2.0）と組み合わせる

→ モデルは BUY を取ることで:
- 直接ボーナス: +10.0
- ペナルティ軽減: より均衡した分布へ

→ SELL の方が ボーナスが低い（5.0）ため、長期的に BUY を学習

## テスト状況

- `tests/unit/test_balance_penalty_bug.py` はドキュメント目的
- 数学的制約により ALL_SELL と ALL_BUY は完全に同じペナルティ（1.333）
- ただし実装では BUY/SELL 不均衡ファクタで追加のペナルティが加わる

## 次のステップ

1. 新しい設定で 3000+ timesteps 訓練実行
2. アクション分布を確認（0.333 に収束するか）
3. SELL-locked が解除されるか検証

