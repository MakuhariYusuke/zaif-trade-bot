# 🎯 重大発見：訓練時vs推論時の乖離の原因判明

## 📊 Stochastic vs Deterministic推論テスト結果

### Deterministic Mode (deterministic=True)
```
Total steps: 597
HOLD:   587 ( 98.3%)  ← quick_backtestと同じHOLD偏重！
BUY:      0 (  0.0%)
SELL:    10 (  1.7%)
```

### Stochastic Mode (deterministic=False)
```
Total steps: 523
HOLD:   267 ( 51.1%)  ← 訓練時（53%）と同じバランス！
BUY:    128 ( 24.5%)
SELL:   128 ( 24.5%)
```

## 🔴 原因特定

### **Deterministicモードが原因でした！**

quick_backtest.py、check_action_distribution.pyでは：
```python
action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
```

この`deterministic=True`により：
1. **確率分布から最大値のアクションのみ選択**
2. **Explorationが完全に無効化**
3. **HOLDの確率がわずかに高いと、常にHOLDを選択**

### 訓練時との違い

**訓練時（PPO rollout）**:
- Stochasticサンプリング
- Entropy bonusでexplorationを促進
- Target Entropy Controller（H*=0.769）が多様性を維持
- 結果: HOLD 53%, BUY 28%, SELL 19%

**推論時（deterministic=True）**:
- 最大確率アクションのみ
- Explorationなし
- わずかな確率差で決定
- 結果: HOLD 98.3%

## 🎯 解決策

### Option 1: Stochastic推論を使用
バックテストで`deterministic=False`を使用：
```python
action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
```

**メリット**:
- 訓練時と同じアクション分布
- 多様な取引パターン

**デメリット**:
- ランダム性により再現性が低い
- 実運用では一貫性が必要

### Option 2: Policy改善（推奨）
**問題の本質**: Deterministicモードで98% HOLDになるのは、Policyが適切に学習できていない証拠

**改善方向**:
1. **Value Function改善**:
   - explained_variance: 0.0 → 0.5以上を目指す
   - より良い状態価値推定

2. **Reward Shaping強化**:
   - HOLDペナルティ増加
   - 成功取引のボーナス増加
   - 多様性ボーナス強化

3. **Training改善**:
   - 長期訓練（1M timesteps）
   - 適切なlearning rate（現在0.0003、設定は0.007503）

## 📊 比較表

| 項目 | 訓練時 | Deterministic推論 | Stochastic推論 |
|------|--------|-------------------|----------------|
| HOLD | 53% | 98.3% | 51.1% |
| BUY | 28% | 0.0% | 24.5% |
| SELL | 19% | 1.7% | 24.5% |
| 一貫性 | - | 高い | 低い |
| 実用性 | - | 低い（HOLD偏重） | 中程度 |

## 🎯 次のステップ

### 短期（検証）
1. ✅ Stochastic推論テスト完了
2. quick_backtest.pyをstochasticモードで再評価
3. 収益性を確認

### 中期（改善）
1. ハイパーパラメータ適用バグ修正
2. v393訓練（learning_rate 0.007503等）
3. Reward設定調整（HOLD penalty強化）

### 長期（根本解決）
1. 1M timesteps訓練
2. Value Function改善
3. Deterministicモードでも良好なパフォーマンス

## 📝 結論

**v392自体は成功**:
- ✅ random_startバグ修正済み
- ✅ 訓練はバランス良好（HOLD 53%）
- ✅ Stochastic推論でも同様の分布

**問題はDeterministicモード**:
- ❌ 確率のわずかな差で常にHOLD選択
- ❌ Explorationなしで多様性ゼロ

**実運用の方向性**:
1. **Stochastic推論** + **閾値ベース意思決定**
2. **Policy改善訓練**でDeterministicモードでも良好な分布を実現
