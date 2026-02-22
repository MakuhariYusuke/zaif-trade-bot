# SAC v444 アクションバイアス改善 - 詳細デバッグガイド

## 📊 現状分析

### 根本原因の特定
現在の問題の本質:

```
Mean Reward: -9845 (95% がバランスペナルティから発生)
BUY Ratio:   18.00% (期待値: 30-40%)
SELL Ratio:  66.85% (期待値: 30-40%)
Penalty: abs(0.18 - 0.6685) * 1000 = 488.5 × 2000ステップ = 977,000 のペナルティ!
```

**結論**: balance_penalty_scale = 1000.0 が過度に大きく、毎ステップ -488 のペナルティが適用されている

---

## 🔧 改善戦略

### Phase 1: Balance Penalty スケール最適化

#### 設定 1: scale_200 (最小ペナルティ)
```json
{
  "balance_penalty": 200.0,
  "buy_bonus": 10.0,
  "sell_bonus": 5.0,
  "hold_bonus": 2.0
}
```

**計算**:
- 毎ステップペナルティ: 200 * 0.4885 ≈ -97.7 (前: -488.5)
- 期待される改善: Reward -5000～-2000 (前: -9845)
- **テスト目標**: 報酬が改善されるか確認

---

#### 設定 2: scale_300 (中程度ペナルティ)
```json
{
  "balance_penalty": 300.0,
  "buy_bonus": 15.0,
  "sell_bonus": 10.0,
  "hold_bonus": 3.0
}
```

**計算**:
- 毎ステップペナルティ: 300 * 0.4885 ≈ -146.6
- Action Bonuses を増強して多様性を促進
- **テスト目標**: バランスと規制のトレードオフを評価

---

#### 設定 3: scale_500 (より高いペナルティ)
```json
{
  "balance_penalty": 500.0,
  "buy_bonus": 20.0,
  "sell_bonus": 15.0,
  "hold_bonus": 5.0
}
```

**計算**:
- 毎ステップペナルティ: 500 * 0.4885 ≈ -244.3
- Action Bonuses を最大化して動機付け
- **テスト目標**: より厳密なバランス強制

---

## 📈 期待される改善

### Reward の改善軌跡

| Config | Balance Penalty | 毎Step ペナルティ | 期待される平均報酬 | 改善度 |
|--------|------------------|------------------|------------------|--------|
| Original | 1000.0 | -488.5 | -9845 | - |
| scale_200 | 200.0 | -97.7 | -5000～-2000 | 80% 改善 |
| scale_300 | 300.0 | -146.6 | -4000～-1500 | 85% 改善 |
| scale_500 | 500.0 | -244.3 | -3000～-500 | 90% 改善 |

### Action Distribution の改善

| Regime | 現在 (Original) | 目標 (scale_200+) |
|--------|-----------------|------------------|
| BUY | 18.00% | 30-40% |
| SELL | 66.85% | 30-40% |
| HOLD | 15.15% | 20-30% |

---

## 🎯 テスト実行ガイド

### Step 1: 最初の設定をテスト (scale_200)

```bash
python quick_train_v444_multi_config.py --config scale_200
```

**検証項目**:
- [ ] Mean Reward が -5000～-2000 の範囲に改善されたか
- [ ] BUY Ratio が 25-40% に改善されたか
- [ ] SELL Ratio が 25-40% に低下したか
- [ ] Training が安定しているか (Loss がスパイクしていないか)

**成功条件**:
- Mean Reward > -5000 (つまり、改善幅 > 4845)
- BUY Ratio > 25%
- SELL Ratio < 50%

---

### Step 2: 第2の設定をテスト (scale_300)

```bash
python quick_train_v444_multi_config.py --config scale_300
```

**検証項目**:
- [ ] Mean Reward が -4000～-1500 に改善されたか
- [ ] BUY Ratio が 30-45% に改善されたか
- [ ] scale_200 との比較で、さらに改善しているか

---

### Step 3: 第3の設定をテスト (scale_500)

```bash
python quick_train_v444_multi_config.py --config scale_500
```

**検証項目**:
- [ ] Mean Reward が -3000～-500 に改善されたか
- [ ] BUY Ratio が 35-50% に改善されたか
- [ ] 過度なペナルティで Training が不安定になっていないか

---

### Step 4: 比較分析

```bash
python analysis/parameter_tuning_analysis.py
```

**出力**:
- 各設定の視覚的比較
- 推奨される最適設定
- 次のステップの指針

---

## 🔍 詳細なデバッグポイント

### A. Continuous Action Distribution の分析

**現状**: Mean = -0.4968 (SELL 方向に強いバイアス)

**原因の仮説**:
1. Reward Function が SELL を無条件に促進している
2. Feature Distribution が SELL バイアスを反映している
3. Regime-specific targets が不適切に設定されている

**確認方法**:
```python
# training ログから以下を確認
- continuous_action_mean (ステップごとの平均値)
- continuous_action_std (標準偏差)
- regime_specific_action_distribution (regime別)

# 正常な値: Mean ≈ 0.0, Std ≈ 0.5-0.7
```

---

### B. Regime-Specific Action Targets の確認

現在の設定では、いくつかのレジームで目標が歪んでいる可能性:

```json
"sideways": {
  "BUY": 42.9%,
  "SELL": 0.0%,      // ← これが問題!
  "HOLD": 57.1%
}
```

**修正案**: すべてのレジームで BUY/SELL バランスを 50/50 に近づける

---

### C. Reward Clipping の影響

```
Reward Clipping: -10000.0 / +10000.0

現在の報酬:
- Mean: -9845 (ほぼ下限に達している)
- Max: +580

この suggests: Reward が十分に negative に圧縮されている
```

**改善案**: Clipping 前に Reward を normalization する

---

## 📋 チェックリスト

### 実装前の準備
- [ ] 現在のモデルを backup (models/ ディレクトリ)
- [ ] Config ファイル 3 つを確認
- [ ] logging パスが正しいか確認

### Phase 1 実行
- [ ] scale_200 で training 実行
- [ ] 結果を analysis/ に保存
- [ ] 統計値を確認

### Phase 2 実行
- [ ] scale_300 で training 実行
- [ ] scale_200 との比較分析

### Phase 3 実行
- [ ] scale_500 で training 実行
- [ ] 3 つすべてを比較

### 最適設定の選択
- [ ] 最高の Mean Reward の設定を選択
- [ ] その設定で backtest 実行
- [ ] 結果をドキュメント化

---

## ⚠️ トラブルシューティング

### 問題 1: 改善がない (Mean Reward が変わらない)

**原因の可能性**:
1. Config ファイルが正しく読み込まれていない
2. `quick_train_v444.py` が旧設定を使用している

**解決方法**:
```bash
# Config の確認
python -c "import json; print(json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))['environment']['behavior_optimization']['balance_penalty'])"

# Output: 200.0 であることを確認
```

---

### 問題 2: Training が不安定 (Loss がスパイク)

**原因の可能性**:
1. Balance Penalty が急激に変動している
2. Action Bonuses が過度に大きい

**解決方法**:
```json
// action_bonuses を調整
{
  "buy_action_bonus": 5.0,  // 10.0 から削減
  "sell_action_bonus": 2.5, // 5.0 から削減
  "hold_action_bonus": 1.0  // 2.0 から削減
}
```

---

### 問題 3: SELL Bias が改善されない

**原因の可能性**:
1. Reward Function がまだ SELL を促進している
2. Continuous Action の初期値がバイアスされている

**解決方法**:
```python
# reward_function.py で以下を確認
# SELL アクション ボーナスを削減または削除
if action == SELL:
    reward += 0  # 前: 5.0

# Continuous action を正規化
continuous_action = np.clip(continuous_action, -1, 1)
```

---

## 🚀 次のステップ

### Short Term (今週)
1. [ ] 3 つの config でそれぞれ 3000 ステップ training
2. [ ] 結果の比較分析
3. [ ] 最適な config を選択

### Medium Term (来週)
1. [ ] 選択した config で 10000+ ステップ training
2. [ ] Backtest で trading performance を検証
3. [ ] Feature Selection の最適化

### Long Term (今月)
1. [ ] Fine-tune regime-specific parameters
2. [ ] Continuous Action Distribution の正規化
3. [ ] Production deployment

---

## 📚 参考資料

### 関連ファイル
- Config: `config/sac_v444_3_balanced_penalty_scale_200.json`
- Config: `config/sac_v444_4_balanced_penalty_scale_300.json`
- Config: `config/sac_v444_5_balanced_penalty_scale_500.json`
- Training: `quick_train_v444_multi_config.py`
- Analysis: `analysis/parameter_tuning_analysis.py`

### 重要な指標
- `mean_reward`: メイン評価指標
- `buy_action_ratio`: BUY アクションの比率
- `sell_action_ratio`: SELL アクションの比率
- `continuous_action_mean`: 連続アクション分布の平均

---

**最後の注意**: このデバッグプロセスは段階的です。各フェーズの結果に基づいて、次のステップを決定してください。
性急に結論を出さず、データに基づいた判断を心がけてください。
