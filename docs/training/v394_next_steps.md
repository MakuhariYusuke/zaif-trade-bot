# v394訓練結果と今後の方針

## 📊 現状分析

### 訓練実行結果
全てのv394バージョンが**早期終了**しています：

| Version | Session | File Size | 推定進捗 | 推定Timesteps |
|---------|---------|-----------|----------|---------------|
| v394a   | session_5 | 15.8 KB | 0.8% | ~800 |
| v394b   | session_7 | 164.7 KB | 8.2% | ~8,200 |
| v394c   | session_8 | 164.7 KB | 8.2% | ~8,200 |
| v394d (1st) | session_9 | 7.5 KB | 0.4% | ~400 |
| v394d (2nd) | session_10 | 164.7 KB | 8.2% | ~8,200 |
| v394e   | session_6 | 164.7 KB | 8.2% | ~8,200 |

### 早期終了の原因
1. **メモリ不足**: 4つ並行実行でメモリ枯渇
2. **KeyboardInterrupt**: 手動中断（ログから確認済み）
3. **環境エラー**: action_mask関連のエラー（一部）

## 🎯 新しい戦略

### 方針転換: 1つずつ完全訓練
メモリ制約を考慮し、**1つのバージョンを確実に100,000 timestepsまで訓練**

### 優先順位
1. **v394d (最優先)**: 初期HOLD 50%で最も有望
2. **v394e**: 高エントロピー版（ent_coef 0.05）
3. **v394b**: 取引報酬強化版
4. **v394c**: バランス調整版

## 🚀 実行計画

### Step 1: v394d完全訓練（NOW）
```bash
.venv311\Scripts\python.exe train_v394d.py
```

**訓練パラメータ**:
- Total timesteps: 100,000
- 報酬設定:
  - HOLD罰則: 0.1 (5倍)
  - 取引報酬: 5.0 (5倍)
  - 利益倍率: 10.0 (2倍)
  - 取引頻度: 0.3 (2倍)
- 期待される結果: 初期HOLD 50%を維持

**モニタリング**:
- メモリ使用量を監視
- 定期的にAction分布を確認
- checkpoint_interval: 10,000 steps

### Step 2: v394d評価（訓練完了後）
```bash
# Stochastic推論でバックテスト
.venv311\Scripts\python.exe stochastic_backtest.py \
  --model models/ppo_v394d_aggressive/best_model.zip \
  --data btc_jpy_real_dataset.csv \
  --episodes 10
```

**評価指標**:
- Return (%)
- Sharpe Ratio
- 取引回数
- Action分布（HOLD/BUY/SELL比率）

### Step 3: 結果に応じて次のバージョン
- **v394dが成功** (Return > 0%): 実運用準備
- **v394dが不十分**: v394e（高エントロピー）を訓練

## 💡 学んだこと

### メモリ管理
- **並行訓練は危険**: 4つ同時実行でメモリ不足
- **順次実行が安全**: 1つずつ確実に完了
- **チェックポイント重要**: 10,000 stepsごとに保存

### 訓練戦略
- **初期Action分布が重要**: v394dの50%が最良
- **報酬シェーピングの効果**: HOLD罰則 + 取引報酬の両方が必要
- **エントロピーの限界**: ent_coef 0.05でも不十分

### 次の改善案
1. **より高いエントロピー**: ent_coef 0.1-0.2
2. **カリキュラム学習**: 段階的に難易度調整
3. **温度パラメータ**: Softmax温度で確率分布調整
4. **Stochastic推論**: 訓練時の多様性を活用

## 📋 実行コマンド

### v394d訓練（最優先）
```bash
# セッションクリーンアップ（オプション）
Remove-Item -Path "checkpoints\ppo_session_*" -Recurse -Force

# v394d訓練実行
.venv311\Scripts\python.exe train_v394d.py

# 進捗モニタリング
.venv311\Scripts\python.exe analyze_v394_training.py
```

### 訓練完了後の評価
```bash
# モデル確認
.venv311\Scripts\python.exe check_v394_completion.py

# Stochasticバックテスト
.venv311\Scripts\python.exe stochastic_backtest.py \
  --model models/ppo_v394d_aggressive/best_model.zip \
  --data btc_jpy_real_dataset.csv \
  --episodes 10
```

---

**次のアクション**: v394d（激辛版）を**1つだけ**100,000 timestepsまで訓練実行
**目標**: 初期HOLD 50%の良好なAction分布を維持したまま訓練完了
