# v394シリーズ訓練進捗サマリー

## 📊 実行状況

### ✅ 完了/進行中
1. **v394a (HOLD罰則5倍)**: 10,240 steps完了（中断）
   - 初期HOLD 48% → 最終91%
   - エントロピー係数0.01では不十分

2. **v394b (取引報酬5倍)**: 訓練中（session_7）
   - 初期データ: HOLD 55%（2,048 steps）

3. **v394c (バランス調整)**: 訓練中（session_8）
   - 初期段階

4. **v394d (激辛版)**: 5,120 steps完了（中断）
   - 🏆 **初期HOLD 50%** - 最良！
   - BUY 67, SELL 61（合計50%）
   - 最も効果的な設定

5. **v394e (高エントロピー)**: 8,192 steps完了（進行中）
   - 初期HOLD 52% → 83%
   - ent_coef 0.05でも効果限定的

## 🎯 重要な発見

### 1. v394d（激辛版）が最も有望
```json
{
  "hold_penalty_weight": 0.1,           // 5倍
  "consecutive_hold_penalty": 0.05,     // 5倍
  "successful_trade_bonus": 5.0,        // 5倍
  "profit_reward_multiplier": 10.0,     // 2倍
  "trading_frequency_bonus": 0.3        // 2倍
}
```

**初期Action分布**:
- HOLD: 128/256 (50%)
- BUY: 67/256 (26%)
- SELL: 61/256 (24%)

### 2. 共通の課題
- **全バージョンでHOLD比率が訓練中に上昇**
- v394a: 48% → 91%
- v394d: 50% → 75%
- v394e: 52% → 83%

### 3. エントロピー係数の効果
- ent_coef 0.01 (v394a): HOLD 91%
- ent_coef 0.05 (v394e): HOLD 83%
- **改善はあるが十分ではない**

## 🚀 次のアクション

### 優先度1: v394d完全訓練
```bash
.venv311\Scripts\python.exe train_v394d.py
```
- 最も有望な設定で100,000 timesteps完了
- 最終的なAction分布を確認

### 優先度2: Stochastic推論評価
```bash
.venv311\Scripts\python.exe stochastic_backtest.py \
  --model models/ppo_v394d \
  --data btc_jpy_real_dataset.csv \
  --episodes 10
```
- deterministic=False で評価
- 訓練時Action分布（HOLD 50-75%）が実用的か確認
- 収益性を検証

### 優先度3: 超高エントロピー版（v394f）
```json
{
  "ent_coef": 0.1,  // 0.05 → 0.1 (2倍)
  // v394dの報酬設定を継承
}
```
- さらに高いエントロピーで探索維持
- HOLD収束を最大限抑制

### 優先度4: 比較評価
- v394a, b, c, d, e, f のStochasticバックテスト
- 収益性、Action分布、取引回数を比較
- 最適モデル選定

## 📈 期待される結果

### シナリオA: Stochastic推論で収益化成功
- v394dの訓練時Action分布（HOLD 50-75%）が実際に使える
- deterministic=Falseでバックテスト
- **Return > 0%** が達成できれば成功

### シナリオB: さらなる改善が必要
- 超高エントロピー（ent_coef 0.1-0.2）
- カリキュラム学習
- 温度パラメータ調整

## 💡 技術的洞察

### RLの保守的学習問題
- PPOは本質的に保守的（安全策優先）
- 取引コスト（手数料、スリッページ）が大きい
- HOLDが「無難な選択」として学習される

### 解決アプローチ
1. **報酬シェーピング**: v394dのように両方強化
2. **エントロピー維持**: 高ent_coefで探索継続
3. **Stochastic推論**: 訓練時の多様性を保持
4. **カリキュラム学習**: 段階的に難易度調整

## 🎬 実行コマンド

```bash
# v394d完全訓練（最優先）
.venv311\Scripts\python.exe train_v394d.py

# 訓練完了後、Stochasticバックテスト
.venv311\Scripts\python.exe stochastic_backtest.py \
  --model models/ppo_v394d/best_model.zip \
  --data btc_jpy_real_dataset.csv \
  --episodes 10

# 進捗モニタリング
.venv311\Scripts\python.exe monitor_v394_progress.py

# TensorBoard起動（オプション）
tensorboard --logdir checkpoints
```

---

**現在の時刻**: 並行訓練実行中
**次の確認**: 数分後に各バージョンの進捗チェック
**目標**: v394dで初期HOLD 50%を維持しつつ100,000 timesteps完了
