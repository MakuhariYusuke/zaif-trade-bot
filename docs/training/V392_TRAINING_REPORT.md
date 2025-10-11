# v392訓練・評価レポート（random_startバグ修正版）

## 📅 実施日時
2025-10-11

## 🎯 目的
random_startバグ修正後の環境で初めての訓練を実施し、以下を検証：
1. random_startが正しく機能するか
2. HOLD偏重問題が解消されるか
3. 収益性のあるモデルが作成できるか

## 🔧 適用したバグ修正

### 1. schema_env_factory.py修正
**問題**: `random_start`がconfig辞書に含まれるだけで、位置引数として渡されていなかった

**修正** (Lines 82-90):
```python
# random_startをconfig辞書から取り出して位置引数として明示的に渡す
random_start = env_config.pop("random_start", False)
logger.info(f"Creating HeavyTradingEnv with random_start={random_start}")
env = HeavyTradingEnv(df=df, config=env_config, random_start=random_start)
```

### 2. HeavyTradingEnv.reset()修正
**問題**: `DEFAULT_RANDOM_START_BUFFER=100`固定で、データ100行以下ではmax_start=0になり、random_startが無効化されていた

**修正** (core.py Lines 269-285):
```python
# バッファを動的計算（データ長の10%、最小10、最大100）
buffer = max(10, min(100, int(self.n_steps * 0.1)))
max_start = max(0, self.n_steps - buffer)
self.current_step = np.random.randint(min_start, max_start + 1)
logger.debug(f"Random start: current_step={self.current_step}, range=[{min_start}, {max_start}], buffer={buffer}, n_steps={self.n_steps}")
```

## ✅ バグ修正の検証結果

### 修正前（v390）
```
Episode 1: Start index: 0, Obs: [7.168750e+01 ...]
Episode 2: Start index: 0, Obs: [7.168750e+01 ...]  # 完全一致（決定論的）
Episode 3: Start index: 0, Obs: [7.168750e+01 ...]  # 完全一致
Reward標準偏差: ±0.00（全エピソード同一値）
```

### 修正後（v392評価前の検証）
```
Episode 1: Start index: 32, Obs: [5.00...e+01 1.7755264e+07 ...]
Episode 2: Start index: 34, Obs: [5.00...e+01 1.7755198e+07 ...]  # 異なる！
Episode 3: Start index: 71, Obs: [5.00...e+01 1.7756068e+07 ...]  # 異なる！
Reward標準偏差: ±28.18（決定論的ではない）
```

**✅ random_startバグは完全に修正されました**

## 📊 v392訓練結果

### 訓練設定
- **Model Name**: ppo_profitable_v392_bugfix（実際: ppo_session.zip → リネーム）
- **Total Timesteps**: 100,352
- **Data**: btc_jpy_real_dataset.csv (100 rows)
- **Features**: 68（スキーマハッシュ: c7a296f3d7c6ece4）

### 訓練中の指標

#### 最終値（iterations=98, total_timesteps=100,352）
```
ep_len_mean:           99
ep_rew_mean:           -27.9
learning_rate:         0.0003
entropy_mean_entropy:  1.06
entropy_target_entropy: 0.769
```

#### PAN (Policy Activation Network) 統計
```
pan_action_counts: [17, 9, 6]（最終バッチ）
  HOLD: 17/32 = 53%
  BUY:   9/32 = 28%
  SELL:  6/32 = 19%
```

**🎯 訓練中はバランスの取れたアクション分布を示していた！**

### ⚠️ 発見された問題

#### 1. ハイパーパラメータ未適用
**設定ファイル（v392設定）**:
```json
{
  "learning_rate": 0.007503,  // 二分探索最適化値（v390の25倍）
  "batch_size": 256,          // v390の2倍
  "n_steps": 1024,            // v390の2倍
  "n_epochs": 16,             // v390の2.67倍
  ...
}
```

**実際に使用された値**:
```
learning_rate: 0.0003  （デフォルト値）
batch_size:    32      （デフォルト値）
```

**原因**: unified_trainer.pyがppo_hyperparametersを正しく渡していない

#### 2. モデル名の問題
- 期待: `ppo_profitable_v392_bugfix.zip`
- 実際: `ppo_session.zip`
- session_idがmodel_nameを上書き

## 📊 v392バックテスト評価結果（10エピソード）

```
Episode  1: Reward= -88.80, Return=  0.00%, Trades=  1
Episode  2: Reward= -15.80, Return=  0.00%, Trades=  1
Episode  3: Reward= -92.80, Return=  0.00%, Trades=  1
Episode  4: Reward= -31.80, Return=  0.00%, Trades=  1
Episode  5: Reward= -76.80, Return=  0.00%, Trades=  1
Episode  6: Reward= -47.80, Return=  0.00%, Trades=  1
Episode  7: Reward= -30.80, Return=  0.00%, Trades=  1
Episode  8: Reward= -59.80, Return=  0.00%, Trades=  1
Episode  9: Reward= -34.80, Return=  0.00%, Trades=  1
Episode 10: Reward= -79.80, Return=  0.00%, Trades=  1

Average Reward:   -55.90 ± 26.07
Average Return:    0.00% ±  0.00%
Total Trades:      10
Trades/Episode:    1.0
```

### アクション分布（1,000ステップ）
```
HOLD:  995 (99.5%) ██████████████████████████████████████████████████
BUY:     2 ( 0.2%)
SELL:    3 ( 0.3%)

❌ HOLD偏重 - ほぼ取引していません
```

## 🔴 重大な発見：訓練時vs推論時の乖離

### 訓練時（exploration有効）
```
HOLD: 53%
BUY:  28%
SELL: 19%
→ バランスの取れた分布
```

### 推論時（deterministic=True）
```
HOLD: 99.5%
BUY:  0.2%
SELL: 0.3%
→ 極端なHOLD偏重
```

**この乖離は異常です！**

## 🔍 推定される原因

### 1. Deterministic推論モードの影響
`quick_backtest.py`, `check_action_distribution.py`では：
```python
action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
```
- `deterministic=True`: explorationを無効化し、最も確率の高いアクションのみ選択
- 訓練時: stochasticサンプリング（多様なアクション）
- 推論時: deterministicモード（常にmax probabilityアクション）

### 2. Value Function問題
訓練ログを見ると：
```
value_loss: 112-196（高い）
explained_variance: 0.0（価値関数が状態を説明できていない）
```
→ 価値関数が正しく学習できていない可能性

### 3. Entropy Coefficient
```
entropy_coef: 0.0（設定値）
```
→ Explorationが報酬に寄与しない
→ 訓練時のentropyはTarget Entropy Controllerで管理されているが、推論時は無関係

## 📈 v390との比較

| 項目 | v390（バグあり） | v392（バグ修正） |
|------|-----------------|------------------|
| random_start機能 | ❌ 全エピソード step 0 | ✅ ランダム開始位置 |
| Reward標準偏差 | ±0.00（決定論的） | ±26.07（ランダム） |
| 訓練時HOLD率 | 52% | 53% |
| 推論時HOLD率 | 99.6% | 99.5% |
| Return | 0.00% | 0.00% |
| Trades/Episode | 0 | 1.0 |

**🔴 結論**: random_startバグは修正されたが、HOLD偏重問題は依然として存在

## 🎯 次のアクション

### 優先度1: ハイパーパラメータ適用バグ修正
- unified_trainer.pyでppo_hyperparametersを正しく渡す
- 二分探索最適化値（learning_rate 0.007503等）を適用

### 優先度2: 訓練時vs推論時の乖離調査
1. **Deterministicモード検証**:
   - `deterministic=False`でバックテスト実行
   - Stochastic推論でHOLD率が変わるか確認

2. **環境設定の一貫性確認**:
   - 訓練時と推論時で環境設定が同一か検証
   - reward_settings、action_maskingの適用状況確認

3. **Value Function診断**:
   - explained_variance=0.0の原因調査
   - 価値関数の学習状況確認

### 優先度3: v393訓練
- ハイパーパラメータ修正後
- 二分探索最適化値を正しく適用
- 訓練時の詳細ログ収集

## 📝 メモ
- v390/v391: バグ修正前の訓練データ → 評価無効
- v392: random_startバグ修正版 → 評価有効だが、ハイパーパラメータ未適用
- v393（予定）: ハイパーパラメータ適用版

## 🔧 修正済みファイル
1. `ztb/trading/environment/schema_env_factory.py` (Lines 82-90)
2. `ztb/trading/environment/heavy_env/core.py` (Lines 269-285)
3. `ztb/training/core/ppo_trainer.py` (List import追加、CURATED追加)
4. `debug_backtest_detailed.py` (新規作成)
