# 🎯 儲かるモデルを探す - 実行計画

**作成日**: 2025-01-10
**目的**: 既知の知見（v378安定性、v379積極性）を活かし、収益性のあるモデルを発見する

---

## 📊 現状分析

### ✅ 完了した改善
1. **Phase 1-4**: 特徴量スキーマ管理システム完全実装
2. **HOLD偏重修正**: 5つの致命的バグ修正（アクション分布が99% → 50% HOLDに改善）
3. **Bitcoin現実価格対応**: 18M円を前提とした少額取引対応
4. **リターン計算修正**: ポジションクローズバグ修正（幻のリターン排除）

### 🔴 発見された問題
- **v385**: 収益性0% （以前は誤って0.85%と表示）
- **v384以下**: 評価不可能または動作不良
- **結論**: 現存する評価可能なモデルに収益性なし

### 💡 既知の知見
- **v378**: 安定性重視モデル（特徴量68次元）
- **v379**: 積極性重視モデル（詳細未確認）

---

## 🔍 Phase 1: 既存モデルの徹底調査

### 1.1 v378の詳細評価 🎯

**目的**: 安定性の源泉を特定

```bash
# アクション分布確認
.venv311\Scripts\python.exe check_action_distribution.py \
  --model models\ppo_reward_v378.zip --steps 1000

# バックテスト（50エピソード）
.venv311\Scripts\python.exe quick_backtest.py \
  --model models\ppo_reward_v378.zip \
  --data ml-dataset-enhanced.csv --episodes 50

# スキーマ確認
.venv311\Scripts\python.exe -c "import json; print(json.dumps(json.load(open('models/ppo_reward_v378_schema.json')), indent=2))"
```

**期待される情報**:
- アクション分布の特徴
- 収益性の有無
- 環境設定（initial_portfolio_value, max_position_size等）
- 使用特徴量（68次元）の内訳

### 1.2 v379の詳細評価 🚀

**目的**: 積極性の源泉を特定

```bash
# アクション分布確認
.venv311\Scripts\python.exe check_action_distribution.py \
  --model models\ppo_reward_v379.zip --steps 1000

# バックテスト（50エピソード）
.venv311\Scripts\python.exe quick_backtest.py \
  --model models\ppo_reward_v379.zip \
  --data ml-dataset-enhanced.csv --episodes 50

# スキーマ確認（存在する場合）
.venv311\Scripts\python.exe -c "import json; print(json.dumps(json.load(open('models/ppo_reward_v379_schema.json')), indent=2))"
```

**期待される情報**:
- BUY/SELL比率の高さ
- 収益性の有無
- v378との設定差分

### 1.3 全モデルスキーマ一覧作成 📋

```bash
# 全モデルのスキーマファイルを確認
Get-ChildItem models\*_schema.json | ForEach-Object {
  Write-Host "`n=== $($_.Name) ===";
  python -c "import json; s=json.load(open('$_')); print(f'Features: {len(s.get(\"features\", []))}'); print(f'Env: {s.get(\"env_config\", {})}'); print(f'Date: {s.get(\"timestamp\", \"N/A\")}')"
}
```

---

## 🧪 Phase 2: 比較分析

### 2.1 v378 vs v379 比較表作成

| 項目 | v378 (安定性) | v379 (積極性) | 差分 |
|------|---------------|---------------|------|
| 特徴量数 | 68 | ??? | ??? |
| HOLD率 | ??? | ??? | ??? |
| BUY率 | ??? | ??? | ??? |
| SELL率 | ??? | ??? | ??? |
| 平均Return | ??? | ??? | ??? |
| トレード頻度 | ??? | ??? | ??? |
| 資金設定 | ??? | ??? | ??? |
| max_position_size | ??? | ??? | ??? |

### 2.2 成功要因の仮説立案

**v378が安定している理由（仮説）**:
- [ ] 特徴量選択が適切
- [ ] HOLD penaltyが適度
- [ ] リスク管理が保守的
- [ ] 訓練データが適切

**v379が積極的な理由（仮説）**:
- [ ] profit reward multiplierが高い
- [ ] HOLD penaltyが強い
- [ ] action maskingが緩い
- [ ] max_position_sizeが大きい

---

## 🚀 Phase 3: 新規訓練戦略

### 3.1 ベースライン訓練 - v378設定復元

**目的**: v378の設定を再現し、スキーマ保存機能付きで訓練

```json
// configs/training/ppo_v378_baseline.json (新規作成)
{
  "algorithm": "PPO",
  "total_timesteps": 1000000,
  "env_config": {
    "initial_portfolio_value": 10000,  // v378の実際の値に合わせる
    "max_position_size": 0.5,          // v378の実際の値に合わせる
    "transaction_cost": 0.0005,
    "use_curated_features": true,
    "feature_count": 68                // v378と同じ特徴量数
  },
  "reward_config": {
    // v378の設定を確認後に記入
  }
}
```

### 3.2 改善訓練 - 現実的Bitcoin設定

**目的**: v378/v379の成功要因 + Bitcoin 18M円対応

```json
// configs/training/ppo_realistic_profitable.json (新規作成)
{
  "algorithm": "PPO",
  "total_timesteps": 1000000,
  "env_config": {
    "initial_portfolio_value": 200000,  // 0.01 BTC購入可能
    "max_position_size": 0.01,          // 180,000円相当
    "transaction_cost": 0.0005,
    "use_curated_features": true,
    "feature_count": 68                 // v378の成功特徴量
  },
  "reward_config": {
    "profit_reward_multiplier": 10.0,   // v379の積極性を参考
    "hold_penalty": -0.01,              // v378の安定性を参考
    "risk_penalty_multiplier": 0.5,
    "trade_count_penalty": -0.001
  }
}
```

### 3.3 実験的訓練 - アンサンブル特性

**目的**: v378安定性 + v379積極性のハイブリッド

```json
// configs/training/ppo_hybrid_v378_v379.json (新規作成)
{
  "algorithm": "PPO",
  "total_timesteps": 2000000,          // 長期訓練
  "env_config": {
    "initial_portfolio_value": 200000,
    "max_position_size": 0.02,         // v378とv379の中間
    "transaction_cost": 0.0005,
    "use_curated_features": true,
    "feature_count": 68
  },
  "reward_config": {
    // v378とv379の報酬設定を混合
    "profit_reward_multiplier": 15.0,
    "hold_penalty": -0.005,
    "risk_penalty_multiplier": 0.3
  }
}
```

---

## 📈 Phase 4: 評価とイテレーション

### 4.1 評価基準

**必須条件**:
- ✅ 平均Return > 0.5%
- ✅ 標準偏差 < 2% （安定性）
- ✅ トレード頻度 > 5回/エピソード （アクション多様性）
- ✅ アクション分布: HOLD < 70%

**理想条件**:
- 🌟 平均Return > 2%
- 🌟 標準偏差 < 1%
- 🌟 Sharpe Ratio > 1.0
- 🌟 最大ドローダウン < 5%

### 4.2 イテレーション計画

```
Iteration 1: v378ベースライン訓練
  ↓
評価 → 成功？ → Yes → 実取引検証へ
  ↓ No
Iteration 2: 報酬関数調整（profit_reward_multiplier +5）
  ↓
評価 → 成功？ → Yes → 実取引検証へ
  ↓ No
Iteration 3: hold_penalty強化（-0.01 → -0.02）
  ↓
評価 → 成功？ → Yes → 実取引検証へ
  ↓ No
Iteration 4: ハイブリッド訓練（v378 + v379）
```

---

## 🎬 実行順序

### Step 1: 既存モデル調査（今すぐ実行） ⏰

```bash
# v378評価
.venv311\Scripts\python.exe check_action_distribution.py --model models\ppo_reward_v378.zip --steps 1000
.venv311\Scripts\python.exe quick_backtest.py --model models\ppo_reward_v378.zip --data ml-dataset-enhanced.csv --episodes 50

# v379評価
.venv311\Scripts\python.exe check_action_distribution.py --model models\ppo_reward_v379.zip --steps 1000
.venv311\Scripts\python.exe quick_backtest.py --model models\ppo_reward_v379.zip --data ml-dataset-enhanced.csv --episodes 50
```

### Step 2: スキーマ分析（調査結果を基に） 📊

```bash
# v378スキーマ確認
python -c "import json; print(json.dumps(json.load(open('models/ppo_reward_v378_schema.json')), indent=2))"

# v379スキーマ確認（存在する場合）
python -c "import json; print(json.dumps(json.load(open('models/ppo_reward_v379_schema.json')), indent=2))"
```

### Step 3: 訓練設定作成（スキーマ分析後） ⚙️

v378/v379の成功要因を特定し、`ppo_v378_baseline.json`を作成

### Step 4: ベースライン訓練（設定完成後） 🚂

```bash
.venv311\Scripts\python.exe -m ztb.training.core.unified_trainer \
  --config configs/training/ppo_v378_baseline.json
```

### Step 5: 評価とイテレーション（訓練完了後） 🔄

新モデルを評価し、必要に応じて報酬関数を調整して再訓練

---

## 📝 予想される課題と対策

### 課題1: v378/v379のスキーマファイルが存在しない

**対策**:
- `models/*.zip`をロードして環境情報を逆算
- 訓練ログやコンフィグファイルから設定を復元
- 最悪の場合、デフォルト設定からスタート

### 課題2: v378/v379も収益性ゼロの可能性

**対策**:
- 報酬関数の根本的見直し
- 市場データの質を確認（ml-dataset-enhanced.csv）
- 別のアルゴリズム（SAC, A2C等）を検討

### 課題3: 訓練時間が長すぎる（1M timesteps）

**対策**:
- まず100k timestepsでクイック検証
- 有望な設定のみ1M timestepsで本訓練
- GPU利用を検討（Dockerfile.training活用）

---

## 🎯 成功の定義

**短期目標** (1週間):
- ✅ v378/v379の詳細評価完了
- ✅ 成功要因の仮説立案
- ✅ 1つ以上の新モデル訓練完了

**中期目標** (1ヶ月):
- 🎯 平均Return > 0.5%のモデル発見
- 🎯 ペーパートレードで安定性確認
- 🎯 実取引ドライラン成功

**長期目標** (3ヶ月):
- 🌟 実取引で月次+5%達成
- 🌟 複数モデルのアンサンブル運用
- 🌟 自動リバランス機能実装

---

## 🚀 次のアクション

**今すぐ実行**:
```bash
# v378とv379の評価を並行実行
.venv311\Scripts\python.exe check_action_distribution.py --model models\ppo_reward_v378.zip --steps 1000
```

この結果を基に次のステップを決定します！
