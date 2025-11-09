# unified_trainer アクションバイアス問題 - デバッグ報告

## 🔍 根本原因の特定

### 問題の構造

unified_trainer では balance_penalty が **一切適用されていません**。

quick_train では正常に動作しているのに対し、unified_trainer では以下の問題が発生:

```
quick_train:
  ✅ balance_penalty が適用される
  ✅ action distribution がバランスしている

unified_trainer:
  ❌ balance_penalty が適用されない
  ❌ SELL bias が発生
```

---

## 🎯 根本原因（2段階の不一致）

### 問題 1: curriculum_stage の値が一致していない

**Config ファイル** (`sac_v444_3_balanced_penalty_scale_200.json`):
```json
{
  "training": {
    "curriculum_learning": {
      "curriculum_stage": "balanced_penalty"
    }
  }
}
```

**Reward Calculator** (`reward_calculator.py` line 233):
```python
if curriculum_stage == "forced_balance":  # ← ここが問題!
    balance_penalty_scale = self._get_behavior_opt("balance_penalty", DEFAULT_BALANCE_PENALTY_SCALE)
    balance_penalty = abs(buy_ratio - sell_ratio) * balance_penalty_scale
```

**一致しない値**:
- Config: `"balanced_penalty"`
- Code: `"forced_balance"`

結果: **if 条件が false になり、balance_penalty が計算されない**

---

### 問題 2: environment_config のデフォルト値

**environment_config.py** line 32:
```python
curriculum_stage: str = "pnl_focused"  # Default for v439 scalping
```

**config_manager.py** line 162:
```python
if isinstance(curriculum_learning, dict) and "curriculum_stage" in curriculum_learning:
    environment["curriculum_stage"] = curriculum_learning["curriculum_stage"]
```

流れ:
1. config に `curriculum_stage` が存在
2. config_manager が environment に設定
3. ここまでは OK

**しかし**、`curriculum_stage` が実装上 "forced_balance" を期待しているのに対し、
config では別の値 ("balanced_penalty") を使用している。

---

## 🐛 バグの流れ

```
1. Config をロード
   ↓
2. config_manager で environment 設定
   → curriculum_stage = "balanced_penalty" が設定される
   ↓
3. HeavyTradingEnv を初期化
   → config に curriculum_stage = "balanced_penalty" が設定される
   ↓
4. RewardCalculator で报酬を計算
   → `if curriculum_stage == "forced_balance"` をチェック
   → "balanced_penalty" ≠ "forced_balance" → FALSE
   ↓
5. balance_penalty の計算がスキップされる ← BUG!
   ↓
6. SELL bias が続く
```

---

## ✅ 修正案

### 修正 1: reward_calculator で複数の値をサポート

**ファイル**: `ztb/trading/environment/components/reward_calculator.py`

**変更箇所** (line 233 付近):
```python
# Before (バグあり)
if curriculum_stage == "forced_balance":
    balance_penalty_scale = self._get_behavior_opt("balance_penalty", DEFAULT_BALANCE_PENALTY_SCALE)
    balance_penalty = abs(buy_ratio - sell_ratio) * balance_penalty_scale

# After (修正版)
# Support multiple curriculum stage names that require balance penalty
if curriculum_stage in ("forced_balance", "balanced_penalty", "balance_optimization"):
    balance_penalty_scale = self._get_behavior_opt("balance_penalty", DEFAULT_BALANCE_PENALTY_SCALE)
    balance_penalty = abs(buy_ratio - sell_ratio) * balance_penalty_scale
```

---

### 修正 2: config ファイルを統一

**すべての config ファイル**で `curriculum_stage` を統一:

**Option A**: "forced_balance" に統一
```json
{
  "training": {
    "curriculum_learning": {
      "curriculum_stage": "forced_balance"
    }
  }
}
```

**Option B**: "balanced_penalty" をサポート (修正 1 後)
```json
{
  "training": {
    "curriculum_learning": {
      "curriculum_stage": "balanced_penalty"
    }
  }
}
```

推奨: **修正 1 を実施した上で、現在の config 値を保持**

---

## 🔧 実装手順

### Step 1: reward_calculator.py を修正

```bash
# ファイルを開く
c:\Users\Admin\dev\zaif-trade-bot\ztb\trading\environment\components\reward_calculator.py

# Line 233 付近を修正
# if curriculum_stage == "forced_balance": 
# ↓
# if curriculum_stage in ("forced_balance", "balanced_penalty", "balance_optimization"):
```

### Step 2: 修正を検証

```bash
# quick_train スクリプトで動作確認 (既に動いている)
python quick_train_v444_configurable.py \
  --config config/sac_v444_3_balanced_penalty_scale_200.json \
  --verbose

# unified_trainer で動作確認 (修正後の新規テスト)
python -c "
from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.config_loader import ConfigLoader

config = ConfigLoader.load_config('config/sac_v444_3_balanced_penalty_scale_200.json')
trainer = UnifiedTrainer(config)
result = trainer.train()
"
```

### Step 3: 複数の config で統一

すべての `sac_v444_*_scale_*.json` ファイルを確認し、
`curriculum_stage` が正しく設定されていることを確認。

---

## 📋 チェックリスト

修正実施項目:

- [ ] `reward_calculator.py` の line 233 を修正
- [ ] 修正内容を確認
- [ ] quick_train スクリプトで動作確認
- [ ] unified_trainer で動作確認
- [ ] すべての config ファイルで curriculum_stage を確認

---

## 🧪 修正後の検証

### 期待される改善

修正後、unified_trainer でも:

```
Before (バグ):
  Mean Reward: -9845
  BUY: 18%, SELL: 66.85%, HOLD: 15.15%
  Balance Penalty: 0.0 (適用されていない)

After (修正):
  Mean Reward: -5000～-2000 (期待値)
  BUY: 30-40%, SELL: 30-40%, HOLD: 20-30% (バランス化)
  Balance Penalty: 正常に適用される
```

### 検証方法

training ログから以下を確認:

```
# unified_trainer で実行
python -c "... trainer.train()" 2>&1 | grep -i "balance_penalty|forced_balance|balanced_penalty"

# 出力例 (修正後):
[INFO] FORCED_BALANCE: total_actions=50, buy=0.360, sell=0.320, hold=0.320, penalty=4.000
[INFO] FORCED_BALANCE: total_actions=100, buy=0.350, sell=0.340, hold=0.310, penalty=1.000
```

---

## 📝 追加のデバッグ情報

### reward_calculator.py での curriculum_stage の使用箇所

他にも curriculum_stage をチェックしている箇所:

```python
# Line 165-180 (action_bonuses)
if curriculum_stage == "forced_balance" or curriculum_stage in ["balanced_penalty", "balance_optimization"]:
    # action_bonuses をサポート
```

これらもすべて修正する必要があります。

---

## 🎓 なぜ quick_train では動作したのか？

quick_train スクリプトをチェック:

```python
# quick_train_v444_configurable.py
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))

# env_config の直接設定
env_config = {
    ...
    "curriculum_stage": "forced_balance",  ← 明示的に設定?
    ...
}
```

または、quick_train では別の環境設定ロジックを使用している可能性があります。

---

## 結論

**unified_trainer での SELL bias の根本原因**:
1. Config で `curriculum_stage: "balanced_penalty"` を設定
2. reward_calculator が `curriculum_stage == "forced_balance"` のみをサポート
3. 一致しないため balance_penalty が計算されない
4. Action bias が発生

**修正方法**:
reward_calculator で複数の curriculum_stage 値をサポートするように修正

**リスク**: 低（ロジックの拡張のみ）
**所要時間**: 5-10分
**テスト期間**: 30分（training 1回）

---

**次のステップ**: reward_calculator.py の修正を実施
