# SELL回避問題 - 緊急修正サマリー（v3.6.2）

**日時:** 2025-10-08
**問題:** SELL rate 1.6% → 極めて深刻
**対応:** 緊急パッチ v3.6.2

---

## 🚨 問題の本質

```
学習結果:
  SELL Rate: 1.6%  （目標: 33%）← 20倍以上のズレ
  Lambda: 2.000000 （上限に飽和）
  Constraint Active: True （機能していない）

推定分布:
  HOLD: 68.88%
  BUY:  29.52%
  SELL: 1.60%  ← ほぼゼロ
```

**診断:** Lagrange制約が完全に飽和し、SELLを誘導できていない。報酬関数でSELLが著しく不利になっている。

---

## ✅ 適用した緊急修正

### 1. Lagrange制約 10倍強化 ⭐

```diff
- "lagrange_eta": 0.05,
- "lagrange_lambda_max": 2.0,
- "lagrange_warmup_steps": 1000,

+ "lagrange_eta": 0.1,           // 2倍速
+ "lagrange_lambda_max": 20.0,   // 10倍の余裕
+ "lagrange_warmup_steps": 500,  // 早期発動
```

### 2. SELL報酬 2倍 + ペナルティ削減 ⭐

```diff
- "profit_bonus_multipliers": [1.0, 1.0, 1.0],
- "action_penalty_scale": 0.01,
- "trade_frequency_penalty": 0.01,
- "consecutive_trade_penalty": 0.05,

+ "profit_bonus_multipliers": [1.0, 1.0, 2.0],  // SELL 2倍
+ "action_penalty_scale": 0.001,    // 1/10に
+ "trade_frequency_penalty": 0.0,   // 一時無効
+ "consecutive_trade_penalty": 0.0, // 一時無効
```

### 3. 探索強化 5倍

```diff
- "ent_coef": 0.1,
- "enable_stratified_sampling": false,

+ "ent_coef": 0.5,                    // 5倍
+ "enable_stratified_sampling": true, // 強制
```

---

## 📋 次のアクション

### 1. 検証学習（今すぐ実行）

```bash
python run_training.py \
  --config configs/training/ppo_balanced_mem_optimized.json \
  --timesteps 10000
```

### 2. 成功基準

**最低限（10000 steps）:**
- SELL rate ≥ 15%
- Lambda < 15.0（飽和していない）

**理想（30000 steps）:**
- SELL rate: 30-35%
- 均等分布（33/33/33）

### 3. 結果確認

```bash
# ログから自動診断
python scripts/analyze_training_logs_v2.py <ログファイル>
```

確認項目:
```
Final Lagrange Statistics:
  SELL Rate (avg): ??%  ← 15%以上？
  Lambda (final): ??    ← 20.0未満？
```

---

## ⚠️ 重要な注意事項

### これは診断用の極端な設定です

- ペナルティを全て無効化 → 一時的措置
- SELL報酬を2倍 → 過矯正のリスク
- エントロピー0.5 → 学習が不安定化する可能性

### SELL率改善後の段階的調整が必須

**Phase 1: SELL率が15%以上になったら**
```json
{
  "profit_bonus_multipliers": [1.0, 1.0, 1.5],  // 2.0 → 1.5
  "ent_coef": 0.3  // 0.5 → 0.3
}
```

**Phase 2: SELL率が25%以上になったら**
```json
{
  "trade_frequency_penalty": 0.005,  // 0.0 → 0.005
  "ent_coef": 0.2  // 0.3 → 0.2
}
```

**Phase 3: SELL率が30-35%で安定したら**
```json
{
  "profit_bonus_multipliers": [1.0, 1.0, 1.2],
  "consecutive_trade_penalty": 0.02,
  "ent_coef": 0.15,
  "lagrange_lambda_max": 10.0  // 過剰なら削減
}
```

---

## 📊 作成したファイル

1. **設定ファイル修正:**
   - `configs/training/ppo_balanced_mem_optimized.json`

2. **診断ツール:**
   - `scripts/analyze_training_logs_v2.py` - ログ自動解析
   - `scripts/diagnose_action_distribution.py` - 対話型診断

3. **ドキュメント:**
   - `docs/SELL_AVOIDANCE_EMERGENCY_FIX.md` - 詳細レポート
   - `training_log_sample.txt` - サンプルログ
   - `sell_avoidance_diagnosis.json` - 診断結果

4. **バージョン管理:**
   - `package.json`: 3.6.1 → 3.6.2
   - `CHANGELOG.md`: v3.6.2セクション追加

---

## 🎯 期待される結果

### Before（現在）
```
SELL: 1.6%   ← 壊滅的
HOLD: 68.9%
BUY:  29.5%
```

### After（目標）
```
SELL: 30-35% ← 理想
HOLD: 30-35%
BUY:  30-35%
```

---

**実行してください:**
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000
```

結果をお知らせください。SELL率が改善されない場合、さらなる調整が必要です。
