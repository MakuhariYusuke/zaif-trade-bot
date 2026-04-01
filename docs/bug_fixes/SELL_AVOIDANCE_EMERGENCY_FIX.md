# SELL回避問題の緊急修正レポート

**日時:** 2025-10-08
**問題:** SELL rate 1.6% → 極めて深刻なSELL回避
**バージョン:** 3.6.1 → 3.6.2（緊急パッチ）

---

## 📊 問題の詳細

### 学習結果
```
SELL Rate (avg): 1.6%  （目標: 33%）
Lambda (final): 2.000000  （上限に張り付き）
Constraint Active: True
```

### 推定アクション分布
- **HOLD:** 68.88% （推定）
- **BUY:** 29.52% （推定）
- **SELL:** 1.60% （実測）

### 診断結果
- **推定エントロピー:** 0.6831 / 1.0986 (62.2%)
- **重症度:** CRITICAL（極めて深刻）
- **状態:** Lagrange制約が完全に飽和、SELL誘導に失敗

---

## 🔍 根本原因の分析

### 1. Lagrange制約の設計問題
- **Lambda上限:** 2.0 → **完全に飽和**
- **学習率η:** 0.05 → 遅すぎる可能性
- **制約の強さ不足:** SELL誘導に失敗

### 2. 報酬関数の問題
SELLアクション時に以下のペナルティが累積:
- `action_penalty_scale: 0.01`
- `trade_frequency_penalty: 0.01`
- `consecutive_trade_penalty: 0.05`
- `transaction_cost: 0.001`

→ **これらが合わさってSELLが著しく不利**

### 3. Action Maskingの影響
- `min_holding_period: 5?` （推定）
- ポジション保持期間中はSELLがブロック
- 過度に制限的な可能性

### 4. データセット/環境の問題
- データセットの価格変動パターンがSELL不利？
- 報酬計算でSELL時のPnLが常にマイナス？

---

## 🔧 適用した緊急修正

### ppo_balanced_mem_optimized.json への変更

#### 1. Lagrange制約の大幅強化 ⭐最重要
```json
{
  "lagrange_eta": 0.05 → 0.1,           // 学習率を2倍に
  "lagrange_lambda_max": 2.0 → 20.0,    // 上限を10倍に（飽和を解消）
  "lagrange_warmup_steps": 1000 → 500   // ウォームアップを短縮
}
```

**理由:**
- Lambda=2.0で完全に飽和していた
- 制約が弱すぎてSELLを誘導できていなかった
- 上限を20.0にすることで十分な制約強度を確保

#### 2. 報酬関数の調整（SELLインセンティブ強化）⭐最重要
```json
{
  "reward_settings": {
    "profit_bonus_multipliers": [1.0, 1.0, 1.0] → [1.0, 1.0, 2.0],  // SELL報酬を2倍
    "action_penalty_scale": 0.01 → 0.001,          // ペナルティを1/10に
    "trade_frequency_penalty": 0.01 → 0.0,         // 一時的に無効化
    "trade_cooldown_penalty": 0.01 → 0.0,          // 一時的に無効化
    "consecutive_trade_penalty": 0.05 → 0.0        // 一時的に無効化
  }
}
```

**理由:**
- SELL時の報酬を2倍にしてインセンティブを強化
- 取引ペナルティを全て無効化してSELLの障壁を除去
- まずSELL率を改善し、その後ペナルティを段階的に復活

#### 3. 多様性の強制 ⭐重要
```json
{
  "ent_coef": 0.1 → 0.5,                    // エントロピー係数を5倍に
  "enable_stratified_sampling": false → true // 層別サンプリング有効化
}
```

**理由:**
- エントロピー係数を大幅に増やして探索を強化
- 層別サンプリングでアクション分布を強制的にバランス

#### 4. その他の設定
```json
{
  "curriculum_stage": "forced_balance",  // カリキュラム維持
  "enable_pan": true,                    // PAN有効
  "enable_probes": true,                 // Probes有効
  "enable_lagrange": true                // Lagrange有効
}
```

---

## 📋 期待される効果

### 短期目標（5000-10000 steps）
- ✅ SELL rate: 1.6% → **15%以上**
- ✅ Lambda: 上限に張り付かない（< 10.0）
- ✅ エントロピー: 0.68 → **0.8以上**

### 中期目標（30000 steps）
- ✅ SELL rate: **30-35%** （理想の33%前後）
- ✅ HOLD/BUY/SELL: **均等分布（33/33/33）**
- ✅ Lambda: 安定（1.0-5.0範囲内）

---

## 🧪 検証手順

### 1. 短期学習セッション実行
```bash
# 10000 stepsで効果を確認
python run_training.py --config archived/configs/ppo_legacy/training/ppo_balanced_mem_optimized.json --timesteps 10000
```

### 2. 確認すべきメトリクス
- **SELL rate:** 15%以上に改善されているか？
- **Lambda値:** 上限（20.0）に張り付いていないか？
- **エントロピー:** 0.8以上に改善されているか？
- **各アクションのadvantage:** SELLが極端にマイナスになっていないか？

### 3. ログで確認すべき項目
```
Final Lagrange Statistics:
  SELL Rate (avg): ??%  ← 15%以上が目標
  Lambda (final): ??    ← 20.0未満が目標
  Constraint Active: True
```

### 4. デバッグ出力の追加（推奨）
以下の値をログに出力して詳細分析:
- SELL時の実際の報酬値
- アクション別のadvantage値
- min_holding_periodによるブロック頻度
- 各ペナルティの寄与度

---

## 🔄 次のステップ

### ステップ1: 短期検証（今すぐ実行）
```bash
python run_training.py --config archived/configs/ppo_legacy/training/ppo_balanced_mem_optimized.json --timesteps 10000
```

### ステップ2: 結果分析
```bash
# ログを分析ツールに入力
python scripts/analyze_training_logs_v2.py <ログファイル>
```

### ステップ3: 段階的な調整
**もしSELL率が改善されたら:**
1. ペナルティを段階的に復活
   - `trade_frequency_penalty: 0.0 → 0.005`
   - `consecutive_trade_penalty: 0.0 → 0.02`
2. SELL bonus を微調整
   - `profit_bonus_multipliers: [1.0, 1.0, 2.0] → [1.0, 1.0, 1.5]`
3. エントロピー係数を徐々に下げる
   - `ent_coef: 0.5 → 0.3 → 0.2`

**もしSELL率が改善されなかったら:**
1. Lambda上限をさらに引き上げ
   - `lagrange_lambda_max: 20.0 → 50.0`
2. SELL bonusをさらに増加
   - `profit_bonus_multipliers: [1.0, 1.0, 2.0] → [1.0, 1.0, 3.0]`
3. 環境設定を確認
   - `min_holding_period`を1に削減
   - データセットのSELL時PnLを調査

---

## 📊 適用前後の設定比較

| 項目 | Before | After | 変更率 |
|------|--------|-------|--------|
| `lagrange_eta` | 0.05 | 0.1 | **+100%** |
| `lagrange_lambda_max` | 2.0 | 20.0 | **+900%** |
| `lagrange_warmup_steps` | 1000 | 500 | -50% |
| `ent_coef` | 0.1 | 0.5 | **+400%** |
| `profit_bonus_multipliers[2]` | 1.0 | 2.0 | **+100%** |
| `action_penalty_scale` | 0.01 | 0.001 | **-90%** |
| `trade_frequency_penalty` | 0.01 | 0.0 | **-100%** |
| `consecutive_trade_penalty` | 0.05 | 0.0 | **-100%** |
| `enable_stratified_sampling` | false | true | **NEW** |

---

## ⚠️ 注意事項

### 1. 過度な矯正のリスク
- 逆にSELL biasになる可能性あり
- 10000 stepsごとに分布を確認し、調整が必要

### 2. 学習の不安定化
- エントロピー係数0.5は非常に高い値
- 学習が発散する可能性に注意
- loss値、policy gradient normを監視

### 3. ペナルティの完全無効化
- 一時的な措置（SELL率改善まで）
- 改善後は段階的に復活させる必要あり

### 4. 本番デプロイ前の再調整
- 現在の設定は**診断用の極端な設定**
- SELL率が改善したら、バランスの取れた設定に戻す
- 本番デプロイ前に必ず再検証

---

## 📝 ログ監視コマンド

### リアルタイム監視
```bash
# 学習中のログをリアルタイム監視
Get-Content -Path "training.log" -Wait | Select-String "SELL Rate"
```

### TensorBoard起動
```bash
tensorboard --logdir tensorboard
```

確認すべきグラフ:
- `rollout/action_dist_sell`
- `lagrange/lambda`
- `lagrange/sell_rate_error`
- `diversity/action_entropy`

---

## 🎯 成功基準

### 最低限の成功基準（10000 steps）
- ✅ SELL rate ≥ 15%
- ✅ Lambda < 15.0（上限に張り付いていない）
- ✅ エントロピー ≥ 0.8

### 理想的な成功基準（30000 steps）
- ✅ SELL rate: 30-35%
- ✅ HOLD rate: 30-35%
- ✅ BUY rate: 30-35%
- ✅ Lambda: 1.0-5.0（安定）
- ✅ エントロピー ≥ 0.95（ほぼ均等分布）

---

**次のアクション:** 短期学習セッション（10000 steps）を実行し、SELL率の改善を確認してください。

**実行コマンド:**
```bash
python run_training.py --config archived/configs/ppo_legacy/training/ppo_balanced_mem_optimized.json --timesteps 10000
```
