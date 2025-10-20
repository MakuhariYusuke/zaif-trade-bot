# v379+v380+Optimized 統合実行ガイド

## 📋 概要

v379の市場適応性、v380の積極性、二分探索最適化の結果を統合した新しい報酬関数設定を作成しました。

## 🎯 作成した設定ファイル

### 1. **v381 Hybrid Optimized** (推奨スタート地点)
- **ファイル**: `configs/training/ppo_reward_v381_hybrid_optimized.json`
- **特徴**: バランス型 - v380の66%強度 + v379の市場適応
- **期待HOLD率**: 35-45%
- **リスク**: Medium
- **用途**: まずこれから試す

### 2. **v382 Aggressive Optimized** (v381で不足時)
- **ファイル**: `configs/training/ppo_reward_v382_aggressive_optimized.json`
- **特徴**: 積極型 - v380の85%強度 + v379の市場適応
- **期待HOLD率**: 30-40%
- **リスク**: Medium-High
- **用途**: v381でHOLD率が45-50%だった場合に使用

### 3. **v381 30k Test** (クイックテスト用)
- **ファイル**: `configs/training/ppo_reward_v381_hybrid_optimized_30k.json`
- **特徴**: v381の短期テスト版
- **timesteps**: 30,000
- **用途**: 設定の動作確認

## 🔧 統合された要素

### v379 (市場適応性) からの採用要素
```python
reward_settings = {
    # 動的市場適応型機能
    "volatility_adjusted_penalty": True,      # 高ボラティリティ時にHOLDペナルティ増加
    "trend_adjusted_bonus": True,             # 強いトレンド時に取引ボーナス増加
    "range_market_hold_tolerance": 0.5,       # レンジ相場時にHOLDペナルティ軽減
}
```

### v380 (積極性) からの採用要素 (調整済み)
```python
# v381では66%、v382では85%の強度で採用
reward_settings = {
    "hold_penalty_weight": 0.035,             # v380: 0.05 → v381: 0.035 (66%)
    "consecutive_hold_penalty": 0.02,         # v380: 0.03 → v381: 0.02 (66%)
    "trading_frequency_bonus": 0.2,           # v380: 0.3 → v381: 0.2 (66%)
    "profit_reward_multiplier": 3.5,          # v380: 5.0 → v381: 3.5 (70%)
    "successful_trade_bonus": 0.7,            # v380: 1.0 → v381: 0.7 (70%)
}
```

### 二分探索最適化 (ppo_100k_optimized) からの採用要素
```python
ppo = {
    # 高信頼度パラメータ
    "learning_rate": 0.007503,      # +37.36pt改善
    "batch_size": 256,               # +27.71pt改善
    "max_grad_norm": 5.05,          # +21.89pt改善
    "n_steps": 1024,                # +11pt改善
    "vf_coef": 0.1,                 # +1pt改善
    
    # 中信頼度パラメータ
    "gamma": 0.8475,                # +1-2pt改善
    "n_epochs": 16,                 # <1pt改善
    "gae_lambda": 0.8,
    "clip_range": 0.1,
    "ent_coef": 0.001,
    "target_kl": 0.001,
}

lagrange = {
    "r_target": 0.175,              # +8.0pt改善
    "tolerance": 0.042625,          # +4.7pt改善
    "eta": 0.062875,                # +9.1pt改善
    "lambda_max": 3.875,            # +5.6pt改善 (3.8755超えでSELL暴走)
    "warmup_steps": 3874,           # +6.1pt改善
}
```

## 🚀 実行手順

### Step 1: 短期テストで動作確認 (30k)

```bash
# v381のクイックテスト (約10-15分)
python run_training.py --config configs/training/ppo_reward_v381_hybrid_optimized_30k.json
```

**確認ポイント**:
- ✅ エラーなく実行完了するか
- ✅ アクション分布が極端に偏っていないか
- ✅ TensorBoardで学習曲線が安定しているか

### Step 2: 100kフルトレーニング (v381から開始)

```bash
# v381ハイブリッド最適化版 (約30-45分)
python run_training.py --config configs/training/ppo_reward_v381_hybrid_optimized.json
```

**監視ポイント**:
- 📊 **目標アクション分布**: HOLD 35-45%, BUY 30-35%, SELL 25-30%
- 📈 **平均報酬**: -300以上を目指す
- 🎯 **KL divergence**: 安定していること (頻繁な早期停止は問題)

### Step 3: 結果評価とv382への移行判断

```bash
# TensorBoardでログ確認
tensorboard --logdir tensorboard_logs/ppo_reward_v381_hybrid_optimized
```

**v382への移行基準**:
- ❌ HOLD率が45-50%超 → v382を試す
- ✅ HOLD率が35-45% → v381を継続
- ⚠️ SELL率が35%超 → lambda_maxを3.5に削減

```bash
# 必要な場合のみv382実行
python run_training.py --config configs/training/ppo_reward_v382_aggressive_optimized.json
```

## 📊 パラメータ比較表

| 項目 | v379 | v380 | v381 (推奨) | v382 (積極) |
|------|------|------|-------------|-------------|
| **HOLD penalty** | 0.02 | 0.05 | 0.035 | 0.0425 |
| **Profit multiplier** | 3.0 | 5.0 | 3.5 | 4.25 |
| **Trading bonus** | 0.15 | 0.3 | 0.2 | 0.255 |
| **市場適応** | ✅ Full | ❌ None | ✅ Full | ✅ Full |
| **PPO最適化** | ❌ | ❌ | ✅ | ✅ |
| **Lagrange最適化** | ❌ | ❌ | ✅ | ✅ |
| **期待HOLD率** | 40-50% | 30-40% | 35-45% | 30-40% |
| **リスク** | Medium-High | High | **Medium** | Medium-High |
| **v380強度** | - | 100% | 66% | 85% |

## 🎛️ 微調整ガイド

### HOLDが多すぎる場合 (50%超)
```bash
# 設定ファイル内で調整
"hold_penalty_weight": 0.035 → 0.04 → 0.045
"consecutive_hold_penalty": 0.02 → 0.025 → 0.03
```

### SELLが多すぎる場合 (35%超)
```bash
# Lagrange制約を緩和
"lambda_max": 3.875 → 3.5 → 3.0
"r_target": 0.175 → 0.15 → 0.125
```

### 取引が少なすぎる場合
```bash
"trading_frequency_bonus": 0.2 → 0.25 → 0.3
"profit_reward_multiplier": 3.5 → 4.0 → 4.5
```

### 利益率が低い場合
```bash
"profit_reward_multiplier": 3.5 → 4.0
"successful_trade_bonus": 0.7 → 0.85 → 1.0
```

## 📈 期待される改善効果

### 二分探索最適化の累積効果
- **batch_size**: +27.71pt
- **learning_rate**: +37.36pt
- **max_grad_norm**: +21.89pt
- **n_steps**: +11pt
- **Lagrange params**: +8.0 + 4.7 + 9.1 + 5.6 + 6.1 = +33.5pt
- **合計**: 約+117pt (推定)

### v379+v380統合の効果
- **市場適応性**: ボラティリティ/トレンド/レンジに応じた柔軟な報酬調整
- **積極性**: HOLD率を35-45%に抑えつつ過剰なSELL暴走を防止
- **安定性**: Lagrange制約でバランスの取れたアクション分布

## ⚠️ 注意事項

### 1. lambda_maxの上限
- **絶対に3.8755を超えないこと**
- 3.875でもSELL暴走のリスクあり
- SELL率が35%超えたら即座に3.5以下に削減

### 2. 段階的アプローチ
1. **まずv381** (66%強度) から開始
2. 不足があれば**v382** (85%強度) へ
3. それでも不足なら**個別パラメータ調整**

### 3. モニタリング必須項目
- ✅ アクション分布 (各ステップでログ確認)
- ✅ 平均報酬の推移
- ✅ KL divergenceの安定性
- ✅ SELL率の暴走兆候

## 📁 関連ドキュメント

- `docs/LAGRANGE_OPTIMIZATION_RESULTS.md` - Lagrangeパラメータ最適化詳細
- `docs/binary_search_optimization/BINARY_SEARCH_COMPREHENSIVE_RESULTS.md` - PPOハイパーパラメータ最適化詳細
- `configs/training/ppo_reward_v379_dynamic.json` - v379の詳細
- `configs/training/ppo_reward_v380_aggressive.json` - v380の詳細
- `configs/training/ppo_100k_optimized.json` - 二分探索最適化結果

## 🎯 成功基準

### 最低限達成すべき目標
- ✅ HOLD率 < 50%
- ✅ SELL率 < 35%
- ✅ 平均報酬 > -350
- ✅ 安定した学習曲線

### 理想的な結果
- 🎯 HOLD率 35-45%
- 🎯 BUY率 30-35%
- 🎯 SELL率 25-30%
- 🎯 平均報酬 > -300
- 🎯 Sharpe Ratio > 0.5

## 🔄 イテレーション戦略

```
v381 (30k test)
    ↓
結果評価
    ↓
┌─────────┴─────────┐
│                   │
HOLD 35-45%     HOLD 45-50%+
    ↓               ↓
v381 (100k)     v382 (30k test)
    ↓               ↓
  完了          結果評価
                    ↓
                v382 (100k)
                    ↓
                  完了
```

このガイドに従って、段階的に最適な設定を見つけていくことをお勧めします。
