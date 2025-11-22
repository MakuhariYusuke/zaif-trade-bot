# SAC v448 Configuration Files

## 📋 Overview

v448設定ファイル群は、1分足学習におけるバイアス崩壊問題を解決するために設計されています。

---

## 🗂️ ディレクトリ構造

```
config/v448/
├── sac_v448_emergency_fix.json          # 緊急修正版（最優先使用）
├── sac_v448_perfect_balance.json        # 完全均衡版（開発中）
├── sac_v448_curriculum.json             # Curriculum版（開発中）
├── sac_v448_trend_aware.json            # Trend-aware版（開発中）
├── templates/
│   └── v448_config_template.json        # テンプレート
└── README.md                             # このファイル
```

---

## 🚨 Emergency Fix版（sac_v448_emergency_fix.json）

### 目的
バイアス崩壊（BUY>90%またはSELL>90%）の完全防止

### 主な変更点（v447比）

| 項目 | v447 | v448 Emergency | 理由 |
|------|------|----------------|------|
| **Action Bonuses** | BUY=0.02 | **全て0.00** | 累積バイアスの主犯 |
| **Asymmetric Scaling** | Long=1.05 | **全て1.00** | BUY偏重の助長を排除 |
| **Balance Targets** | BUY=40%, SELL=30%, HOLD=30% | **BUY=47.5%, SELL=47.5%, HOLD=5%** | 実績ベース |
| **Balance Penalty** | 5.0 | **8.0** | 強化 |
| **Forced Balance Min** | 10 | **100** | 1分足適応 |
| **Forced Balance Threshold** | 0.15 | **0.08** | 早期介入 |
| **Emergency Penalty** | なし | **500.0** | 新規（>30%偏差時） |
| **Entropy Coefficient** | 0.01 | **0.05** | 探索強化 |
| **MTF Weights** | 1min=60%, 5min=40% | **1min=30%, 5min=55%, 15min=15%** | ノイズ抑制 |

### 使用方法

```bash
# 単独テスト（1000 steps × 3 seeds）
python tools/training/ab_test_runner.py \
  --configs config/v448/sac_v448_emergency_fix.json \
  --seeds 3 \
  --timesteps 1000 \
  --name "v448_emergency_test"

# 結果確認
python tools/analysis/analyze_recent_reports.py --filter "v448_emergency"
```

### 成功基準

- ✅ バイアス崩壊 0件（BUY<90%, SELL<90%）
- ✅ BUY-SELL差 < 25%
- ✅ Final Reward > -5.0

---

## 🔧 設定ファイルの作成方法

### テンプレートからの作成

```bash
# テンプレートをコピー
cp config/v448/templates/v448_config_template.json config/v448/my_config.json

# 編集
code config/v448/my_config.json
```

### 重要なパラメータ

#### 1. Action Bonuses（バイアス源）
```json
"action_bonuses": {
  "buy_action_bonus": 0.00,   // ⚠️ 0.00推奨（バイアス防止）
  "sell_action_bonus": 0.00,
  "hold_action_bonus": 0.00
}
```

#### 2. Balance Targets（目標分布）
```json
"balance_penalty_targets": {
  "buy_target": 0.475,   // 47.5%（ほぼ50%）
  "sell_target": 0.475,  // 47.5%
  "hold_target": 0.05    // 5%（取引活性化）
}
```

#### 3. Forced Balance（強制均衡）
```json
"forced_balance_min_actions": 100,         // 1分足では100以上推奨
"forced_balance_threshold": 0.08,          // 8%偏差で介入
"forced_balance_emergency_penalty": 500.0  // 緊急ペナルティ
```

#### 4. Multi-Timeframe Weights（ノイズ対策）
```json
"multi_timeframe": {
  "feature_weights": {
    "1min": 0.30,   // ノイズ多→低重み
    "5min": 0.55,   // バランス良→高重み
    "15min": 0.15   // トレンド確認用
  }
}
```

---

## 📊 バージョン比較

| 設定 | 目的 | バイアス対策 | 収益性 | 開発状況 |
|------|------|------------|--------|---------|
| Emergency Fix | バイアス崩壊防止 | 最強 | 中 | ✅ 完成 |
| Perfect Balance | 完全均衡 | 強 | 中〜高 | 🚧 開発中 |
| Curriculum | 段階的学習 | 強 | 高 | 🚧 開発中 |
| Trend Aware | 市場適応 | 中 | 最高 | 📝 計画中 |

---

## 🧪 実験ガイドライン

### Phase 0: Emergency Fix検証（Day 1-4）

```bash
# 1. 短期テスト
python tools/training/ab_test_runner.py \
  --configs config/v448/sac_v448_emergency_fix.json \
  --seeds 3 \
  --timesteps 1000

# 2. 中期テスト（成功時）
python tools/training/ab_test_runner.py \
  --configs config/v448/sac_v448_emergency_fix.json \
  --seeds 5 \
  --timesteps 3000

# 3. 分析
python tools/analysis/analyze_recent_reports.py
```

### Phase 1: 他バージョン開発（Day 5-12）

Emergency Fixで基本動作確認後、他バージョンを開発

---

## ⚠️ 注意事項

### 絶対に避けるべき設定

```json
// ❌ Action bonusを不均等に設定
"action_bonuses": {
  "buy_action_bonus": 0.02,  // ❌ バイアス崩壊の原因
  "sell_action_bonus": 0.00
}

// ❌ Asymmetric scalingを偏らせる
"asymmetric_reward_scaling": {
  "long_position_reward_multiplier": 1.12,  // ❌ BUY優遇
  "short_position_reward_multiplier": 0.92
}

// ❌ Forced balanceを甘くする
"forced_balance_threshold": 0.20,  // ❌ 20%は遅すぎる（0.08推奨）
"forced_balance_min_actions": 10   // ❌ 10は早すぎる（100推奨）
```

### デバッグ時の確認ポイント

1. **Action distribution**: BUY-SELL差が30%超えたら要注意
2. **Final reward**: -9.0付近は完全崩壊のサイン
3. **Training時間**: 異常に短い場合、早期収束の可能性

---

## 📚 関連ドキュメント

- `docs/SAC_v448_DEVELOPMENT_PLAN.md` - 開発計画全体
- `docs/SAC_v448_IMPLEMENTATION_ROADMAP.md` - 実装ロードマップ
- `docs/current/BALANCE_EXPLORATION_AND_MEMORY_OPTIMIZATION.md` - 分析詳細

---

**Version**: 1.0  
**Created**: 2025-11-21  
**Status**: Emergency Fix版のみ完成、他は開発中
