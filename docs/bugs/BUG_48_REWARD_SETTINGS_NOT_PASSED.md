# 🐛 重大バグ発見・修正レポート

**日時**: 2025-10-08 12:15
**バグID**: Bug #48
**重要度**: **CRITICAL** 🔴

---

## 📋 バグの概要

### **症状**
- v3.6.3緊急修正(Lambda=30.0, SELL報酬3倍)を適用してもSELL率が改善せず
- むしろ悪化: 10.7% → 8.8%
- Lambda=30.0到達、ペナルティ強化されているのに効果なし

### **根本原因**
**`reward_settings`が環境に渡されていなかった** ⚠️

`ztb/training/sell_mitigation_ppo_trainer.py`の`env_config`構築時に、設定ファイルの`reward_settings`が含まれていませんでした。

---

## 🔍 バグの詳細

### **問題のコード箇所**
**ファイル**: `ztb/training/sell_mitigation_ppo_trainer.py:281-288`

```python
# ❌ バグのあるコード
env_config = {
    "curriculum_stage": self.config.get("curriculum_stage", "full"),
    "allow_reverse": self.allow_reverse,
    "transaction_cost": self.config.get("transaction_cost", 0.001),
    "max_position_size": self.config.get("max_position_size", 1.0),
    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
    "reward_scaling": self.config.get("reward_scaling", 1.0),
    # ★ reward_settings が欠落！
}
```

### **影響範囲**
1. **profit_bonus_multipliers**: SELL報酬3倍が適用されず
2. **action_penalty_scale**: ペナルティ撤廃が無効
3. **その他すべてのreward_settings**: 一切反映されず

結果として、設定ファイルに記述した報酬調整が**完全に無視**されていました。

---

## ✅ 修正内容

### **修正後のコード**
```python
# ✅ 修正版
env_config = {
    "curriculum_stage": self.config.get("curriculum_stage", "full"),
    "allow_reverse": self.allow_reverse,
    "transaction_cost": self.config.get("transaction_cost", 0.001),
    "max_position_size": self.config.get("max_position_size", 1.0),
    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
    "reward_scaling": self.config.get("reward_scaling", 1.0),
    # ★ BUG FIX: Pass reward_settings from config to environment
    "reward_settings": self.config.get("reward_settings", {}),
}
```

### **修正の効果**
- ✅ `profit_bonus_multipliers: [1.0, 1.0, 3.0]` が環境に渡される
- ✅ すべてのペナルティ撤廃(`action_penalty_scale: 0.0`など)が有効化
- ✅ カスタム報酬設定が正しく適用される

---

## 🎯 期待される改善

### **v3.6.3設定の実際の適用**
| 設定項目 | 設定値 | 以前の状態 | 修正後 |
|----------|--------|------------|--------|
| **profit_bonus_multipliers[2]** | 3.0 | ❌ 未適用 | ✅ 適用 |
| **action_penalty_scale** | 0.0 | ❌ デフォルト値 | ✅ 0.0 |
| **position_penalty_scale** | 0.0 | ❌ デフォルト値 | ✅ 0.0 |
| **volatility_penalty_scale** | 0.0 | ❌ デフォルト値 | ✅ 0.0 |

### **予測されるSELL率の改善**
- **修正前**: 8.8% (reward_settings無効のため)
- **修正後(予測)**: **15-20%** (SELL報酬3倍+ペナルティ撤廃の効果)

---

## 🧪 検証計画

### **再検証の実行**
```bash
python run_training.py --config archived/configs/ppo_legacy/training/ppo_balanced_mem_optimized.json --timesteps 10000 --force
```

### **確認ポイント**
1. ✅ SELL率が15%以上に到達するか
2. ✅ Lambda値の推移(上限30.0未満で安定するか)
3. ⚠️ 逆バイアス(SELL率>40%)の発生有無

---

## 📊 過去の検証結果との比較

| 検証回 | Lambda上限 | SELL報酬 | ペナルティ | SELL率 | reward_settings |
|--------|-----------|---------|-----------|--------|----------------|
| **v3.6.1** | 20.0 | 2x | 微小 | 10.7% | ❌ 無効 |
| **v3.6.3(バグ有)** | 30.0 | 3x | 撤廃 | 8.8% | ❌ **無効** |
| **v3.6.3(修正後)** | 30.0 | 3x | 撤廃 | **?%** | ✅ **有効** |

---

## 🔧 その他の影響調査

### **他のトレーナーも同様のバグがないか確認が必要**
- `ppo_trainer.py`
- `curriculum_learning.py`
- `ensemble.py`

これらのファイルでも`env_config`構築時に`reward_settings`が渡されているか要確認。

---

## 📝 学んだ教訓

1. **設定の伝播チェーンを追跡する重要性**
   - Config → UnifiedTrainer → PPOTrainer → env_config → Environment
   - 途中で欠落するリスク

2. **デバッグ時はログで実際の設定値を出力**
   - `logger.info(f"env_config: {env_config}")`
   - `logger.info(f"reward_settings: {self.reward_settings}")`

3. **単体テストでconfigの伝播を検証**
   - `test_reward_settings_propagation.py` を作成すべき

---

**修正ファイル**: `ztb/training/sell_mitigation_ppo_trainer.py` (1行追加)
**ステータス**: ✅ 修正完了 → 再検証待ち
**次のアクション**: 10,000ステップ検証実行 → SELL率15%達成を確認
