# Bug #47, #48, #49 総合修正サマリー

**修正日時**: 2025-10-08
**対象バージョン**: v3.6.4

---

## 🐛 修正したバグ一覧

### **Bug #47: Magic Numbers (マジックナンバー撤廃)**
- **概要**: アクション定数の数値直接使用を撤廃
- **修正**: `ztb/trading/constants.py`に定数を定義し、全ファイルで使用

### **Bug #48: reward_settings伝播バグ**
- **概要**: `reward_settings`が環境に渡されていなかった
- **修正**:
  - `ztb/training/sell_mitigation_ppo_trainer.py`
  - `ztb/training/trainers/sell_mitigation_trainer.py`
  - 両ファイルの`env_config`に`"reward_settings": self.config.get("reward_settings", {})`を追加

### **Bug #49: profit_bonus_multipliers順序エラー** ⚠️ **最重要**
- **概要**: 配列順序を誤認し、SELLではなくHOLDを強化していた
- **影響**: v3.6.1〜v3.6.3の全検証でSELL強化が無効
- **修正**:
  - 配列順序を`[1.0, 1.0, 3.0]` → `[1.0, 3.0, 1.0]`に変更
  - マジックナンバーインデックスを定数化

---

## 📝 今回の修正内容

### **1. constants.pyに配列インデックス定数を追加**
```python
# Array indices for profit_bonus_multipliers [BUY, SELL, HOLD]
# CRITICAL: The order is [BUY, SELL, HOLD], NOT [HOLD, BUY, SELL]!
MULTIPLIER_INDEX_BUY = 0
MULTIPLIER_INDEX_SELL = 1
MULTIPLIER_INDEX_HOLD = 2
```

### **2. reward_calculator.pyでマジックナンバーを定数に置き換え**

#### 修正前:
```python
if action == ACTION_BUY:
    profit_bonus = base_profit_bonus * multipliers[0] * trend_multiplier
elif action == ACTION_SELL:
    profit_bonus = base_profit_bonus * multipliers[1] * trend_multiplier
else:  # HOLD
    profit_bonus = base_profit_bonus * multipliers[2] * trend_multiplier
```

#### 修正後:
```python
# profit_bonus_multipliers array order: [BUY, SELL, HOLD]
if action == ACTION_BUY:
    profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY] * trend_multiplier
elif action == ACTION_SELL:
    profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL] * trend_multiplier
else:  # HOLD
    profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD] * trend_multiplier
```

### **3. ppo_balanced_mem_optimized.jsonでSELLを正しく3倍に設定**

#### 修正前 (Bug #49):
```json
"profit_bonus_multipliers": [1.0, 1.0, 3.0]  // BUY, SELL, HOLD
                                              // 実際: BUY=1.0, SELL=1.0, HOLD=3.0
```

#### 修正後 (正しい):
```json
"profit_bonus_multipliers": [1.0, 3.0, 1.0]  // BUY, SELL, HOLD
                                              // BUY=1.0, SELL=3.0, HOLD=1.0
```

---

## 🎯 期待される効果

### **修正前の実際の設定 (v3.6.3)**
- BUY: 1.0倍 ✅
- SELL: 1.0倍 ❌ (強化なし)
- HOLD: 3.0倍 ❌ (意図せず強化 → SELL抑制)
→ **SELL率: 9.5%**

### **修正後の設定 (v3.6.4)**
- BUY: 1.0倍 ✅
- SELL: 3.0倍 ✅ (初めて正しく強化)
- HOLD: 1.0倍 ✅ (通常に戻る)
→ **予測SELL率: 15-25%**

---

## 📊 修正ファイル一覧

| ファイル | 修正内容 |
|---------|---------|
| `ztb/trading/constants.py` | 配列インデックス定数追加 |
| `ztb/trading/environment/components/reward_calculator.py` | マジックナンバー → 定数化 (3箇所) |
| `configs/training/ppo_balanced_mem_optimized.json` | 配列順序修正 + バージョン更新 |

---

## ✅ チェックリスト

- [x] Bug #47: マジックナンバー撤廃完了
- [x] Bug #48: reward_settings伝播修正完了 (2ファイル)
- [x] Bug #49: profit_bonus_multipliers順序修正完了
- [x] 配列インデックス定数化完了
- [x] コメント追加で順序を明示
- [x] 設定ファイルバージョン更新 (v3.6.4)
- [ ] **検証実行待ち**

---

## 🧪 次のアクション

### **検証コマンド**
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000 --force
```

### **期待される結果**
1. **SELL率 ≥ 15%** (初めて正しいSELL強化が適用)
2. **Lambda < 25.0** (制約が緩和される)
3. **HOLD率減少** (3倍インセンティブ削除の効果)

---

**修正バージョン**: v3.6.4
**ステータス**: ✅ 全修正完了 → 検証準備完了
**重要度**: CRITICAL - 過去3回の検証が全て無効だったため
