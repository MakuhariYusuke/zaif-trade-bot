# 🐛 Bug #49: profit_bonus_multipliers の順番エラー

**発見日時**: 2025-10-08 12:30  
**重要度**: **CRITICAL** 🔴🔴🔴

---

## 📋 問題の概要

### **症状**
- v3.6.3でSELL報酬を3倍に設定したはずが、SELL率が9.5%と低いまま
- Bug #48修正後も改善が微小(8.8% → 9.5%)
- Lambda=30.0、ペナルティ最大でも効果なし

### **根本原因**
**`profit_bonus_multipliers`の配列順序が間違っていた！** ⚠️⚠️⚠️

---

## 🔍 詳細分析

### **設定ファイルの記述 (間違い)**
```json
"profit_bonus_multipliers": [1.0, 1.0, 3.0]
```

### **意図した配列の意味**
開発者は以下のつもりで設定:
```
[HOLD, BUY, SELL] = [1.0, 1.0, 3.0]
```
→ SELL を 3倍に強化したい

### **実際の配列の定義 (正しい仕様)**
ソースコード (`curriculum_transition.py:51`, `analyze_sell_bias.py:51`) より:
```python
"reward_profit_bonus_multipliers": [1.1, 1.15, 0.8],  # BUY, SELL, HOLD
```

**正しい順番**: **[BUY, SELL, HOLD]**

### **実際に適用されていた値**
```
multipliers[0] = 1.0 → BUY用  (意図通り)
multipliers[1] = 1.0 → SELL用 (❌ 3.0であるべき!)
multipliers[2] = 3.0 → HOLD用 (❌ 1.0であるべき!)
```

**結果**: 
- ✅ BUY報酬: 1.0倍 (正常)
- ❌ **SELL報酬: 1.0倍** (3.0倍のつもりが無強化!)
- ❌ **HOLD報酬: 3.0倍** (意図せず強化!)

---

## 💥 影響範囲

### **v3.6.1 → v3.6.3 の全検証が無効**

| バージョン | 設定値 | BUY | SELL | HOLD | SELL率 |
|-----------|--------|-----|------|------|--------|
| **v3.6.1** | [1.0, 1.0, 2.0] | 1.0x | ❌ **1.0x** | ❌ **2.0x** | 10.7% |
| **v3.6.3** | [1.0, 1.0, 3.0] | 1.0x | ❌ **1.0x** | ❌ **3.0x** | 9.5% |

**衝撃的な事実**:
- v3.6.1では**HOLDを2倍**に強化していた
- v3.6.3では**HOLDを3倍**に強化していた
- SELLは一度も強化されていなかった！

### **なぜSELL率が低下したか**
- v3.6.1: HOLD 2倍 → HOLD有利
- v3.6.3: HOLD 3倍 → さらにHOLD有利
- **HOLDを強化するほど、SELLが抑制された**

---

## ✅ 正しい修正

### **修正前 (間違い)**
```json
"profit_bonus_multipliers": [1.0, 1.0, 3.0]  // 意図: HOLD, BUY, SELL
                                               // 実際: BUY, SELL, HOLD
```

### **修正後 (正しい)**
```json
"profit_bonus_multipliers": [1.0, 3.0, 1.0]  // BUY, SELL, HOLD
```

**これで本当にSELLが3倍になる！**

---

## 📊 予測される効果

### **修正前 (実際の設定)**
- BUY: 1.0x
- SELL: 1.0x (❌ 強化なし)
- HOLD: 3.0x (❌ 意図せず強化)
→ **HOLD有利 → SELL率低下**

### **修正後 (正しい設定)**
- BUY: 1.0x
- SELL: 3.0x (✅ 意図通り強化)
- HOLD: 1.0x (✅ 通常に戻る)
→ **SELL有利 → SELL率上昇**

**期待されるSELL率**: **15-25%** (HOLDの逆インセンティブが解消され、SELL強化が効く)

---

## 🧪 検証計画

### **緊急再検証の実行**
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000 --force
```

### **成功基準**
- ✅ SELL率 ≥ 15%
- ✅ Lambda < 25.0 (上限未到達)
- ✅ HOLD率が減少 (3倍インセンティブ削除の効果)

---

## 📝 学んだ教訓

### **1. 配列順序の明示が不足**
```python
# ❌ 曖昧
"profit_bonus_multipliers": [1.0, 3.0, 1.0]

# ✅ 明確
"profit_bonus_multipliers": {
    "buy": 1.0,
    "sell": 3.0,
    "hold": 1.0
}
```

### **2. 設定のバリデーション不足**
- 配列の順番が正しいかをチェックする機構がない
- コメントでの説明も不十分

### **3. 単体テストの欠如**
```python
def test_profit_bonus_multipliers_order():
    """profit_bonus_multipliersの順番が正しいことを確認"""
    config = {"profit_bonus_multipliers": [1.0, 3.0, 1.0]}
    calc = RewardCalculator(reward_settings=config)
    
    # SELLアクションで3倍の報酬が適用されるか
    assert calc.get_multiplier(ACTION_SELL) == 3.0
    assert calc.get_multiplier(ACTION_BUY) == 1.0
    assert calc.get_multiplier(ACTION_HOLD) == 1.0
```

---

## 🔧 追加の修正提案

### **設定ファイルを辞書形式に変更**
```json
"reward_settings": {
    "profit_bonus_multipliers": {
        "buy": 1.0,
        "sell": 3.0,
        "hold": 1.0
    }
}
```

### **RewardCalculatorで辞書をサポート**
```python
multipliers_raw = self.reward_settings.get("profit_bonus_multipliers", [1.0, 1.0, 1.0])

if isinstance(multipliers_raw, dict):
    # 辞書形式をサポート
    multipliers = [
        multipliers_raw.get("buy", 1.0),
        multipliers_raw.get("sell", 1.0),
        multipliers_raw.get("hold", 1.0),
    ]
elif isinstance(multipliers_raw, list):
    # 既存の配列形式もサポート (BUY, SELL, HOLD の順)
    multipliers = [float(x) if isinstance(x, (int, float)) else 1.0 for x in multipliers_raw[:3]]
    multipliers += [1.0] * (3 - len(multipliers))
```

---

## 📊 過去の検証結果の再解釈

| 検証回 | 設定(意図) | 設定(実際) | SELL率 | 解釈 |
|--------|-----------|-----------|--------|------|
| **v3.6.1** | SELL 2x | HOLD 2x | 10.7% | HOLD強化でSELL抑制 |
| **v3.6.3 (Bug #48前)** | SELL 3x | HOLD 3x | 8.8% | さらにHOLD強化 |
| **v3.6.3 (Bug #48後)** | SELL 3x | HOLD 3x | 9.5% | reward_settings有効化でわずか改善 |
| **v3.6.4 (今回修正)** | SELL 3x | **SELL 3x** | **予測:18-25%** | 初めて正しく適用 |

---

**修正ファイル**: `configs/training/ppo_balanced_mem_optimized.json` (1行修正)  
**ステータス**: 修正準備中 → 緊急再検証が必要  
**次のアクション**: 配列順序を修正 → 10,000ステップ検証 → SELL率15%達成確認
