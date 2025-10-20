# SELL回避緊急修正 v3.6.3 変更サマリー

**適用日時**: 2025-10-08
**対象ファイル**: `configs/training/ppo_balanced_mem_optimized.json`

---

## 🔧 適用した修正内容

### **1. Lagrange制約の強化 (+50%)**
```diff
- "lagrange_lambda_max": 20.0,
+ "lagrange_lambda_max": 30.0,

- "lagrange_eta": 0.1,
+ "lagrange_eta": 0.15,
```

**意図**: Lambda上限を30.0に引き上げ、制約をさらに強化。収束速度も向上。

---

### **2. SELL報酬の強化 (+50%)**
```diff
- "profit_bonus_multipliers": [1.0, 1.0, 2.0],
+ "profit_bonus_multipliers": [1.0, 1.0, 3.0],
```

**意図**: SELL時の報酬を3倍に増強し、学習インセンティブを最大化。

---

### **3. ペナルティの完全撤廃 ⚠️**
```diff
- "action_penalty_scale": 0.001,
+ "action_penalty_scale": 0.0,

- "position_penalty_scale": 0.1,
+ "position_penalty_scale": 0.0,

- "inventory_penalty_scale": 0.01,
+ "inventory_penalty_scale": 0.0,

- "volatility_penalty_scale": 0.01,
+ "volatility_penalty_scale": 0.0,
```

**意図**: すべてのペナルティを除去し、SELL行動への障壁をゼロに。

**⚠️ リスク**: 逆バイアス(過剰SELL)の可能性あり。検証で監視が必要。

---

## 📊 期待される効果

| 項目 | v3.6.1 (前回) | v3.6.3 (今回) | 期待値 |
|------|---------------|---------------|--------|
| **Lambda上限** | 20.0 | 30.0 (+50%) | より強い制約 |
| **SELL報酬** | 2x | 3x (+50%) | 学習速度向上 |
| **ペナルティ** | 微小 | 0.0 (撤廃) | 行動自由度↑ |
| **予測SELL率** | 10.7% | 14-17% | **15%目標達成** |

---

## 🎯 検証計画

### **実行コマンド**
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000 --force
```

### **成功基準**
- ✅ **SELL率 ≥ 15%**
- ✅ Lambda < 25.0 (上限未到達)
- ⚠️ **SELL率 ≤ 40%** (逆バイアス防止)

### **監視指標**
1. `lagrange_r_sell`: 各ステップのSELL率
2. `lagrange_lambda_dual`: Lambda値の推移
3. `pan_action_counts`: HOLD/BUY/SELL分布

---

## ⚠️ リスク管理

### **想定される逆バイアスシナリオ**
- **SELL率 > 40%**: 過剰なSELL偏重
- **BUY率 < 10%**: BUY行動の抑制

### **対処法**
- Lambda上限をさらに調整 (25.0など)
- SELL報酬を2.5倍に減衰
- 一部ペナルティを復活 (position_penalty_scaleなど)

---

**バージョン**: v3.6.3
**ステータス**: ⚠️ EXPERIMENTAL - 逆バイアス監視下
**次のアクション**: 10,000ステップ検証実行 → 結果分析
