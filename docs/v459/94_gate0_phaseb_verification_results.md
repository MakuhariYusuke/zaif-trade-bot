# 94# Gate 0 + Phase B 検証結果

## 検証日時
2026-02-02 03:42 - 04:10

## 経緯

### 1. Gate 0 (設定伝播検証) の発見した問題

**初回実行（修正前）：**
```
EXPECTED: alpha=0.0, beta=0.0, edge_penalty_rate=0.0, ...
ACTUAL:   alpha=NOT_FOUND, beta=NOT_FOUND, ...
STATUS: ❌ MISMATCH - Settings may not be applied correctly!
```

**根本原因:**
P1実験で設定した `alpha, beta, gamma, fee_penalty_weight` などは、**現在の `RewardSettings` dataclass に存在しないフィールド**だった。

これは v444 のconfig伝播バグと同様の問題パターン：
- 設定を指定したつもりでも、**フィールド名が不一致で無視**されていた
- Gate 0 がこの問題を検出し、修正を可能にした

### 2. 修正内容

**正しいパラメータ名に修正:**
```python
# Before (存在しないフィールド)
REWARD_PARAMS_P1_1 = {
    "alpha": 0.0,              # ❌ NOT_FOUND
    "beta": 0.0,               # ❌ NOT_FOUND  
    "gamma": 0.0,              # ❌ NOT_FOUND
    ...
}

# After (RewardSettingsの実際のフィールド)
REWARD_PARAMS_P1_1 = {
    "balance_penalty": 0.0,              # ✅ FOUND
    "position_penalty_scale": 0.0,       # ✅ FOUND
    "inventory_penalty_scale": 0.0,      # ✅ FOUND
    "trade_frequency_penalty": 0.0,      # ✅ FOUND
    "consecutive_trade_penalty": 0.0,    # ✅ FOUND
    ...
}
```

**Config伝播経路も修正:**
```python
# reward_settings を training.environment.reward_settings に配置
config = {
    "training": {
        "environment": {
            "reward_settings": reward_params.copy(),  # ← 追加
        },
    },
    "reward": reward_params,  # 検証用
}
```

### 3. Gate 0 成功確認

**修正後の検証：**
```
REWARD PARAMS VERIFICATION ==========
EXPECTED: balance_penalty=0.0, balance_penalty_tolerance=1.0, consecutive_trade_penalty=0.0, 
          consistency_penalty=0.0, hold_penalty_multiplier=0.0, inventory_penalty_scale=0.0, 
          position_penalty_exponent=1.0, position_penalty_scale=0.0, profit_weight=1.0, 
          redundant_trade_penalty=0.0, reward_scale=100.0, trade_cooldown_penalty=0.0, 
          trade_frequency_penalty=0.0, volatility_penalty_scale=0.0
ACTUAL:   balance_penalty=0.0, balance_penalty_tolerance=1.0, consecutive_trade_penalty=0.0, 
          consistency_penalty=0.0, hold_penalty_multiplier=0.0, inventory_penalty_scale=0.0, 
          position_penalty_exponent=1.0, position_penalty_scale=0.0, profit_weight=1.0, 
          redundant_trade_penalty=0.0, reward_scale=100.0, trade_cooldown_penalty=0.0, 
          trade_frequency_penalty=0.0, volatility_penalty_scale=0.0
STATUS: ✅ MATCH - Settings correctly applied
```

**Gate 0 完全成功！**

---

## Phase B: コスト分解結果

### P1-1 vs P1-3 比較表

| 項目 | P1-1 (ペナルティ無効) | P1-3 (デフォルト) | 差分 |
|------|---------------------|------------------|------|
| **Gross PnL** | **+121 JPY (+0.12%)** | **-32 JPY (-0.03%)** | **+153 JPY (+0.15%)** |
| Total Fees | -5,147 JPY (-5.15%) | -4,991 JPY (-4.99%) | -156 JPY |
| Total Slippage | 0 JPY | 0 JPY | 0 |
| **Net PnL** | -5,027 JPY (-5.03%) | -5,023 JPY (-5.02%) | -4 JPY |
| Cost Ratio | 4,256% | 15,771% | - |

### P1-1: PnLのみ（ペナルティ全無効）

| 項目 | 値 | 割合 |
|------|------|------|
| **Gross PnL（手数料前）** | **+121 JPY** | **+0.12%** |
| Total Fees | -5,147 JPY | -5.15% |
| Total Slippage | 0 JPY | 0.00% |
| **Net PnL** | **-5,027 JPY** | **-5.03%** |
| Cost Ratio | 4,256% | |

### P1-3: デフォルト設定

| 項目 | 値 | 割合 |
|------|------|------|
| **Gross PnL（手数料前）** | **-32 JPY** | **-0.03%** |
| Total Fees | -4,991 JPY | -4.99% |
| Total Slippage | 0 JPY | 0.00% |
| **Net PnL** | **-5,023 JPY** | **-5.02%** |
| Cost Ratio | 15,771% | |

### 解釈

```
P1-1: 取引自体は利益だがコストに負けている
P1-3: 取引自体が損失
```

**重大な発見：**
1. **P1-1（ペナルティ無効）のGross PnLは+0.12%でプラス** → 取引戦略自体は機能
2. **P1-3（デフォルト）のGross PnLは-0.03%でマイナス** → ペナルティが取引を歪めている
3. **手数料が両者とも Gross PnL を大幅に超過** → 過剰取引が共通問題
4. **ペナルティ無効化でGross PnLが+153 JPY改善** → 現行ペナルティは有害

---

## 結論と次のステップ

### 確定した事実
1. **取引戦略自体は機能している**（P1-1: Gross PnL > 0）
2. **現行のペナルティは有害**（P1-3: Gross PnL < 0）
3. **問題は過剰取引**（手数料が利益の42〜157倍）
4. **Gate 0 は正常動作**し、設定伝播問題を検出・修正に貢献
5. **Phase B コスト分解は成功**し、問題の本質を特定

### 改善仮説（優先度順）
1. **ペナルティ無効のまま取引頻度を下げる**（最優先）
   - Continuous action の閾値を上げる（|action| > 0.5, 0.7 でのみ取引）
   - これだけで Net PnL > 0 が実現可能な可能性

2. **v451 "Golden Era" 設定の再現**
   - Gamma=0.80, Hold Penalty=0
   - ペナルティ無効で既に類似設定

3. **v457.3 方式の適用**
   - アクション空間を簡略化（TTL固定）
   - 取引判断のみに集中

---

## 技術詳細

### 修正したファイル
1. `scripts/v459/run_phase45_p1.py` - パラメータ名とconfig構造
2. `ztb/training/unified_trainer/algorithms/sac_trainer.py` - ログフォーマット修正

### Gate 0 ログ追加位置
- `sac_trainer.py:_log_reward_params_verification()` - EXPECTED vs ACTUAL 比較

### Phase B コスト分解ログ追加位置
- `sac_trainer.py:_log_cost_breakdown()` - Gross/Fees/Slip/Net PnL

---

## 参照
- [93# 改訂版ピボット計画](93_revised_pivot_plan.md)
- [92# レビュー](92_pivot_plan_review.md) - 「設定反映検証が不足」の指摘
- v444 BALANCE_PENALTY_ROOT_CAUSE_FIX_FINAL.md - 設定伝播バグの教訓
- v457.2/v457.3 ログ - コスト問題のパターン
