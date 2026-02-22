# 85# Day 11: 評価基盤修正後の再実験

**作成日**: 2025-01-XX  
**カテゴリ**: Phase 4 - 包括実験（修正版）

---

## 1. 背景と目的

### 1.1 84#で特定した問題

| 問題 | 根本原因 | 影響 |
|------|----------|------|
| final_balance常にNone | UnifiedTrainerは`trainer.model.env`ではなく`trainer.algorithm_trainer.model.env` | ROI=final_reward×100にフォールバック |
| ROI=-36% vs -5% | 45#はwalk-forward有効、Day10は無効 | 31%の乖離 |
| 環境属性名不一致 | `portfolio_value`ではなく`balance` | final_balance取得失敗 |

### 1.2 Day11の目的

1. **45# Day5設定の再現確認** - ROI=-5%が再現できるか
2. **walk-forward影響の定量化** - 有効/無効でどれだけ変わるか
3. **修正版環境アクセスの検証** - final_balance取得成功率

---

## 2. 実験設計

### 2.1 実験カテゴリ

| カテゴリ | 条件 | 目的 | 期待ROI |
|----------|------|------|---------|
| A | walk-forward有効, 50k | 45# Day5再現 | -5% ± 3% |
| B | walk-forward無効, 50k | Day10条件比較 | -36%付近? |
| C | walk-forward無効, 25k | 崩壊前確認 | -5%付近? |

### 2.2 固定設定（45# Day5と同一）

```python
SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 100000,
    "batch_size": 256,
    "gamma": 0.99,
    "gradient_steps": 1,
    "ent_coef": "auto",
}
```

### 2.3 シード

- seed=42, 123（2シードで統計検定）

### 2.4 合計実験数

- 3カテゴリ × 2シード = **6実験**
- 推定時間: 約60分

---

## 3. 修正内容

### 3.1 SACTrainer直接使用

84#で判明した問題を回避するため、UnifiedTrainerではなくSACTrainerを直接使用：

```python
# 45# run_ab_feature_test.pyと同じ方式
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
trainer = SACTrainer(config=config, logger=logger)
result = trainer.train()
```

### 3.2 環境アクセス修正

```python
# SACTrainerの場合
if hasattr(trainer, 'model') and trainer.model is not None:
    env = trainer.model.env
    
# 属性優先順位
if hasattr(unwrapped_env, 'balance'):        # 優先
    final_balance = unwrapped_env.balance
elif hasattr(unwrapped_env, 'portfolio_value'):  # フォールバック
    final_balance = unwrapped_env.portfolio_value
```

### 3.3 ROI計算

```python
# balance-basedで計算（Day10のreward×100ではない）
roi = (final_balance - initial_balance) / initial_balance * 100
```

---

## 4. 実行方法

```powershell
cd c:\Users\Admin\dev\zaif-trade-bot
python scripts/v459/run_day11_verification.py
```

出力先: `results/phase4_day11_verification/`

---

## 5. 検証基準

### 5.1 成功判定

| 項目 | 基準 |
|------|------|
| 45#再現 | A_wf_enabled ROI = -5% ± 5% |
| walk-forward影響 | A - B の差が統計的に有意 |
| balance取得 | 6/6実験でfinal_balance取得成功 |

### 5.2 解釈パターン

| A ROI | B ROI | 解釈 |
|-------|-------|------|
| -5% | -36% | walk-forwardが安定化に寄与 |
| -5% | -5% | Day10の問題は他にある |
| -30% | -30% | 45#からの環境変化あり |

---

## 6. 次のステップ

1. **A,B同等** → Day10 C/Dカテゴリの再実験（gamma, ent_coef）
2. **A >> B** → 全実験をwalk-forward有効で再実施
3. **両方不調** → 45# run_ab_feature_test.pyの環境との差分調査

---

## 7. 参考値

| 実験 | ROI | 条件 |
|------|-----|------|
| 45# Day5 | -5.07% | walk-forward有効, 50k |
| Day10 A1 | -36.04% | walk-forward無効, 50k |
| Day10 C1 | -5.71% | walk-forward無効, 50k, gamma=0.95 |
