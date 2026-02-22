# Phase 4 Day 6 時間短縮方策の検討

## 現状分析

### 実測値（テスト実行より）
- 1実験所要時間: **43.6分**
- 全10実験（5 configs × 2 seeds）: **7.2時間**（436分）
- 設定: total_timesteps=50,000, SAC default設定

### ボトルネック分析
```python
# 時間配分（推定）
学習ループ: ~95%（41.4分）
  - Experience収集: ~60%（26分）
  - Gradient更新: ~30%（13分）
  - Logging/Evaluation: ~5%（2.2分）
データロード: ~3%（1.3分）
初期化: ~2%（0.9分）
```

---

## 時間短縮方策の比較

### 方策1: total_timesteps削減

**案A: 50,000 → 30,000（-40%）**
```yaml
メリット:
  - 時間: 43.6分 → 26.2分（-40%）
  - 全体: 7.2時間 → 4.3時間
  - 学習はある程度収束（Day 5でも50,000で収束傾向）

デメリット:
  - 学習不足の可能性（特に実験D, E）
  - 0番との整合性（50,000は既定値）

リスク: 中
推奨度: ★★★☆☆
```

**案B: 50,000 → 25,000（-50%）**
```yaml
メリット:
  - 時間: 43.6分 → 21.8分（-50%）
  - 全体: 7.2時間 → 3.6時間

デメリット:
  - 学習不足のリスク高
  - 報酬調整効果が見えない可能性

リスク: 高
推奨度: ★★☆☆☆
```

---

### 方策2: SAC設定最適化

**案C: buffer_size削減**
```python
SAC_DEFAULT = {
    "buffer_size": 50000 → 25000,  # -50%
    "learning_starts": 1000 → 500,  # 早期開始
    ...
}

メリット:
  - メモリ使用量削減: ~100MB
  - 時間短縮: ~5-10%（2-4分）
  - 全体: 7.2時間 → 6.5時間

デメリット:
  - 学習安定性低下の可能性
  - Experience多様性減少

リスク: 低
推奨度: ★★★★☆
```

**案D: batch_size削減（全実験）**
```python
SAC_DEFAULT = {
    "batch_size": 256 → 128,  # -50%
    ...
}

メリット:
  - 時間短縮: ~15%（6.5分）
  - 全体: 7.2時間 → 6.1時間
  - 実験Eで既に実施済み

デメリット:
  - Gradient更新の精度低下
  - 学習曲線の不安定化

リスク: 中
推奨度: ★★★☆☆
```

---

### 方策3: 並列実行

**案E: 2実験同時実行**
```python
import multiprocessing
pool = multiprocessing.Pool(processes=2)

メリット:
  - 理論上2倍速: 7.2時間 → 3.6時間
  - CPU余裕あり（現在20-30%使用率）

デメリット:
  - Windows環境での安定性懸念
  - メモリ消費: ~4GB → ~8GB
  - PyTorch multiprocessing問題のリスク

実装難易度: 高
リスク: 高（Windows環境）
推奨度: ★★☆☆☆（Linuxなら★★★★☆）
```

**案F: 順次実行 + バックグラウンド**
```powershell
# 昼間実行、夜間放置
Start-Process python "scripts/v459/run_day6_reward_tuning.py"

メリット:
  - リスクなし
  - 実装コスト0

デメリット:
  - 時間短縮効果なし

推奨度: ★★★★★（最も安全）
```

---

### 方策4: 実験数削減

**案G: 5 configs × 2 seeds → 5 configs × 1 seed**
```yaml
SEEDS = [42]  # 123を削除

メリット:
  - 時間: 7.2時間 → 3.6時間（-50%）

デメリット:
  - 51番基準「2seed平均」を満たせない
  - Seed依存性が見えない
  - 科学的厳密性の低下

リスク: 高（基準違反）
推奨度: ★☆☆☆☆
```

**案H: 段階的実行（Phase分割）**
```yaml
Phase 1（優先度高）:
  - A_Baseline, C_HoldRemoved, D_TradeReduced
  - 3 configs × 2 seeds = 6実験
  - 時間: 4.4時間
  
Phase 2（Phase 1結果次第）:
  - B_Stage1, E_ExplorationTuned
  - 2 configs × 2 seeds = 4実験
  - 時間: 2.9時間

メリット:
  - Phase 1結果で判断可能
  - 無駄な実験を回避

デメリット:
  - 総時間は変わらず
  - 手動介入が必要

推奨度: ★★★★☆（柔軟性高い）
```

---

## 推奨方策の組み合わせ

### 🏆 推奨案: 「最小リスク最適化」

**組み合わせ: 案C（buffer削減） + 案F（バックグラウンド実行）**

```python
# run_day6_reward_tuning.py 修正
SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 25000,  # 50000 → 25000
    "learning_starts": 500,  # 1000 → 500
    "batch_size": 256,  # 維持
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto"
}

# total_timesteps維持: 50,000
```

**効果:**
- 時間短縮: 7.2時間 → **6.5時間**（-10%）
- メモリ削減: ~100MB
- リスク: **低**
- 51番基準: **満たす**（2seed平均維持）
- 0番整合性: **維持**（50,000維持）

**実行方法:**
```powershell
# 夜間実行（23:00開始 → 翌朝5:30完了）
$env:ZTB_SIGINT_POLICY="ignore"
python scripts/v459/run_day6_reward_tuning.py 2>&1 | Tee-Object -FilePath "logs/day6_full_$(Get-Date -Format 'yyMMdd_HHmmss').log"
```

---

### 🥈 代替案: 「積極的最適化」（リスク許容時）

**組み合わせ: 案A（30,000 steps） + 案C（buffer削減） + 案D（batch削減）**

```python
# total_timesteps: 50,000 → 30,000
# buffer_size: 50,000 → 25,000
# batch_size: 256 → 128

SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 128,  # 削減
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto"
}
```

**効果:**
- 時間短縮: 7.2時間 → **3.5時間**（-51%）
- リスク: **中**（学習不足の可能性）
- 判断基準: Phase 1の3実験（A, C, D）で30,000でも収束すれば採用

---

### 🥉 最速案: 「段階的実行」（柔軟性重視）

**Phase 1実行（優先度高）:**
```yaml
実験: A_Baseline, C_HoldRemoved, D_TradeReduced
Seeds: 42, 123
Config: total_timesteps=30,000, buffer=25,000
時間: 6実験 × 26分 = 2.6時間
```

**Phase 1判定:**
- ✅ ROI改善あり（C or D で > -3%） → Phase 2実行
- ❌ 改善なし → 50,000に戻してPhase 2実行

**Phase 2実行:**
```yaml
実験: B_Stage1, E_ExplorationTuned
Seeds: 42, 123
Config: Phase 1結果に応じて調整
時間: 4実験 × 26-43分 = 1.7-2.9時間
```

**総時間:**
- 最良: 2.6 + 1.7 = **4.3時間**
- 最悪: 2.6 + 2.9 = **5.5時間**

---

## 52番文書への反映事項

### タイムライン修正

```markdown
# 修正前
Day 6-7（1/29-1/30）: 報酬調整A/Bテスト
  - 実験実行: 7.2時間（バックグラウンド）
  
# 修正後（推奨案）
Day 6-7（1/29-1/30）: 報酬調整A/Bテスト
  - 実験実行: 6.5時間（バックグラウンド、夜間実行推奨）
  - 設定最適化: buffer_size削減（50k→25k）
  
# 修正後（積極的最適化）
Day 6-7（1/29-1/30）: 報酬調整A/Bテスト
  - 実験実行: 3.5時間（バックグラウンド）
  - 設定: total_timesteps=30,000, buffer=25k, batch=128
  - リスク: 学習不足の可能性、Phase 1で判定
```

### 実験仕様修正

```python
# 52番 Day 6-7実行仕様に追記
BASE_CONFIG = {
    "training": {
        "total_timesteps": 50000,  # or 30000（積極案）
        "sac_hyperparameters": {
            "buffer_size": 25000,  # 最適化
            "learning_starts": 500,  # 最適化
            "batch_size": 256,  # or 128（積極案）
            ...
        }
    }
}

# 推定時間
# 推奨案: 6.5時間（10実験 × 39分）
# 積極案: 3.5時間（10実験 × 21分）
```

---

## 実行判断フローチャート

```
スタート
  ↓
時間的余裕は？
  ├─ YES（7時間以上） → 推奨案（6.5時間、低リスク）
  ├─ NO（4時間程度）  → 積極案（3.5時間、中リスク）
  └─ 柔軟対応したい   → 段階的実行（4.3-5.5時間）
  ↓
実行開始
  ↓
Phase 1完了（推奨案: 3.9時間、積極案: 2.1時間）
  ↓
中間評価
  ├─ ROI改善あり → Phase 2続行
  └─ 改善なし    → 設定見直し（50,000に戻す）
  ↓
Phase 2完了
  ↓
統合分析（Day 10）
```

---

## 最終推奨

### 時間的余裕がある場合（推奨）
```powershell
# buffer最適化のみ、total_timesteps=50,000維持
# 時間: 6.5時間、リスク: 低
python scripts/v459/run_day6_reward_tuning.py --optimized
```

### 時間短縮が必要な場合
```powershell
# total_timesteps=30,000 + buffer最適化
# 時間: 3.5時間、リスク: 中
python scripts/v459/run_day6_reward_tuning.py --fast
```

### 最も柔軟な場合
```powershell
# Phase 1（A, C, D）のみ実行、結果次第でPhase 2
# 時間: 2.6時間 + Phase 2判断
python scripts/v459/run_day6_reward_tuning.py --phased
```

---

**作成日**: 2026-01-29  
**関連文書**: 52_phase4_week2_implementation_plan_revised.md  
**次のアクション**: 52番文書のタイムライン修正
