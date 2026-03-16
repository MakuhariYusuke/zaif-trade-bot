# 52. Phase 4 Week 2 実装計画（修正版）: 報酬調整 + 軽量MTF戦略

**日付**: 2026-01-28（51番レビュー対応版）、2026-01-29更新  
**前提**: Phase 4 Day 5完了、外部レビュー対応（51番）  
**目標**: 50,000ステップでROI 0%以上達成  
**詳細分析**: [53_time_optimization_strategies.md](53_time_optimization_strategies.md)

---

## 51番レビュー対応サマリー

### ✅ 既存実装の発見と活用

| 項目 | 既存実装 | 活用方針 |
|------|----------|---------|
| 報酬設定 | `stage1_basic.yaml` | そのまま使用 |
| PnL Focused | `PnLFocusedRewardCalculator` | そのまま使用 |
| A/Bフレームワーク | `run_ab_reward_experiments.py` | 新規スクリプトで簡略化 |
| 取引コスト | 環境でnet_pnl計算済み | 二重計上回避 |

### 🔧 修正事項

1. **報酬設定経路**: 既存`stage1_basic.yaml`ベース
2. **MTFリーク防止**: 既存パイプライン活用（shift済み）
3. **評価基準**: 10実験・2seed平均で統一
4. **Stage 1定義**: `use_simple_reward: false` + PnLFocusedRewardCalculator

---

## 0番プロポーザルとの整合性

| フェーズ | 状態 |
|----------|------|
| Phase 0-2 | ✅ 完了 |
| **Phase 3（報酬設計）** | 🔄 **Week 2実施中** |
| **Phase 4前半（評価）** | 🔄 **Week 2実施中** |
| Phase 4後半 | ⏳ 未着手 |

**0番Stage 1定義**:
```python
# reward = (current_balance - previous_balance) / initial_balance
# 実装: PnLFocusedRewardCalculator + stage1_basic.yaml
```

---

## Phase 4 Day 5結果（ベースライン）

```
ROI: -5.074%（8特徴Parquet平均）
負報酬比率: 99.34%
SELL偏重: 47.4%
取引回数: 275回平均
```

**課題**: 負報酬過多、SELL偏重、取引コスト高

## Week 2実装計画

### 設計方針

✅ **既存実装活用（DRY原則）**:
1. `stage1_basic.yaml`をベース使用
2. `PnLFocusedRewardCalculator`そのまま使用
3. 新規スクリプト`run_day6_reward_tuning.py`で簡略化
4. 既存MTFパイプライン活用

❌ **回避**:
- 新規報酬計算ロジック実装
- 複雑なA/Bフレームワーク
- 独自MTF生成

---

## Day 6-7: 報酬調整A/Bテスト（既存実装活用）

### 目的
50,000ステップで損益分岐点突破（-5% → 0%以上）

### 実験設計（10実験）

| 実験ID | 報酬設定ファイル | 変更内容 | 期待効果 |
|--------|-----------------|---------|---------|
| **A (Baseline)** | 現状の設定 | - | -5.074% |
| **B (Stage1)** | `stage1_basic.yaml` | 0番Stage 1準拠 | ベースライン確立 |
| **C (Hold削除)** | `stage1_hold_removed.yaml` | `trading_bonus: 0.0` | 報酬バランス改善 |
| **D (取引抑制)** | `stage1_trade_reduced.yaml` | `trade_frequency_penalty: 0.01` | 取引回数50%削減 |
| **E (探索調整)** | `stage1_exploration_tuned.yaml` | SAC: `ent_coef: 0.01, gamma: 0.95` | 早期収束 |

**Seeds**: 42, 123  
**Total**: 5 configs × 2 seeds = **10実験**

### 具体的な実装（YAML設定のみ）

#### 実験B: stage1_basic.yaml（既存）
```yaml
# configs/rewards/stage1_basic.yaml（既存ファイルをそのまま使用）
name: "stage1_basic"
curriculum_stage: "simple"
use_simple_reward: false
reward_scale: 100.0
profit_weight: 1.0
risk_weight: 0.3
consistency_weight: 0.1
```

#### 実験C: stage1_hold_removed.yaml（新規）
```yaml
# configs/rewards/stage1_hold_removed.yaml
name: "stage1_hold_removed"
curriculum_stage: "simple"
use_simple_reward: false
reward_scale: 100.0
profit_weight: 1.0
risk_weight: 0.3
consistency_weight: 0.1

# Hold削除（49番優先1）
trading_bonus: 0.0  # 0.01 → 0.0

# Drawdown縮小は環境側で実装済み（設定不要）

# 報酬クリップ
reward_clip_min: -1.0  # -80.0 → -1.0に縮小
reward_clip_max: 1.0   # 80.0 → 1.0に縮小
```

#### 実験D: stage1_trade_reduced.yaml（新規）
```yaml
# configs/rewards/stage1_trade_reduced.yaml
name: "stage1_trade_reduced"
curriculum_stage: "simple"
use_simple_reward: false
reward_scale: 100.0
profit_weight: 1.0

# Hold削除（継承）
trading_bonus: 0.0

# 取引抑制強化（49番優先2）
trade_frequency_penalty: 0.01  # 0.001 → 0.01（10倍）
trade_cooldown_steps: 10  # 5 → 10
trade_cooldown_penalty: 0.05  # 0.01 → 0.05
action_smoothing: 0.01  # 0.0 → 0.01（アクション変化ペナルティ）

# 報酬クリップ
reward_clip_min: -1.0
reward_clip_max: 1.0
```

#### 実験E: stage1_exploration_tuned.yaml（新規） + SAC設定
```yaml
# configs/rewards/stage1_exploration_tuned.yaml
name: "stage1_exploration_tuned"
curriculum_stage: "simple"
use_simple_reward: false
reward_scale: 100.0
profit_weight: 1.0

# 取引抑制（継承）
trading_bonus: 0.0
trade_frequency_penalty: 0.01
trade_cooldown_steps: 10
trade_cooldown_penalty: 0.05
action_smoothing: 0.01

# 報酬クリップ
reward_clip_min: -1.0
reward_clip_max: 1.0
```

```python
# SAC設定（run_ab_reward_experiments.py内で設定）
sac_hyperparameters = {
    "learning_rate": 0.0005,  # 0.0003 → 0.0005
    "ent_coef": 0.01,  # "auto" → 0.01（固定）
    "gamma": 0.95,  # 0.99 → 0.95
    "batch_size": 128,  # 256 → 128
    "gradient_steps": 2,  # 1 → 2
}
```

### 実行仕様

```python
# scripts/v459/run_day6_reward_tuning.py（新規スクリプト）

SEEDS = [42, 123]
REWARD_CONFIGS = [
    None,  # A: Baseline
    "configs/rewards/stage1_basic.yaml",  # B
    "configs/rewards/stage1_hold_removed.yaml",  # C
    "configs/rewards/stage1_trade_reduced.yaml",  # D
    "configs/rewards/stage1_exploration_tuned.yaml"  # E
]

BASE_CONFIG = {
    "training": {
        "total_timesteps": 50000,
        "sac_hyperparameters": {
            "buffer_size": 25000,  # 最適化: 50000 → 25000
            "learning_starts": 500,  # 最適化: 1000 → 500
            "batch_size": 256,  # 実験A-D
            # 実験E: batch_size=128, ent_coef=0.01等
        }
    }
}
```

**時間短縮方策**（詳細: [53_time_optimization_strategies.md](53_time_optimization_strategies.md)）:
- buffer_size: 50k → 25k（メモリ-100MB、時間-10%）
- learning_starts: 1000 → 500

**推定時間**: 
- 推奨案: 10実験 × 39分 = **6.5時間**
- 積極案: 10実験 × 21分 = 3.5時間（30k steps、リスク中）

### 成功基準（51番指摘対応）

| 指標 | 目標 | 判定方法 |
|------|------|---------|
| **ROI** | 0%以上 | **2seed平均** |
| **取引回数** | 140回以下（50%削減） | 実験D以降 |
| **正報酬比率** | 10%以上 | 実験C以降 |

**最低ライン**: 5 configs × 2 seeds = 10実験のうち、**2 configs以上で2seed平均ROI > 0%**

## Day 8: 軽量MTF実装

### 目的
Phase 3.5の99.83%削減効果維持 + 0番v456/v457の88次元思想部分採用

### 設計（既存活用）

**軽量MTF設定**:
```json
{
  "timeframes": ["5T", "15T"],
  "features": {
    "5T": ["ma_trend", "volatility", "regime", "volume_trend"],
    "15T": ["ma_trend", "price_position", "momentum"]
  },
  "lookback": {"5T": 20, "15T": 12},
  "shift": 1
}
```

**実装**: 既存パイプライン呼び出しのみ
```python
from ztb.features.generators.multi_timeframe import create_multi_timeframe_system
system = create_multi_timeframe_system(config_path=config_path)
df_with_mtf = system.process_multi_timeframe_data(df, timeframes=["5T", "15T"])
```

**目標**: 生成時間 < 2秒

## Day 9: MTF A/Bテスト

### 実験設計

| ID | 特徴構成 | Seeds | 目的 |
|----|---------|-------|------|
| **F** | 8特徴 | 42, 123 | ベースライン |
| **G** | 16特徴（+MTF） | 42, 123 | MTF効果検証 |

**Total**: 2 configs × 2 seeds = 4実験（2.9時間）

### 評価指標

| 指標 | 目標 | 判定 |
|------|------|------|
| **ROI** | 0%以上 | 2seed平均 |
| **SELL偏重** | 35%以下 | 2seed平均 |
| **取引回数** | 140回以下 | 2seed平均 |

## Day 10: 統合分析とPhase 4完了判定

### Phase 4完了判定（51番対応）

#### ✅ 合格条件
1. ROI改善: **2 configs以上**で2seed平均ROI > 0%
2. 取引抑制: 実験D以降で50%削減達成
3. 報酬バランス: 正報酬比率10%以上
4. MTF効果: 実験Gが実験Fより優位

#### ⚠️ 条件付き合格
- ROI: -2% ～ 0%（1 config以上）
- 取引抑制: 30%削減達成

#### ❌ 不合格（Phase 4延長）
- ROI: -3%以下
- 取引削減率: 20%未満

### 0番Gate 1-2判定

**Gate 1（収益性）**: ROI > 0%（2 configs以上で達成）  
**Gate 2（安定性）**: Sharpe > 0.3（最良configで達成）

---

## 成果物と完了条件

### Day 6-7
- [x] 報酬設定YAML 3種
- [ ] 実験結果JSON
- [ ] 取引コスト実測分析
- [ ] 報酬調整効果レポート

### Day 8
- [ ] 軽量MTF設定ファイル
- [ ] 16特徴Parquet
- [ ] MTF生成時間ベンチマーク（< 2秒）

### Day 9
- [ ] MTF A/Bテスト結果JSON
- [ ] MTF効果分析レポート

### Day 10
- [ ] 統合分析レポート
- [ ] Gate 1-2判定結果
- [ ] Phase 5移行判断

---

## 工数削減効果

| 項目 | 当初 | 修正後 | 削減 |
|------|------|--------|------|
| 報酬計算 | 新規8h | 既存活用 | **-8h** |
| A/Bフレームワーク | 新規6h | 簡略化 | **-6h** |
| MTF生成 | 新規10h | 既存活用 | **-10h** |
| YAML作成 | 5種 | 3種 | **-2h** |
| **合計** | 30h | 19.5h | **-35%** |

### タイムライン（修正版）

```
Phase 4 Week 2（Day 6-10）: 2026-01-29 ～ 2026-02-02

Day 6-7（1/29-1/30）: 報酬調整A/Bテスト
  - YAML作成: 2時間（3種、完了済み）✅
  - スクリプト作成: 1時間（run_day6_reward_tuning.py、完了済み）✅
  - 実験実行: 6.5時間（バックグラウンド、夜間推奨）
    * 推奨案: 6.5時間（buffer最適化、低リスク）
    * 積極案: 3.5時間（30k steps、中リスク）
  - 分析: 3時間（取引コスト実測含む）
  
Day 8（1/31）: 軽量MTF実装
  - 設定ファイル作成: 1時間
  - ベンチマーク実行: 0.5時間
  - 検証: 1時間
  
Day 9（2/1）: MTF A/Bテスト
  - 実験実行: 2.9時間（バックグラウンド）
  - 分析: 2時間
  
Day 10（2/2）: 統合分析とPhase 4完了判定
  - 統合分析: 3時間
  - Gate判定: 2時間
  - 報告書作成: 3時間
```

**合計**: 
- **推奨案**: 約19.5時間（当初30時間 → **-35%削減**）
- **積極案**: 約16.5時間（当初30時間 → **-45%削減**）

---

## リスクと対策（51番対応）

### リスク1: 既存MTFパイプラインが想定外に遅い
**確率**: 低  
**対策**: 
- Day 8開始時に即ベンチマーク実行
- 3秒超過の場合、15分足特徴を削除（16特徴 → 12特徴）
- 5秒超過の場合、MTF追加を延期（Phase 5へ）

### リスク2: stage1_basic.yamlが期待通り動作しない
**確率**: 低  
**対策**: 
- 実験B開始前に小規模テスト（1000ステップ）
- 問題がある場合、`use_simple_reward: true`に切り替え

### リスク3: 評価基準の母数不整合（51番指摘）
**確率**: ゼロ（修正済み）  
**対策**: 
- 全実験で「5 configs × 2 seeds = 10実験」を明記
- 判定基準を「2 configs以上で2seed平均ROI > 0%」に統一

## まとめ

Phase 4 Week 2は**既存実装を最大活用**し、0番プロポーザルを効率的に実現します。

**3つの柱**:
1. 既存stage1_basic.yamlで0番Stage 1実現
2. 49番提案を段階適用
3. 既存MTFパイプラインで軽量MTF追加

**工数削減**: 30時間 → 19.5時間（**-35%**）

**最終目標**: 50,000ステップでROI 0%以上達成（2 configs以上で2seed平均）

---

## 実行オプション（2026-01-29追記）

### オプション1: 推奨案（低リスク、6.5時間）
```powershell
$env:ZTB_SIGINT_POLICY="ignore"
python scripts/v459/run_day6_reward_tuning.py 2>&1 | Tee-Object -FilePath "logs/day6_full_$(Get-Date -Format 'yyMMdd_HHmmss').log"
```
- 51番基準満たす（2seed平均、50,000 steps）
- リスク: 低

### オプション2: 積極案（中リスク、3.5時間）
- total_timesteps=30,000、batch削減
- リスク: 中（学習不足可能性）

### オプション3: 段階的実行（柔軟、4-5時間）
- Phase 1（A, C, D）のみ実行
- 結果次第でPhase 2判断

**推奨**: 時間余裕あれば**オプション1**、制約あれば**オプション3**

---

**次のアクション**:
1. ✅ Day 6準備完了
2. 実行判断: オプション1-3から選択
3. 全実験実行（夜間推奨）
4. Day 7: 結果分析

**文書管理**:
- 作成: 2026-01-28（51番レビュー対応）
- 更新: 2026-01-29（時間短縮方策、実装完了）
- 付録: [53_time_optimization_strategies.md](53_time_optimization_strategies.md)
- 次回更新: Day 10（統合分析完了時）
