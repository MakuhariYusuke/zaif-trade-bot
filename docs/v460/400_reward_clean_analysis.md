# 400# 報酬関数分析 — vXXXシリーズ横断的レビューと reward-clean 設計

**作成日**: 2025-07-16
**根拠**: vXXXシリーズ全体の報酬関数実験結果 + 399#実験のG3 FAIL原因分析

---

## 1. 背景: 399# 実験結果 (20K × 4 seeds)

| Seed | ROI | PF | Sharpe | Reward-PnL相関 |
|------|-------|------|--------|----------------|
| 42 | -0.31% | 0.955 | -2.85 | **-0.14** |
| 123 | +0.29% | 1.050 | +2.82 | +0.26 |
| 456 | +0.03% | 1.012 | +0.61 | +0.13 |
| 789 | +0.001% | 1.000 | -0.18 | **-0.38** |

**G2=PASS, G3=FAIL** (PF median=1.006 < 1.05, Sharpe=0.22 < 0.8)

**核心問題**: Seed 42, 789 の **reward-PnL相関が負** → 報酬関数が利益を出す行動を正しく強化していない

---

## 2. 問題分析: 3つの構造的欠陥

### 問題1: `scale_adjustment` 100x増幅 + clip[-80,80] による勾配破壊

```
scale_adjustment = scale_adjustment_base(1.0) / max(0.01, max_position_size(0.01))
                 = 1.0 / 0.01 = 100.0
→ reward_scaling(実効) = 1.0 × 100.0 = 100.0
```

**影響**: BTC ¥15M × 0.01BTC ポジションで:
- ¥1(小幅利確): reward = 1 × 100 = 100 → clip(80) ← **切り詰め**
- ¥50(大幅利確): reward = 50 × 100 = 5000 → clip(80) ← **¥1の利確と同じ報酬！**
- agent は「大きく稼ぐ」vs「小さく稼ぐ」を区別できない

**コード箇所**: [reward_calculator.py](ztb/trading/environment/components/calculators/reward_calculator.py#L992-L994)

### 問題2: PnLと無関係な7つのペナルティ/シェーピング

`_calculate_default_reward()` → `calculate_reward()` のパイプラインで **10コンポーネント** が加算:

| # | コンポーネント | 値 | 状態 |
|---|---|---|---|
| 1 | pnl_reward | PnL × 100.0(実効) | ✅ 主信号 |
| 2 | position_penalty | -0.01 × (pos/max)² | ✅ 微小 |
| 3 | hold_penalty | -0.001 | ✅ 微小 |
| 4 | consistency_penalty | -0.01 (反転時) | ✅ 小 |
| 5 | confidence_penalty | 可変 | ✅ 中 |
| 6 | balance_penalty | -excess × 0.1 | ✅ 小〜中 |
| 7 | balance_shaping | **0.5 × improvement** | ⚠️ **未制御で有効** |
| 8 | entropy_shaping | 0.01 × shortfall | ⚠️ **未制御で有効** |
| 9 | skew_penalty | 0.0 | ❌ 無効 |
| 10 | action_bonus | 0.0 | ❌ 無効 |

**balance_shaping(value=0.5)** が特に問題。PnL=0のステップでも「行動バランスを改善する」ボーナスを与え、エージェントが **利益最大化より行動分布均等化を学習** してしまう。

### 問題3: v459 E設定の知見が反映されていない

v459 62_day6 A/Bテスト（5 config × 2 seeds = 10実験）で明確な勝因が判明:

```
B→C: hold_penalty=0, clip[-1,1]     → +3,500% 改善 🥇
D→E: ent_coef=0.01固定, γ=0.95      → +295% 改善 🥈
A→B: reward_scale=100.0             → +111% 改善 🥉
```

**現行config vs v459 E設定(Best)**:

| パラメータ | v459 E設定 | reward-tuned | 差 |
|---|---|---|---|
| hold_penalty | **0** | 0.001 | 残存 |
| reward_clip | **[-1,1]** | [-80,80] | 80倍広い |
| ent_coef | **0.01(固定)** | "auto" | 不安定 |
| gradient_steps | **2** | 1 | 半分 |
| batch_size | **128** | 256 | 2倍 |
| lr | **5e-4** | 3e-4 | 異なる |
| balance_shaping | **off** | on(0.5) | 有効 |
| entropy_shaping | **off** | on(0.01) | 有効 |

---

## 3. vXXXシリーズ 報酬関数の教訓マトリクス

| バージョン | アプローチ | 結果 | 教訓 |
|---|---|---|---|
| v378 | Scale-adjusted (HOLD×4, profit×3) | PPO実験 | ペナルティスケーリング初期試行 |
| v380 | Aggressive anti-HOLD (×10, ×5, ×6) | 崩壊 | 過激なペナルティは逆効果 |
| v435.2 | Curriculum学習 | **+0.601% (唯一の正ROI)** | 段階的報酬が機能 |
| v435.7 | 非対称報酬 | SELL=0%に崩壊 | 非対称設計のリスク |
| v451 | シンプル報酬 (γ=0.80) | **最も効果的** | シンプル is ベスト |
| v455 | Edge/Vol/Time Penalty | -9.3% | ペナルティ積層は機能しない |
| v456 | 9項目ペナルティ | BUY:100%に崩壊 | ペナルティ追加は最後の手段 |
| v457.1 | Gross PnL+, Net PnL- | PF=1.14 | 取引自体は機能する実証 |
| v458 | コスト計算修正 | コスト二重計上発覚 | Net PnL経路の一元化が必要 |
| v459 | 5 config A/B テスト | **E設定が最高** | hold=0, clip[-1,1], ent_coef=0.01 |
| v460-387# | ペナルティ縮小 | G2 PASS | ペナルティ縮小は正しい方向 |
| v460-399# | reward-tuned 本番 | G3 FAIL | まだペナルティが多すぎる |

> **核心原則**: ペナルティを積み上げるほど、モデルは「罰を避ける」ことを学習し、収益目標から乖離する
> — v455, v456, v457.2 で3回実証済み

---

## 4. 改善策: `g2_sac_reward_clean.yaml`

### 設計哲学: **PnL信号のみ、ペナルティゼロ**

v459 E設定の知見を全て適用し、さらにペナルティを完全撤廃:

### 4.1 報酬構造の変更

| パラメータ | reward-tuned | **reward-clean** | 根拠 |
|---|---|---|---|
| scale_adjustment_enabled | true(暗黙) | **false** | 100x増幅 → clip飽和を防止 |
| reward_scaling | 1.0(→実効100) | **100.0(明示)** | v459 B設定 |
| reward_clip | [-80, 80] | **[-1, 1]** | v459 C設定 (+3500%改善) |
| hold_penalty | 0.001 | **0.0** | v459 C設定 (完全撤廃) |
| consistency_penalty | 0.01 | **0.0** | ペナルティ最小化 |
| balance_penalty | 0.1 | **0.0** | PnL主体 |
| position_penalty | 0.01 | **0.0** | 同上 |
| confidence_penalty | threshold=0.2 | **threshold=1.0** | 実質無効化 |
| balance_shaping | on(0.5) | **off** | 行動バランス → PnLとの干渉 |
| entropy_shaping | on(0.01) | **off** | 同上 |

### 4.2 SAC ハイパーパラメータの変更

| パラメータ | reward-tuned | **reward-clean** | 根拠 |
|---|---|---|---|
| ent_coef | "auto" | **0.01** | v459 E設定: 探索暴走防止 |
| gradient_steps | 1 | **2** | v459 E設定 |
| batch_size | 256 | **128** | v459 E設定 |
| learning_rate | 3e-4 | **5e-4** | v459 E設定 |

### 4.3 コード変更

**`reward_calculator.py`**: `scale_adjustment_enabled` フラグ追加

```python
# 変更前: 常に100x増幅
scale_adjustment_base = self.get_setting_float("scale_adjustment_base", 1.0)
scale_adjustment = scale_adjustment_base / max(0.01, max_position_size)
reward_scaling = reward_scaling * scale_adjustment

# 変更後: YAML設定で制御可能に（デフォルト: True で後方互換保持）
scale_adjustment_enabled = self.get_setting_bool("scale_adjustment_enabled", True)
if scale_adjustment_enabled:
    scale_adjustment_base = self.get_setting_float("scale_adjustment_base", 1.0)
    scale_adjustment = scale_adjustment_base / max(0.01, max_position_size)
    reward_scaling = reward_scaling * scale_adjustment
```

---

## 5. 期待される効果

### 理論的根拠

1. **PnL信号の線形性回復**: clip[-1,1] + reward_scaling=100 で、¥0.01の差が reward 0.01 の差になり、勾配信号が利益の大小を正しく反映
2. **ペナルティ撤廃**: agent の全学習容量を「利益最大化」に集中
3. **SAC探索安定化**: ent_coef=0.01固定で探索の暴走を防止

### 予測 vs ベースライン

| 指標 | reward-tuned (399#) | 予測 (reward-clean) |
|---|---|---|
| reward-PnL相関 | -0.14〜+0.26 | **+0.3〜+0.6** |
| PF median | 1.006 | **>1.05** (G3閾値) |
| Sharpe | 0.22 | **>0.8** (G3閾値) |

---

## 6. セルフレビューで発見した追加バグ (400# FIX)

### 問題4: `balance_shaping_enabled` / `action_entropy_shaping_enabled` が YAML 設定を無視

**発見経緯**: 400# セルフレビューで `behavior_optimization.balance_shaping_enabled: false` の
設定伝搬経路を追跡した結果、設定が `BehavioralPenaltyCalculator` に到達しないことを確認。

**原因**:
1. `BehavioralPenaltyCalculator._load_settings()` は `config.reward_settings` (RewardSettings dataclass) を優先取得
2. `config.behavior_optimization` へのフォールバックは `reward_settings is None` の場合のみ発動
3. `RewardSettings` dataclass に `balance_shaping_enabled` / `action_entropy_shaping_enabled` フィールドが存在しない
4. → `_rs_get("balance_shaping_enabled", True)` は常に default `True` を返却

**修正**:
- `_load_settings()` で `config.behavior_optimization` を二次フォールバックソースとして追加
- `_rs_get` の dataclass パスで primary source に見つからないキーを fallback dict から取得

**影響**: 387# 以降のすべての実験で `balance_shaping` (value=0.5) と `action_entropy_shaping` (value=0.01) が
**意図せず常に有効**のまま実行されていた。これは問題2の「未制御で有効」の根本原因。

---

## 7. vXXX 重複排除レビュー

| 項目 | 状態 | 詳細 |
|---|---|---|
| V457RewardCalculator | ❌ 重複なし | 別クラス、別用途（純粋PnL計算器） |
| v459 scale_deconfounding | ❌ 重複なし | 実験スクリプト、configベースの設定とは別 |
| 386# reward_scaling dead code | ✅ 387#で修正済み | `_calculate_default_reward` に `reward_scaling` パラメータ追加済み |
| scale_adjustment ロジック | ✅ 400#で制御可能化 | `scale_adjustment_enabled` フラグ追加 |

---

## 8. config ファイル

[g2_sac_reward_clean.yaml](configs/v460/experiments/g2_sac_reward_clean.yaml)

## 9. 次のステップ

1. `g2_sac_reward_clean.yaml` で 20K × 4 seeds 実験実行
2. G2/G3 判定結果を 399# と比較
3. 改善が見られたら 100K × 4 seeds で本番実験
