# 96# 改訂実行計画: 報酬純粋性確立と統計的検証

**Date**: 2026-02-06  
**基づく**: 95# Gate0/PhaseB レビュー → 0番/66番整合性要件 → vXXXシリーズ教訓  
**Status**: Phase A完了、Phase B実行準備完了  
**大義**: 短期間での高収益性システム — 正しい土台の上にのみ建てる

---

## 0. 経緯と現在地

### 95#レビュー指摘の妥当性判定

| # | 指摘 | 重大度 | コード検証結果 | 判定 |
|---|------|--------|---------------|------|
| 1 | P1-1は「PnLのみ」ではない | Critical | **✅ 妥当** — 後述§1で詳述 |
| 2 | 1 seed/10K stepsで「確定事実」は過大 | Critical | **✅ 妥当** — 0番§5.6でn≥16要求 |
| 3 | Gate0は「伝播確認」であり「有効化確認」ではない | High | **✅ 妥当** — 未知キーが`custom_reward_params`に吸収 |
| 4 | Cost Ratioが不安定 | High | **✅ 妥当** — gross≈0で発散する構造的欠陥 |
| 5 | データリーク警告が残存 | High | **✅ 妥当** — `train_end_index`未指定 |
| 6 | Phase5移行基準未充足 | Medium | **✅ 妥当** — 0番§5.2の全指標が未測定 |

**総合判定**: 95#の指摘は全項目妥当。94#の結論は修正が必要。

---

## 1. [Critical] P1-1の報酬経路汚染 — コード検証の詳細

### 1.1 P1-1の意図 vs 実際の経路

**意図**: ペナルティ全無効のPnLのみ報酬  
**実際**: `use_simple_reward=False`（デフォルト）のため`_calculate_default_reward` + 6つの追加コンポーネントが動作

### 1.2 P1-1が無効化に**失敗**しているコンポーネント

| コンポーネント | デフォルト値 | P1-1での設定 | 影響 |
|---|---|---|---|
| `confidence_penalty_factor` | `1.0` | **未設定** | 損失時に暗黙ペナルティ |
| `balance_shaping_enabled` | `True` | **未設定** | アクション分布を歪める |
| `balance_shaping_value` | `0.5` | **未設定** | base_rewardに最大0.5加算 |
| `action_entropy_shaping_enabled` | `True` | **未設定** | エントロピー項が混入 |
| `action_entropy_shaping_value` | `0.01` | **未設定** | base_rewardに加算 |
| `position_penalty_weight` | `0.01` | **未設定** | `_calculate_default_reward`内部 |
| `hold_penalty_weight` | `0.01` | **未設定** | HOLD時にペナルティ |

### 1.3 `_calculate_default_reward` 内部の非PnL項

```python
# reward_calculator.py L1756-L1783
def _calculate_default_reward(self, ...):
    pnl_reward = self._calculate_pnl_reward(pnl, 1.0)          # ← PnL部分
    position_penalty = self._calculate_position_penalty(...)     # ← 非PnL
    hold_penalty = self._calculate_hold_penalty(action)          # ← 非PnL
    consistency_penalty = self.behavioral_penalty_calculator...   # ← 非PnL
    reward = pnl_reward + position_penalty + hold_penalty + consistency_penalty
    return reward
```

**さらに**`calculate_reward()`内で加算:
```
base_reward（上記4項）
  + confidence_penalty      ← 暗黙動作
  + action_bonus             ← 通常0
  + balance_penalty          ← P1-1で0設定済み ✅
  + skew_penalty             ← enabled=False で0 ✅
  + balance_shaping          ← enabled=True, value=0.5 ❌ 動作中
  + entropy_shaping          ← enabled=True, value=0.01 ❌ 動作中
→ _post_process_reward()（scaling, clipping, signal_integration）
```

### 1.4 94#結論への影響

| 94#の主張 | 修正判定 |
|---|---|
| 「P1-1: Gross PnL > 0、取引戦略自体は機能」 | ⚠️ **保留** — 実際はPnL以外の報酬成分が行動を歪めている可能性 |
| 「現行ペナルティは有害」 | ⚠️ **保留** — P1-1自体がペナルティ混在 |
| 「問題は過剰取引」 | ✅ **依然妥当** — 手数料/Gross PnL比は明確 |
| 「Gate 0は正常動作」 | ⚠️ **部分的** — 伝播は確認したが有効化は未確認 |

---

## 2. vXXXシリーズからの統合教訓

### 参照すべきバージョン知見

| バージョン | 教訓 | 本計画への適用 |
|---|---|---|
| **v435.2** | **Curriculum学習が唯一の+0.601%** | 段階的報酬導入パターンを継承 |
| **v435.7** | 非対称報酬でSELL=0%に崩壊 | BUY/SELL比率を常時監視 |
| **v444** | 設定伝播バグ（curriculum_stage未到達） | Gate 0 + Gate 0.5の二段検証 |
| **v451** | シンプル報酬(γ=0.80)が最も効果的 | Stage 1の基準設定値 |
| **v455** | Edge/Vol/Time Penaltyは-9.3%止まり | ペナルティ積層は機能しない |
| **v456** | 9項目ペナルティでBUY:100%/SELL:100% | ペナルティ追加は最後の手段 |
| **v457.1** | Gross PnL+, Net PnL-（PF=1.14） | 取引自体は機能する実証 |
| **v457.2** | 重いFee Penalty→逆効果 | PnL主体を先に確立 |
| **v457.3** | TTL固定で+36.8M利益 | 仕組みは解ける、過剰取引が構造問題 |
| **v458** | コスト二重計上発覚 | Net PnL計算経路の一元化 |

### 核心的結論

> **ペナルティを積み上げるほど、モデルは「罰を避ける」ことを学習し、収益目標から乖離する**  
> — v455, v456, v457.2で3回と実証済み。最初に純PnLの正確なベースラインを取ることが絶対条件。

---

## 3. 0番/66番との整合性要件

### 0番 §3.3 報酬設計の段階化

| Stage | 0番の定義 | 現在の実施状況 | 本計画 |
|---|---|---|---|
| Stage 1 | `R = PnL_net` | ❌ P1-1は複合報酬混在 | **再実装（§4.1）** |
| Stage 2 | `R = PnL_net - 0.05 * TrendPenalty` | ❌ 未着手 | 将来（Stage 1確立後） |
| Stage 3 | `R = PnL_net - W(t) * TrendPenalty` | ❌ 未着手 | 将来 |

### 0番 §5.2 成功基準（必須測定指標）

| 指標 | 最低基準 | 測定状況 | 本計画 |
|---|---|---|---|
| Net ROI | > 5% | ❌ 未測定 | §4.3で実装 |
| Profit Factor | > 1.20 | ❌ 未測定 | §4.3で実装 |
| Sharpe Ratio | > 1.0 | ❌ 未測定 | §4.3で実装 |
| Max Drawdown | < 15% | ❌ 未測定 | §4.3で実装 |
| Win Rate | > 35% | ❌ 未測定 | §4.3で実装 |

### 0番 §5.6 統計検定仕様

| 項目 | 要求 | 現状 | 本計画 |
|---|---|---|---|
| サンプル数 | n ≥ 16（4seed × 4split） | n = 1 | **4 seeds × 1 split（最低16→暫定4）** |
| 検定方法 | Mann-Whitney U | 未実施 | §4.4で実装 |
| 多重比較補正 | Holm-Bonferroni | 未実施 | §4.4で実装 |
| 効果量 | Cliff's Delta > 0.33 | 未実施 | §4.4で実装 |

### 66番からの軌道修正の正当な逸脱

| 逸脱点 | 理由 | 正当性 |
|---|---|---|
| Stage構造 → 49番優先策 | 外部専門家の実践的アドバイス | ✅ 引き続き妥当 |
| n≥16 → 暫定4 seeds | 計算時間制約（760min/フル構成） | ⚠️ 条件付き許容 |

---

## 4. 実行計画

### 全体方針

```
Phase A: 報酬純粋性の確立（Gate 0.5）     ← 最優先、全ての土台
Phase B: 真のPnLベースライン確立           ← 4 seeds、メトリクス完備
Phase C: コスト最適化（取引頻度制御）       ← Net PnL > 0 を目指す
Phase D: Phase 5 移行判定                  ← 0番 §5.2 基準
```

### 4.1 Phase A: 報酬純粋性の確立（推定: 2-3時間）

**目的**: `use_simple_reward=True`の報酬経路が真にPnLのみであることを確認し、実験基盤を確立

#### A-1: `use_simple_reward=True` 経路の検証

```python
# reward_calculator.py L914-L926
if use_simple_reward:
    return self.calculate_reward_simple(...)  # ← この経路を使う
```

**確認事項**:
- `calculate_reward_simple()` の実装を監査
- confidence_penalty / balance_shaping / entropy_shaping が混入しないことを確認
- 戻り値が `pnl * scaling` のみであることを検証

#### A-2: Gate 0.5 自動テスト作成

```python
# tests/v459/test_gate05_reward_purity.py
class TestRewardPurity:
    """報酬純粋性の自動テスト"""
    
    def test_simple_reward_is_pnl_only(self):
        """use_simple_reward=True で PnL 以外のコンポーネントが 0"""
        env = create_env(use_simple_reward=True)
        # ... step実行 ...
        components = env.reward_calculator._last_reward_components
        assert components.get("confidence_penalty", 0) == 0
        assert components.get("balance_shaping", 0) == 0
        assert components.get("entropy_shaping", 0) == 0
    
    def test_penalty_toggle_changes_reward(self):
        """balance_penalty=0 vs balance_penalty=0.5 で reward_components に差分"""
        reward_no_penalty = run_episode(balance_penalty=0.0)
        reward_with_penalty = run_episode(balance_penalty=0.5)
        assert reward_no_penalty != reward_with_penalty
    
    def test_unknown_keys_cause_warning(self):
        """未知キーが custom_reward_params に入った場合に警告"""
        settings = RewardSettings.from_dict({"nonexistent_key": 1.0})
        assert "nonexistent_key" in settings.custom_reward_params
        # → 将来的にはエラーにすべき
```

#### A-3: データリーク警告の解消

```python
# run_phase45_p1.py への追加
config = {
    "training": {
        "environment": {
            "reward_settings": {...},
            "train_end_index": int(len(df) * 0.8),  # 明示指定
        },
    },
}
```

### 4.2 Phase B: 真のPnLベースライン（推定: 4-8時間）

**目的**: 4 seeds で統計的に信頼できるPnLベースラインを取得

#### B-1: 実験設定

```python
# Pure PnL experiment
EXPERIMENT = {
    "name": "P1_pure_pnl",
    "seeds": [42, 123, 456, 789],
    "total_timesteps": 50_000,  # 10K→50K（v457.2の教訓: 短すぎると崩壊）
    "reward_settings": {
        "use_simple_reward": True,  # ← 最重要: 複合経路をバイパス
        # 念のため全ペナルティも0に
        "balance_penalty": 0.0,
        "position_penalty_scale": 0.0,
        "hold_penalty_multiplier": 0.0,
        "confidence_penalty_factor": 0.0,
        "balance_shaping_enabled": False,
        "action_entropy_shaping_enabled": False,
    },
    "train_end_index": "auto_80pct",  # リーク防止
}

# Default reward experiment（比較用）
EXPERIMENT_DEFAULT = {
    "name": "P1_default",
    "seeds": [42, 123, 456, 789],
    "total_timesteps": 50_000,
    "reward_settings": {},  # 全てデフォルト
}

# Random baseline
EXPERIMENT_RANDOM = {
    "name": "random_baseline",
    "seeds": [42, 123, 456, 789],
    "total_timesteps": 50_000,
    # ランダムアクション
}
```

#### B-2: 必須メトリクス（0番 §5.2 準拠）

```python
REQUIRED_METRICS = {
    # 収益性（0番 §5.2 Gate 2）
    "gross_pnl": "Gross PnL (JPY)",
    "net_pnl": "Net PnL (JPY)",
    "total_fees": "Total Fees (JPY)",
    "net_roi_pct": "Net ROI (%)",
    "profit_factor": "Profit Factor",
    "sharpe_ratio": "Sharpe Ratio (annualized)",
    "max_drawdown_pct": "Max Drawdown (%)",
    "win_rate_pct": "Win Rate (%)",
    
    # コスト分析（95#改善提案）
    "total_trades": "Total Trades",
    "turnover": "Turnover (sum |Δposition| × price)",
    "fee_rate_effective": "Effective Fee Rate (fees / turnover)",
    "cost_roi": "Cost ROI (fees / initial_balance)",
    
    # 取引特性
    "avg_holding_time_min": "Avg Holding Time (min)",
    "buy_sell_ratio": "BUY/SELL Ratio",
}
```

#### B-3: コスト評価軸の改善（95#指摘対応）

```python
# 不安定な cost_ratio に代えて
def calculate_stable_cost_metrics(gross_pnl, net_pnl, total_fees, 
                                   turnover, initial_balance):
    return {
        # 旧指標（参考のみ）
        "cost_ratio_legacy": total_fees / max(abs(gross_pnl), 1.0),
        # 新指標（安定）
        "fee_rate_effective": total_fees / max(turnover, 1.0),
        "cost_roi": total_fees / initial_balance,
        "net_margin": (gross_pnl - total_fees) / max(abs(gross_pnl), 1.0),
    }
```

### 4.3 Phase C: コスト最適化（推定: 4-8時間）

**前提条件**: Phase B で Gross PnL > 0 を確認済み

**目的**: 取引頻度を制御して Net PnL > 0 を達成

#### C-1: アクション閾値実験

```python
# v457.3の教訓: 過剰取引が構造的問題
ACTION_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
# |continuous_action| > threshold でのみ取引を実行
```

#### C-2: 段階的ペナルティ導入（v435.2の教訓: Curriculum）

```python
# Phase Bの結果を見てから決定
# ペナルティ係数はPnLの10%以下（v456の教訓）
PENALTY_SCALES = [0.0, 0.001, 0.005, 0.01]
```

### 4.4 Phase D: Phase 5 移行判定

#### 暫定Gate基準（95#提案を採用）

```yaml
Phase 5 移行条件:
  Gate 2 (収益):
    - Balance ROI > 0%（4seed平均）   # 最低ライン
    - Profit Factor > 1.1             # 最低ライン
    - Max Drawdown < 15%
  Gate 3 (ベースライン):
    - Model > Random（Mann-Whitney U, p < 0.05）
    - Cliff's Delta > 0.33
  
  未達の場合:
    - Phase 4.5 継続
    - 追加実験または設計見直し
```

#### 統計検定

```python
from scipy.stats import mannwhitneyu

def phase5_gate_check(model_results, random_results):
    """4 seedsの結果でPhase 5移行判定"""
    stat, p = mannwhitneyu(model_results, random_results, alternative='greater')
    # 効果量
    n1, n2 = len(model_results), len(random_results)
    effect = (stat / (n1 * n2) - 0.5) * 2  # rank-biserial相関 ≈ Cliff's Delta
    
    return {
        "p_value": p,
        "effect_size": effect,
        "mean_model": np.mean(model_results),
        "mean_random": np.mean(random_results),
        "gate_passed": p < 0.05 and effect > 0.33,
    }
```

---

## 5. タイムライン

```
Phase A: 報酬純粋性確立          ← 2-3時間（実装+テスト）
  A-1: calculate_reward_simple() 監査
  A-2: Gate 0.5 テスト作成・実行
  A-3: データリーク修正

Phase B: PnLベースライン         ← 4-8時間（4seeds × 50K steps）
  B-1: Pure PnL実験 (4 seeds)
  B-2: Default実験 (4 seeds)
  B-3: Random baseline (4 seeds)
  B-4: メトリクス収集・分析

Phase C: コスト最適化            ← 4-8時間（条件: Gross PnL > 0）
  C-1: アクション閾値実験
  C-2: 段階的ペナルティ導入

Phase D: Phase 5 判定            ← 2時間（分析+判定）
  D-1: 統計検定
  D-2: 0番§5.2基準照合
  D-3: Go/No-Go判定

推定合計: 12-21時間
```

---

## 6. 94#の結論の修正

### 修正前（94#）

> 1. 取引戦略自体は機能している（P1-1: Gross PnL > 0）  
> 2. 現行のペナルティは有害（P1-3: Gross PnL < 0）  
> 3. 問題は過剰取引（手数料が利益の42〜157倍）  

### 修正後

> 1. P1-1の Gross PnL > 0 は**暫定的な所見**。報酬経路の汚染により純PnL性能は未確定。  
> 2. ペナルティの有害性は**示唆されるが未確定**。P1-1自体が複合報酬だったため比較が成立しない。  
> 3. 過剰取引は**構造的に確認済み**（v457.1, v457.3でも同様の傾向）。  
> 4. Gate 0 は設定伝播問題を検出・修正できた点で**有効**。ただし有効化確認（Gate 0.5）が不足。  
> 5. **次のステップ**: 真のPnLのみ経路（`use_simple_reward=True`）での再検証が最優先。

---

## 7. リスク評価

| リスク | 影響 | 緩和策 |
|---|---|---|
| `calculate_reward_simple()`にも非PnL項がある | Phase A全体のやり直し | A-1で先に監査 |
| 4 seedsでも偶然の可能性 | Phase 5判定の信頼性低下 | 判定を「暫定」とし、本番前に追加検証 |
| 50K stepsでも学習不足 | ベースライン性能が低すぎる | v457.2の轍を踏まないよう100K stepも視野 |
| Gross PnL < 0 の場合 | 戦略自体の見直しが必要 | Phase Cをスキップし、報酬設計/特徴量を再検討 |

---

## 8. 撤退基準

| 条件 | 判断 |
|---|---|
| Phase B で全4 seedの Gross PnL < 0 | 戦略根本見直し（特徴量/アーキテクチャ/データ） |
| Phase C で Net PnL > 0 が3設定以上で達成不可 | ペナルティ設計の抜本変更 or v457.3方式（TTL固定） |
| 20回以上の実験でROI > 0%未達 | v459のスコープを縮小し、v460への引き継ぎを検討 |

---

## 参照

- [00# プロジェクト提案](00_project_proposal_v459.md) — §3.3報酬段階化、§5.2成功基準、§5.6統計仕様
- [66# 0番整合性検証](66_doc00_consistency_check.md) — 軌道修正の正当性評価
- [94# Gate0/PhaseB結果](94_gate0_phaseb_verification_results.md) — 現行データ（結論は要修正）
- [95# Gate0/PhaseBレビュー](95_gate0_phaseb_review.md) — 本計画のトリガー
- [93# 改訂版Pivot計画](93_revised_pivot_plan.md) — Gate 0導入の根拠
- [89# Phase4.5詳細実行計画](89_phase4.5_detailed_execution_plan.md) — P0-P4の元計画
- v444 BALANCE_PENALTY_ROOT_CAUSE — 設定伝播バグの教訓
- v456 報酬過剰設計の失敗 — ペナルティ積層の限界
- v457.1 Gross PnL+ / Net PnL- — コスト負けの実証
- v457.3 TTL固定 +36.8M — 過剰取引が構造問題の実証
