# 50. Phase 4 Week 2 実装計画: 報酬調整 + 軽量MTF戦略

**日付**: 2026-01-28  
**前提**: Phase 4 Day 5 A/Bテスト完了（45番）、外部アドバイス受領（49番）  
**目標**: 50,000ステップで損益分岐点突破（-5% → 0%以上）  
**v459大義**: **「短期間での高収益性システム」の実現**（0番 Section 1）  
**Phase 4位置づけ**: 0番Phase 3（報酬設計検証）+ Phase 4前半（評価・検証）

---

## 0番プロポーザルとの整合性確認

### 現在位置の確認

| 0番フェーズ | 計画内容 | 実際の進捗 | 状態 |
|------------|---------|-----------|------|
| Phase 0-2 | 仕様固定 + P0/P1バグ修正 | v458で完了 | ✅ 完了 |
| **Phase 3** | **報酬設計の段階検証** | **Phase 4 Week 2で実施中** | 🔄 **進行中** |
| **Phase 4前半** | **評価・検証** | **Phase 4 Week 2で実施中** | 🔄 **進行中** |
| Phase 4後半 | Paper Trading統合 | Phase 5以降 | ⏳ 未着手 |
| Phase 5 | 本番運用 | Phase 6以降 | ⏳ 未着手 |

### 0番の報酬設計戦略との整合性

**0番 Section 3.3「報酬設計の段階化」**:
```python
# Stage 1: 純PnL（ベースライン）
reward = (current_balance - previous_balance) / initial_balance

# Stage 2: 固定ガイダンス
trend_penalty = 0.05 if action_opposes_ichimoku else 0
reward = pnl_net - trend_penalty

# Stage 3: Decay付きガイダンス
W = max(0, 1 - lifetime_steps / 50000)  # 50kステップで0に
reward = pnl_net - W * trend_penalty
```

**現状のギャップ**:
- ❌ Stage 1（純PnL）の実装・検証が未完了
- ❌ 現在の報酬関数は複雑すぎる（Hold penalty, Drawdown penalty等）
- ❌ 49番が指摘する「報酬不均衡（99.34%負）」は0番の設計思想と矛盾

**Week 2での対応方針**:
1. ✅ **0番Stage 1に立ち戻る**: 純PnL報酬でベースライン確立
2. ✅ **49番提案を段階的に適用**: Hold削除 → 取引抑制 → 探索調整
3. ✅ **MTFを再評価**: 0番v456/v457の教訓を活かした軽量MTF

---

## Phase 4 Day 5 A/Bテスト結果の分析

### 現状の課題（45番より）

```
ROI: -5.074%（8特徴Parquet平均）
負報酬比率: 99.34%
正報酬比率: 0.66%
SELL偏重: 47.4%
取引回数: 260-284回（平均275回）
1取引あたり損失: -18円（-0.018%）
```

### 根本原因の診断（49番 + 0番の統合分析）

#### 1. 報酬設計の問題（0番Phase 3未完了）
- **現状**: 複雑な報酬関数（Hold penalty, Drawdown penalty, Transaction cost等）
- **0番の意図**: 純PnL → 段階的ガイダンス追加
- **実態**: Stage 1が未実装のまま複雑化している

#### 2. 過剰売買による構造的損失
- **取引回数**: 275回/50,000ステップ = 0.55%
- **往復コスト**: 0.2% × 275回 = 55% ≈ 55,000円相当
- **実際の損失**: -5,074円
- **分析**: **売買頻度抑制だけで大幅改善の可能性**

#### 3. 1分足ノイズとMTF不在
- **0番v456/v457の教訓**: MTF + Cyclical + Regimeで88次元観測空間
- **Phase 4 Day 1-3**: MTF強制無効化（44番）→ 8特徴のみに削減
- **現状**: 市場構造（トレンド/レンジ）を捉える特徴が不足
- **結果**: SELL偏重47.4%（レンジ相場での誤判定）

---

## Week 2実装計画

### 設計方針の明確化

#### ❌ 避けるべきアプローチ
1. **複雑な報酬追加**: 0番v456の失敗を繰り返さない
2. **フルMTF復活**: Phase 3.5の99.83%削減効果を失う
3. **1分足→5分足変更**: データ再生成の工数大

#### ✅ 採用するアプローチ
1. **0番Stage 1への回帰**: 純PnL報酬でシンプル化
2. **49番提案の段階適用**: 報酬調整 → 取引抑制 → 探索調整
3. **軽量MTF追加**: 0番の88次元思想を8→16特徴で実現

---

## Day 6-7: 報酬調整A/Bテスト（49番 優先1-3）

### 目的
50,000ステップで損益分岐点突破（-5% → 0%以上）

### 実験設計

| 実験ID | 報酬設計 | 期待効果 | 0番との関係 |
|--------|---------|---------|-----------|
| **A (Baseline)** | 現状（複雑） | -5.074% | - |
| **B (Stage 1)** | 純PnL only | ベースライン確立 | 0番Stage 1実装 |
| **C (49番-1)** | B + Hold削除 + Drawdown-1.0 | 報酬バランス改善 | Stage 1改良 |
| **D (49番-2)** | C + action_change_penalty | 取引回数50%削減 | 49番優先2 |
| **E (49番-3)** | D + ent_coef=0.01 + gamma=0.95 | 早期収束 | 49番優先3 |

### 具体的な実装

#### 実験B: 純PnL報酬（0番Stage 1）
```python
# ztb/training/reward_config.py
class RewardSettings:
    # すべてのペナルティを無効化
    hold_penalty: float = 0.0
    transaction_cost_penalty: float = 0.0
    drawdown_penalty_scale: float = 0.0
    
    # 純粋なPnL計算
    def calculate_reward(self, env_state):
        pnl = (env_state.portfolio_value - env_state.previous_value) / env_state.initial_capital
        return pnl
```

#### 実験C: Hold削除 + Drawdown縮小（49番優先1）
```python
class RewardSettings:
    hold_penalty: float = 0.0  # 削除
    drawdown_penalty_scale: float = -1.0  # -10.0 → -1.0に縮小
    
    # 報酬クリップ
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
```

#### 実験D: アクション変化ペナルティ（49番優先2）
```python
class RewardSettings:
    # 継承: 実験Cの設定
    
    # 新規追加
    action_change_penalty: float = 0.001  # |a_t - a_{t-1}|にペナルティ
```

#### 実験E: SAC探索調整（49番優先3）
```python
# config/v459/experiments/reward_tuning_e.yaml
algorithm:
  sac:
    learning_rate: 0.0005  # 0.0003 → 0.0005
    ent_coef: 0.01  # auto → 固定0.01（探索抑制）
    gamma: 0.95  # 0.99 → 0.95（短期志向）
    batch_size: 128  # 256 → 128（早期収束）
    gradient_steps: 2  # 1 → 2（更新頻度増）
```

### 実行仕様
```bash
# scripts/v459/run_reward_tuning_ab_test.py
Seeds: [42, 123]
Timesteps: 50,000/実験
Total experiments: 5 configs × 2 seeds = 10実験
Estimated time: 10 × 43分 = 7.2時間
```

### 成功基準
- **必須**: ROI -5% → 0%以上（1実験以上）
- **目標**: 取引回数50%削減（275回 → 140回以下）
- **期待**: 正報酬比率10%以上（0.66% → 10%）

---

## Day 8: 軽量MTF実装（0番 + 49番統合戦略）

### 目的
Phase 3.5の99.83%削減効果を維持しつつ、0番v456/v457の88次元思想を部分採用

### 設計思想

#### 0番の特徴設計（v456/v457）
```
88次元 = Base(市場特徴) + MTF(複数時間軸) + Cyclical(時間周期) + Regime(市場状態)
```

#### Phase 4での軽量化
```
16特徴 = 8特徴(Base) + 8特徴(軽量MTF)

8特徴MTF内訳:
1. MA_5m_trend: 5分足移動平均の傾き
2. MA_15m_trend: 15分足移動平均の傾き
3. volatility_5m: 5分足ボラティリティ
4. volatility_ratio_5m_to_1m: 5分/1分ボラティリティ比
5. regime_5m: 5分足レジーム（トレンド=1, レンジ=0）
6. price_position_15m: 15分足レンジ内の価格位置（0-1）
7. volume_trend_5m: 5分足出来高トレンド
8. momentum_15m: 15分足モメンタム
```

### 実装方針

#### 1. 事前計算スクリプト
```python
# scripts/v459/create_lightweight_mtf_features.py

import pandas as pd
import numpy as np

def add_lightweight_mtf(df: pd.DataFrame) -> pd.DataFrame:
    """8特徴Parquetに+8 MTF特徴を追加（合計16特徴）"""
    
    # 1. 5分足リサンプリング
    df_5m = df.resample('5T', on='timestamp').agg({
        'close': 'last',
        'volume': 'sum',
        'high': 'max',
        'low': 'min'
    })
    
    # 2. MTF特徴計算
    df_5m['MA_5m'] = df_5m['close'].rolling(20).mean()
    df_5m['MA_5m_trend'] = (df_5m['MA_5m'] - df_5m['MA_5m'].shift(1)) / df_5m['MA_5m'].shift(1)
    
    # 3. 1分足へマージ（forward fill）
    mtf_features = df_5m[['MA_5m_trend', ...]].reindex(df.index, method='ffill')
    
    # 4. 統合
    df_with_mtf = pd.concat([df, mtf_features], axis=1)
    
    return df_with_mtf
```

#### 2. 実行時間の見積もり
```
Phase 3.5実績: 466秒 → 0.79秒（99.83%削減）
軽量MTF追加: +0.5-1.0秒（resample処理）
合計: 1.5-2.0秒
削減率維持: 99.6%以上
```

#### 3. ファイルサイズ
```
8特徴Parquet: 13.4 MB
16特徴Parquet: 約25 MB（推定）
```

### 成功基準
- ✅ 生成時間 < 2秒
- ✅ ファイルサイズ < 30 MB
- ✅ 特徴欠損率 < 1%

---

## Day 9: MTF A/Bテスト

### 実験設計

| 実験ID | 特徴構成 | 報酬設計 | 目的 |
|--------|---------|---------|------|
| **F** | 8特徴 | 最適報酬（Day 6-7結果） | ベースライン |
| **G** | 16特徴（+MTF） | 最適報酬（同上） | MTF効果検証 |

### 評価指標

| 指標 | 目標 | 0番基準との対応 |
|------|------|---------------|
| **ROI** | 0%以上 | Gate 1: 手数料込みプラス |
| **SELL偏重** | 35%以下 | アクション均衡 |
| **取引回数** | 140回以下 | 過剰取引防止 |
| **正報酬比率** | 10%以上 | 報酬バランス |
| **Sharpe Ratio** | 0.5以上 | Gate 2: リスク調整後収益 |

### 実行仕様
```bash
Seeds: [42, 123]
Timesteps: 50,000/実験
Total experiments: 2 configs × 2 seeds = 4実験
Estimated time: 4 × 43分 = 2.9時間
```

---

## Day 10: 統合分析とPhase 4完了判定

### 統合分析内容

#### 1. 報酬調整効果の定量評価
```python
# A → B → C → D → E の改善パス
baseline_roi = -5.074%
pure_pnl_roi = ?  # 実験B
hold_removed_roi = ?  # 実験C
trade_reduced_roi = ?  # 実験D
exploration_tuned_roi = ?  # 実験E

improvement_path = {
    "純PnL効果": pure_pnl_roi - baseline_roi,
    "報酬バランス効果": hold_removed_roi - pure_pnl_roi,
    "取引抑制効果": trade_reduced_roi - hold_removed_roi,
    "探索調整効果": exploration_tuned_roi - trade_reduced_roi
}
```

#### 2. MTF効果の定量評価
```python
# F vs G の比較
mtf_effect = {
    "ROI改善": roi_g - roi_f,
    "SELL偏重是正": (sell_ratio_f - sell_ratio_g) / sell_ratio_f,
    "取引回数変化": trades_g - trades_f,
    "正報酬比率改善": positive_ratio_g - positive_ratio_f
}
```

#### 3. 0番Gate 1-2の判定

**Gate 1: 収益性（必須）**
```
目標: 手数料込みプラス（ROI > 0%）
測定: 50,000ステップでの最終ROI
判定基準: 5実験中2実験以上でROI > 0%
```

**Gate 2: 安定性（必須）**
```
目標: Sharpe Ratio > 0.5
測定: リターンの標準偏差で調整
判定基準: 5実験平均でSharpe > 0.3（緩和基準）
```

**Gate 3-4: リスク管理・実行コスト（Phase 5で評価）**

### Phase 4完了判定

#### ✅ 合格条件
1. **ROI改善**: -5% → 0%以上（2/5実験以上）
2. **取引抑制**: 275回 → 140回以下（50%削減）
3. **報酬バランス**: 正報酬比率10%以上
4. **MTF効果**: SELL偏重47% → 35%以下

#### ⚠️ 条件付き合格
- ROI: -2% ～ 0%（微損だが改善トレンド明確）
- 取引抑制: 30%削減達成
- 正報酬比率: 5%以上

#### ❌ 不合格（Phase 4延長）
- ROI: -3%以下（改善不十分）
- 取引回数: 削減率20%未満
- 正報酬比率: 3%未満

### Phase 5移行判断

**合格時**:
- Phase 5: 長期学習（500,000ステップ）+ バックテスト（3-6ヶ月）
- 目標: Gate 3-4の検証、Paper Trading統合準備

**条件付き合格時**:
- Phase 4.5: 追加調整（Learning Rate Schedule, Curriculum Learning等）
- 期間: 2-3日
- 目標: ROI 0%確実突破

**不合格時**:
- Phase 4再設計: 根本的な戦略見直し
- 選択肢: 離散アクション空間、時間足変更、報酬関数全面刷新

---

## 0番との整合性確認（最終チェック）

### Phase 3: 報酬設計の段階検証

| 0番Stage | 計画 | Week 2実装 | 状態 |
|---------|------|-----------|------|
| Stage 1 | 純PnL only | 実験B | ✅ 対応 |
| Stage 2 | PnL + Trend Guidance (固定) | （Phase 5で検討） | ⏳ 保留 |
| Stage 3 | PnL + Trend Guidance (Decay) | （Phase 5で検討） | ⏳ 保留 |

**Week 2の方針**: 
- 0番Stage 1を優先実装
- Trend Guidanceは49番提案（取引抑制、探索調整）を先行
- Stage 2-3はPhase 5で検討（収益性確立後）

### Phase 4前半: 評価・検証

| 0番項目 | 計画 | Week 2実装 | 状態 |
|--------|------|-----------|------|
| Walk-Forward | 4ウィンドウ × 4seed | 50,000ステップ × 2seed | 🔄 簡略版 |
| リーク検査 | スケーラfit範囲確認 | Phase 5で実施 | ⏳ 延期 |
| Gate 1-2 | 収益性・安定性検証 | Day 10で判定 | ✅ 対応 |
| Gate 3-4 | リスク・実行コスト | Phase 5で実施 | ⏳ 延期 |

**Week 2の簡略化理由**:
- 50,000ステップで損益分岐点突破が最優先
- 長期評価（4ウィンドウ × 4seed）はPhase 5で実施
- Gate 1-2を先行検証、Gate 3-4は本番運用準備で実施

---

## リスクと対策

### リスク1: 報酬調整が効果不十分
**確率**: 中  
**対策**: 
- 実験B（純PnL）で正報酬比率が1%未満の場合、報酬スケーリング（×10）を追加
- 実験D（取引抑制）で取引回数が200回以上の場合、action_change_penaltyを0.005に増強

### リスク2: MTF追加が逆効果
**確率**: 低  
**対策**: 
- 実験GのROIが実験Fより悪化した場合、MTF特徴を5分足のみに削減（16特徴 → 12特徴）
- MTF生成時間が3秒超過の場合、15分足特徴を削除

### リスク3: 50,000ステップでは収益化不可能
**確率**: 中  
**対策**: 
- Day 10判定で「条件付き合格」の場合、Phase 4.5で100,000ステップ実験を追加
- 根本的な収益化困難の場合、0番Phase 5（Paper Trading）を先行し、実市場データで検証

---

## 成果物と完了条件

### Day 6-7成果物
- [ ] 報酬調整実験スクリプト（run_reward_tuning_ab_test.py）
- [ ] 5設定の報酬設定ファイル（config/v459/experiments/reward_*.yaml）
- [ ] A/Bテスト結果JSON（results/phase4_day6_7_reward_tuning/）
- [ ] 報酬調整効果分析レポート（docs/v459/51_reward_tuning_results.md）

### Day 8成果物
- [ ] 軽量MTF生成スクリプト（scripts/v459/create_lightweight_mtf_features.py）
- [ ] 16特徴Parquet（data/btc_jpy_1m_v451_optimized_mtf_features.parquet）
- [ ] MTF生成時間測定結果（< 2秒確認）

### Day 9成果物
- [ ] MTF A/Bテスト実験スクリプト（run_mtf_ab_test.py）
- [ ] A/Bテスト結果JSON（results/phase4_day9_mtf_test/）
- [ ] MTF効果分析レポート（docs/v459/52_mtf_effect_results.md）

### Day 10成果物
- [ ] 統合分析レポート（docs/v459/53_phase4_week2_summary.md）
- [ ] Gate 1-2判定結果（0番基準との対応表）
- [ ] Phase 5移行判断（合格/条件付き/不合格）

### 完了条件
- ✅ ROI改善: -5% → 0%以上（2/10実験以上）
- ✅ 取引抑制: 50%削減達成
- ✅ 報酬バランス: 正報酬比率10%以上
- ✅ 全実験の結果JSON保存完了
- ✅ 統合分析レポート完成
- ✅ Phase 5移行判断完了

---

## タイムライン

```
Phase 4 Week 2（Day 6-10）: 2026-01-29 ～ 2026-02-02

Day 6-7（1/29-1/30）: 報酬調整A/Bテスト
  - 実験設計: 4時間
  - 実験実行: 7.2時間（バックグラウンド実行可能）
  - 分析: 4時間
  
Day 8（1/31）: 軽量MTF実装
  - スクリプト作成: 4時間
  - 特徴生成実行: 0.5時間
  - 検証: 2時間
  
Day 9（2/1）: MTF A/Bテスト
  - 実験実行: 2.9時間（バックグラウンド実行可能）
  - 分析: 4時間
  
Day 10（2/2）: 統合分析とPhase 4完了判定
  - 統合分析: 4時間
  - Gate判定: 2時間
  - 報告書作成: 3時間
```

---

## まとめ

Phase 4 Week 2は**0番プロポーザルの中核（Phase 3報酬設計 + Phase 4前半評価）**を実行する重要なマイルストーンです。

**3つの柱**:
1. **0番Stage 1への回帰**: 純PnL報酬でシンプル化
2. **49番提案の段階適用**: 報酬調整 → 取引抑制 → 探索調整
3. **軽量MTF追加**: 0番の88次元思想を16特徴で実現

**最終目標**:
- 50,000ステップで損益分岐点突破（ROI 0%以上）
- 0番Gate 1-2合格
- Phase 5（長期学習 + バックテスト）への移行承認

**v459大義の実現**:
このWeek 2で**「短期間での高収益性システム」の実現可能性**を実証します。

---

**次のアクション**:
1. Day 6開始: 報酬調整実験スクリプト作成
2. 実験B（純PnL）から順次実行
3. 日次で進捗報告、必要に応じて計画修正

**文書管理**:
- 作成: 2026-01-28
- 次回更新: Day 10（統合分析完了時）
