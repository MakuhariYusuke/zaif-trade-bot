# Project Proposal v459: "Alpha Resurrection" - 統合と収益化への最終章

**Date**: 2026-01-22  
**Status**: 📝 Planning  
**Predecessor**: v458 "Lost Alpha" Integration & Stabilization

---

## 1. Executive Summary

**v459 "Alpha Resurrection"** は v458 で残された重大課題を解決し、vXXXシリーズ全体の知見を集約して**本番運用可能な高収益システム**を構築する最終統合フェーズです。

### v458からの引き継ぎ状況

| 領域 | v458完了度 | v459での対応 |
|------|------------|--------------|
| Walk-Forward評価基盤 | 80% | P0バグ修正後の完成 |
| Entry Gate システム | 60% | Config配線・属性アクセス修正 |
| 指標整合性 | 50% | コスト二重計上・統計分離 |
| AB Testing | 30% | 複数シード比較の有効化 |
| 本番運用準備 | 20% | Paper Trading統合 |

### v459の大義

> **本プロジェクトの大義は「短期間での高収益性システム」の実現**

v459はこの大義達成のため、**これ以上の機能追加を止め、既存資産の完全統合と品質担保**に集中します。

---

## 2. vXXXシリーズ知見の集約

### 2.1 v451 "Golden Era" の教訓

| 成功要因 | 設定値 | v459への適用 |
|----------|--------|--------------|
| 短期指向の割引率 | γ = 0.80 | ✅ 継承 |
| Holdペナルティなし | 0.0 | ✅ 継承 |
| 非対称報酬 | Loss: 1.2倍 | ✅ 検討 |
| SAC採用 | SAC | ✅ 維持 |

**教訓**: シンプルな報酬設計が最も効果的だった。

### 2.2 v455 "HFT Stability" の教訓

| 成果 | 詳細 | v459への適用 |
|------|------|--------------|
| レバレッジ制御 | 10x → 1.37x | ✅ リスク管理 |
| メモリ最適化 | OnlineScaler改善 | ✅ 継承済み |
| 報酬パラメータ | min_edge_mult=1.5 | 🔄 オプション |

**教訓**: 安定性基盤は整った。Alpha（予測力）の注入が次の課題。

### 2.3 v456 "MTF Integration" の教訓

| 成果 | 詳細 | v459への適用 |
|------|------|--------------|
| 88次元観測空間 | MTF + Cyclical + Regime | ✅ 継承 |
| 環境Factory | factory_v456.py | ✅ 継承 |
| 7件のバグ修正 | P0-P2全て完了 | ✅ 基盤 |

**教訓**: 複雑な報酬シェーピングはPnL学習を阻害する。

### 2.4 v457 "Lost Alpha Recovery" の教訓

| 成果 | 詳細 | v459への適用 |
|------|------|--------------|
| Causal MTF | Future leakage修正 | ✅ 維持 |
| Trend Guidance | Ichimoku + Decay | ✅ 継承 |
| Cyclical Features | sin/cos時間特徴 | ✅ 継承 |

**教訓**: Bimodal Instability（seed依存）は報酬設計で回避可能。

### 2.5 v458 "Integration & Stabilization" の教訓

| 成果 | 課題 | v459での対応 |
|------|------|--------------|
| Walk-Forward基盤 | P0バグ4件残存 | 即時修正 |
| 複数seed検証 | 統計分離不十分 | Reporter分離 |
| Baseline比較 | 機能するが不正確 | コスト計上統一 |

**教訓**: インフラは整備されたが、実行詳細の品質が不十分。

---

## 3. v459技術方針

### 3.1 設計原則

1. **No New Features**: 機能追加凍結、統合と品質に集中
2. **Fix First**: v458 P0/P1バグを最優先で解決
3. **Consolidate**: 重複実装の統一（特にReporter）
4. **Validate**: エンドツーエンドの指標整合性検証
5. **Simplify**: 複雑な報酬設計を避け、v451志向に回帰

### 3.2 アーキテクチャ決定

```
v459 Stack (Confirmed)
├── Algorithm: SAC (Soft Actor-Critic)
├── Environment: FastIntradayEnvV456 (v457.5 patches)
├── Features: 88-dim (Base + MTF + Cyclical + Regime)
├── Reward: 段階的設計（下記参照）
├── Evaluation: Walk-Forward (Multi-Window, Multi-Seed)
└── Risk: Circuit Breaker + Virtual Portfolio Manager
```

### 3.3 報酬設計の段階化（Doc01レビュー対応）

> **方針**: 複雑な報酬設計を避け、純PnLでベースラインを取った後に段階的にガイダンスを追加

| Stage | 報酬関数 | 目的 |
|-------|----------|------|
| **Stage 1** | `R = PnL_net` | 純粋な収益性ベースライン |
| **Stage 2** | `R = PnL_net - 0.05 * TrendPenalty` | ガイダンス効果検証 |
| **Stage 3** | `R = PnL_net - W(t) * TrendPenalty` | カリキュラム効果検証 |

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

**検証方法**: 各Stageで同一条件（seed, データ期間）のAB比較を実施

### 3.3 コンフィグ戦略

```yaml
# v459 Config Philosophy
config/v459/
├── base/
│   └── config.yaml      # 唯一の正解設定
├── experiments/
│   └── *.yaml           # ABテスト用差分のみ
└── production/
    └── config.yaml      # 本番用（base継承）
```

---

## 4. 実装フェーズ（現実的見直し版）

> **注**: Doc01レビューを受け、工数見積もりと完了条件を厳格化

### Phase 0: 仕様固定（前提作業） - 1日

| 作業 | 成果物 | 完了条件 |
|------|--------|----------|
| Reporter I/O仕様 | 指標定義書 | gross/net規約確定 |
| Entry Gate仕様 | インターフェース定義 | 入出力型確定 |
| スケーラfit範囲 | リーク検査手順書 | 因果性保証手順 |
| 実行モデル定義 | コスト・遅延パラメータ | 数値確定 |

### Phase 1: Critical Bug Fixes (P0) - 2-3日

| Issue | 修正内容 | ファイル | テスト |
|-------|----------|----------|--------|
| Entry Gate Crash | `gate_result["should_enter"]`に修正 | fast_intraday_env_v456.py | 単体テスト |
| Entry Gate Config | env_configに配線 | v457_config_utils.py | 統合テスト |
| Cost Double-Count | PnL規約統一（env=net, reporter=検証のみ） | evaluator.py, reporter.py | E2Eテスト |
| Val/Test汚染 | Reporter分離（別インスタンス） | evaluator.py | 分離確認テスト |

**完了条件**: 全テスト合格 + 10エピソード手動検証

### Phase 2: Major Bug Fixes (P1) - 3-4日

| Issue | 修正内容 | ファイル | テスト |
|-------|----------|----------|--------|
| Trade Type分類 | "close"の明示処理 | evaluator.py, reporter.py | 統計検証 |
| Entry Price更新 | 反転時の価格更新 | fast_intraday_env_v456.py | PnL整合性 |
| Reporter統合 | 3実装→1実装 | reporter.py | 指標一致確認 |
| AB Testing有効化 | 複数Result収集→比較 | run_walk_forward_v458.py | 2seed比較動作 |

**完了条件**: Reporter統一 + AB Test動作確認

### Phase 3: 報酬設計の段階検証 - 3-4日

> **方針**: 純PnLでベンチマーク→Trend Guidance段階追加

| ステップ | 報酬設計 | 検証内容 |
|----------|----------|----------|
| 3.1 | 純PnL only | ベースライン性能 |
| 3.2 | PnL + Trend Guidance (固定) | ガイダンス効果 |
| 3.3 | PnL + Trend Guidance (Decay) | カリキュラム効果 |

**完了条件**: 各ステップのAB比較データ取得

### Phase 4: 評価・検証 - 4-5日

| 検証項目 | 条件 | 完了条件 |
|----------|------|----------|
| Walk-Forward | 4ウィンドウ × 4seed | 全組み合わせ完走 |
| リーク検査 | スケーラfit範囲確認 | 検査スクリプト合格 |
| Baseline比較 | BH, SMA, Random, Momentum | 4種全てで超過 |
| AB Testing | Gate ON/OFF, 報酬3種 | 統計的有意差検定 |

**完了条件**: 成功基準5.2の最低基準クリア

### Phase 5: Paper Trading統合 - 5-7日

| 作業 | 成果物 | 完了条件 |
|------|--------|----------|
| 実行モデル実装 | スリッページ/遅延シミュレータ | 実測値反映 |
| PaperTrader接続 | 統合スクリプト | 24時間連続動作 |
| リスク閾値設定 | Circuit Breaker設定 | 発動テスト合格 |
| 実行ログ収集 | 約定率・遅延ログ | 1週間データ蓄積 |

**完了条件**: 成功基準5.3, 5.4の基準クリア

### Phase 6: Go/No-Go判定 - 2日

| 判定項目 | 基準 | 判定者 |
|----------|------|--------|
| Gate 1 (技術) | 5.1全項目 | 自動テスト |
| Gate 2 (収益) | 5.2最低基準 | データ検証 |
| Gate 3 (リスク) | 5.3全項目 | 手動確認 |
| Gate 4 (実行) | 5.4全項目 | 実測データ |

**最終判定**: 全Gate通過でGo、1つでも未達でNo-Go（条件付き再検証）

---

## 5. 成功基準（厳格化版）

> **注**: Doc01レビューを受け、「動作確認」レベルから「収益判断」レベルへ引き上げ

### 5.1 技術検証（Gate 1: 必須）

- [ ] Walk-Forward評価がエラーなく完了
- [ ] Entry Gateが正常に動作（ON/OFF切替で挙動変化確認）
- [ ] 指標の二重計上なし（env PnL = reporter PnL）
- [ ] Val/Test統計が完全分離（別Reporterインスタンス）
- [ ] スケーラfit範囲がtrain期間に限定（リーク検査合格）

### 5.2 収益性検証（Gate 2: Go/No-Go判定軸）

| 指標 | 最低基準 | 目標基準 | 測定条件 |
|------|----------|----------|----------|
| **Net ROI** | > 5% | > 15% | 年率換算、コスト込み |
| **Profit Factor** | > 1.20 | > 1.50 | 手数料・スリッページ後 |
| **Sharpe Ratio** | > 1.0 | > 1.5 | 日次リターン、年率換算 |
| **Max Drawdown** | < 15% | < 10% | 高値からの最大下落 |
| **Win Rate** | > 35% | > 45% | 手数料込み勝敗 |
| **期待値/取引** | > ¥500 | > ¥1,000 | コスト控除後 |

### 5.3 リスク管理検証（Gate 3: 運用可否）

| 指標 | 基準 | 測定方法 |
|------|------|----------|
| **日次損失上限** | < 3% | 1日あたり最大損失 |
| **連敗時縮退** | 5連敗で50%縮小 | ポジションサイズ制御 |
| **回転率** | 10-50回/日 | 過剰取引防止 |
| **平均保有時間** | 5-60分 | HFT想定範囲 |

### 5.4 実行コスト検証（Gate 4: 実運用条件）

| 項目 | 想定値 | 許容範囲 |
|------|--------|----------|
| **手数料** | 0.1% | 片道 |
| **スリッページ** | 0.05% | 推定値 |
| **約定遅延** | < 500ms | Paper Trading実測 |
| **約定率** | > 95% | 指値注文想定 |

### 5.5 ベースライン比較（最終判定基準）

> **注**: Doc02との整合性を確保。こちらが最終判定基準。

| 比較対象 | 条件 | 判定 | 統計検定 |
|----------|------|------|----------|
| Buy-and-Hold | 同期間、同コスト、同ポジション | **必須超過** | 要 |
| SMA Crossover | 20/50期間、同条件 | **必須超過** | 要 |
| Random Action | 同頻度、同コスト | **必須超過** | 要 |
| Momentum (1h) | 1時間リターン追従 | **参考**（判定外） | 任意 |

### 5.6 統計検定仕様（Doc03対応）

| 項目 | 仕様 |
|------|------|
| **検定方法** | Mann-Whitney U検定（ノンパラメトリック） |
| **有意水準** | α = 0.05 |
| **多重比較補正** | Holm-Bonferroni法（3比較） |
| **効果量** | Cliff's Delta（|d| > 0.33で中程度） |
| **サンプル数** | 各条件n ≥ 16（4seed × 4split） |

```python
# 統計検定の実装例
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta

def compare_with_baseline(model_results, baseline_results, alpha=0.05):
    """
    モデルとベースラインの統計的比較
    
    Returns:
        is_superior: bool - 統計的に有意に優れているか
        p_value: float - 検定のp値
        effect_size: float - Cliff's Delta
    """
    stat, p_value = mannwhitneyu(
        model_results, baseline_results, 
        alternative='greater'  # 片側検定（モデル > ベースライン）
    )
    effect_size, _ = cliffs_delta(model_results, baseline_results)
    
    is_superior = (p_value < alpha) and (effect_size > 0.33)
    return is_superior, p_value, effect_size

def apply_holm_correction(p_values, alpha=0.05):
    """多重比較補正（Holm-Bonferroni）"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    corrected_alpha = [alpha / (n - i) for i in range(n)]
    
    rejections = []
    for i, idx in enumerate(sorted_indices):
        if p_values[idx] < corrected_alpha[i]:
            rejections.append(idx)
        else:
            break
    return rejections
```

### 5.7 運用準備

- [ ] Paper Trading 1週間で上記基準維持
- [ ] Circuit Breaker発動テスト完了
- [ ] 緊急停止の応答時間 < 1秒
- [ ] 実行ログの完全記録（スリッページ・遅延含む）

---

## 6. リスク評価

### 高リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| 指標不整合の見落とし | 誤ったモデル選択 | エンドツーエンド検証 |
| Entry Gate未動作 | Buy-onlyバイアス | 統合テスト必須 |
| Reporter重複による混乱 | 指標不一致 | 単一実装への統合 |

### 中リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| AB Testing不十分 | 統計的妥当性欠如 | 4seed以上で検証 |
| Calibration未学習 | 長期性能低下 | load_state実装 |

---

## 7. 既存資産活用マップ

### 最優先活用

| 資産 | パス | 用途 |
|------|------|------|
| Walk-Forward | `ztb/evaluation/walk_forward/*` | 評価基盤 |
| Reporter | `ztb/evaluation/walk_forward/reporter.py` | 統計統一 |
| Entry System | `ztb/trading/signal/entry_system.py` | フィルタリング |
| Circuit Breaker | `ztb/trading/production/circuit_breaker.py` | リスク管理 |

### 統合対象

| 資産 | パス | 用途 |
|------|------|------|
| AB Test Runner | `tools/ab_test_runner.py` | 実験管理 |
| Paper Trader | `ztb/trading/live/simulation/paper_trader.py` | 運用評価 |
| Risk Allocator | `ztb/trading/production/risk_based_allocator.py` | 段階的配分 |

---

## 8. タイムライン概要（現実的見直し版）

```
Week 1 (01/23-01/26)
├── Day 1: Phase 0 (仕様固定)
├── Day 2-4: Phase 1 (P0修正 + テスト)
└── Checkpoint: P0完了確認

Week 2 (01/27-01/31)
├── Day 1-4: Phase 2 (P1修正 + Reporter統合)
└── Checkpoint: 基盤安定確認

Week 3 (02/01-02/07)
├── Day 1-4: Phase 3 (報酬設計段階検証)
├── Day 5-7: Phase 4 前半 (評価・検証)
└── Checkpoint: 収益性初期確認

Week 4 (02/08-02/14)
├── Day 1-5: Phase 4 後半 + Phase 5 (Paper Trading)
├── Day 6-7: Phase 6 (Go/No-Go判定)
└── Final Decision: 02/14

予備期間: 02/15-02/21 (再検証・調整用)
```

**重要**: 各Checkpointで未達の場合は次フェーズに進まない

---

## 9. 次のステップ

1. **本ドキュメント承認後**: Phase 1即時開始
2. **P0修正完了後**: 検証テスト実行
3. **全Phase完了後**: Go/No-Go判定会議

---

**Status**: 📝 Planning → ⏳ Review Pending  
**Author**: GitHub Copilot  
**Date**: 2026-01-22
