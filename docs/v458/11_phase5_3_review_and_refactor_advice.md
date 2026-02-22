# Phase 5.3 Review: Reuse & Refactor Advice (Doc 11)

**Date**: 2026-01-22  
**Scope**: `docs/v458/10_phase5_3_walk_forward_fix_plan.md` + vXXX資産整理  
**Purpose**: 過去資産の活用度・重複実装・リファクタリング余地を明確化し、修正方針を具体化する

---

## 0. Executive Summary
- Doc10の方向性は正しいが、**前提に誤りがあり、そのまま実装すると破綻する**。  
- 特に **env infoに trade_executed/trade_pnl が存在しない**ため、BacktestReporter統合は設計変更が必要。  
- v456/v457の資産は十分に活用されておらず、**指標定義と評価の統一**が最優先。

---

## 1. Doc10の改善点（正確性・実現性）

### 1.1 env info前提の誤り
- Doc10は `trade_executed` / `entry_price` / `exit_price` が env info に存在すると想定しているが、  
  **FastIntradayEnvV456のinfoはそれらを返していない**。  
  - 参照: `ztb/trading/environment/fast_intraday_env_v456.py`  
  - 影響: BacktestReporter統合は **追加の取引記録ロジックが必要**

### 1.2 v456 evaluatorの再利用は破綻
- `scripts/v456/phase4/modules/evaluator.py` は `BacktestStatsRecorder` を import するが実体が無い。  
- **そのまま活用不可**。再利用するなら「薄いラッパー化」か削除が妥当。  

### 1.3 評価期間固定化の実装ポイント不足
- `create_fast_intraday_env_v456()` の `known_utils_keys` に `max_steps` が含まれていない。  
- **max_stepsは渡せない状態**なので、Doc10の指示だけでは改善されない。  

### 1.4 Doc10の末尾にノイズ
- `</content>` や `<parameter ...>` が末尾に残っている。  
  **文書として不要なので除去推奨**。

---

## 2. vXXX資産の再利用ポイント（必須）

### 2.1 v456: Walk-Forward統合資産
- `ztb/evaluation/walk_forward/*` は **本来の正規系統**。  
- `tests/unit/evaluation/test_walk_forward_*` を再実行し、改修の回帰検証に使うべき。  
- `docs/v456/41_METRICS_INTEGRATION_MEMO.md` の「公式メトリクス統一方針」に沿う。

### 2.2 v456: 統合評価アダプタ
- `ztb/analysis/evaluation/walk_forward_adapter.py` で  
  **overfitting / consistency / robustness の定義が整理済み**。  
- Doc10の修正は **adapter側と整合する形で統一**する必要がある。

### 2.3 v457: BacktestReporter
- `scripts/v457/backtest_v457.py` の `BacktestReporter` は  
  **PF / Expectancy / AvgWinLoss の基準実装**として使える。  
- ただし env info不足を補う「trade_recordingアダプタ」が必須。

### 2.4 v455/v456: Entry Gate
- `ztb/trading/signal/entry_system.py` / `calibration_map.py`  
  は **無効トレード抑制の既存資産**。  
- Reward調整の前に、**ゲート導入で取引品質を上げる余地**がある。

---

## 3. リファクタリング余地（重複削減）

1) **WalkForwardResult二重定義の解消**  
   - `ztb/evaluation/walk_forward/types.py` と `result.py` を統一する。  

2) **Phase4 legacy modulesの整理**  
   - `scripts/v456/phase4/modules/*` は **re-exportか削除**。  
   - 直接修正するより、`ztb/evaluation/walk_forward/*` を唯一の正規系にする。

3) **overfitting_ratioの式統一**  
   - `WindowPerformance.overfitting_ratio`  
   - `run_walk_forward_v458.py`  
   - `walk_forward_adapter.py`  
   の定義が分裂。**1つに統一して再計算**。

4) **Trade統計の責務分離**  
   - envが `trade_pnl` を提供しない以上、  
     `evaluator` が tradeのopen/closeを判定する層になる。  
   - ここは **BacktestReporter互換の薄いアダプタを新設**するのが安全。

---

## 4. 修正方針（Doc10を上書きする推奨版）

### P0: 評価の正確性を確保
1) **max_stepsの引き渡し**  
   - `fast_intraday_env_v456_utils.py` の `known_utils_keys` に `max_steps` 追加。  
2) **固定開始位置**  
   - `reset(options={"start_index": ...})` のような明示指定を追加。  
   - これにより **val/test全区間評価が保証**される。  
3) **overfitting_ratioの統一**  
   - `WindowPerformance.overfitting_ratio` に統一し、全てそこを参照。  

### P1: Trade統計の再設計（最重要）
1) **Position差分でtradeを判定**  
2) **entry/exitを内部記録**  
3) **BacktestReporter互換形式で集計**  
   - v457の基準と揃え、PF/Expectancy/AvgWinLossを算出可能にする。  

### P2: vXXX資産の統合強化
1) **walk_forward_adapter活用**（統合評価）  
2) **tools/ab_test_runner.py** で reward/threshold探索  
3) **Entry Gate導入**で無意味な取引を削減

---

## 5. Doc11 完了時のチェックリスト
- Walk-Forward評価が **全期間固定**で実行される  
- WinRate/Profit Factor/Expectancy が **0やInfinityでない**  
- overfitting_ratio の計算式が **一箇所に統一**  
- tests/unit/evaluation の既存テストで回帰確認  
- legacy modules の重複が削減される

---

## 6. コメント
Doc10は「修正方針の方向性」として有効だが、  
**過去資産を正確に引き継ぐための前提整備が不足**している。  
まず「評価の正確性」と「指標の定義統一」を先に完了させるべき。
