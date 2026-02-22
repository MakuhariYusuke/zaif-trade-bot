# Phase 5.4 Review & Remaining Gaps (Doc 13)

**Date**: 2026-01-22  
**Scope**: `docs/v458/12_phase5_4_implementation_results.md` + code review  
**Purpose**: 実装内容の妥当性確認、コードレビュー指摘、未達事項の整理

---

## 1. Code Review Findings（重大度順）

### CRITICAL
1) **trade記録で即クラッシュする可能性**
   - `ztb/evaluation/walk_forward/evaluator.py`
   - `info.get("entry_price", eval_env.close)` の **default引数評価で `eval_env.close` を参照**  
     `FastIntradayEnvV456` に `close` 属性は無く、trade発生時に `AttributeError` が出る。

2) **全期間評価が未達**
   - `ztb/trading/environment/fast_intraday_env_v456.py` は `reset()` がランダム開始のまま。  
   - `scripts/v458/run_walk_forward_v458.py` は **env_configに `max_steps` を渡していない**。  
   - `_evaluate_on_df()` の `max_steps` はループ制限のみで、**start_index固定にはならない**。

3) **trade統計の根拠が不正確**
   - `ztb/evaluation/walk_forward/evaluator.py`
   - `pnl = eval_env.balance - prev_balance` は **「取引の損益」ではなく1ステップの評価**。  
   - `fee/slippage` が env info に存在せず（`fee_paid/slippage_paid` が正）、  
     **PF/Expectancyが歪む**。

4) **Doc12結果と判定ロジックが矛盾**
   - `ztb/evaluation/walk_forward/result.py` は ROI≥1.05 / Sharpe≥0.5 / PF≥1.05 を要求。  
   - Doc12の例では `average_test_roi=-0.098` なのに `is_robust=true`。  
   - これは **実装結果との不整合**（または記録ミス）に該当。

### HIGH
5) **Sharpe Consistency 指標が破綻**
   - `scripts/v458/run_walk_forward_v458.py`
   - `1 - (std/mean)` は **meanが負の場合に1を超える**。  
     Doc12の `1.4` は指標定義として不正確。

6) **trade種別の判定が不正確**
   - `ztb/evaluation/walk_forward/evaluator.py`
   - `trade_type = "long" if current_position > 0 else "short"` は  
     **クローズ時（0）やロング→ショート反転で誤判定**する。

7) **`run_walk_forward_v458.py` がエラー時に落ちる**
   - `performances.append(perf)` の `perf` が `None` の場合、  
     集計 (`p.val_roi`) でクラッシュする設計のまま。

### MEDIUM
8) **BacktestReporter import が脆い**
   - `ztb/evaluation/walk_forward/evaluator.py`
   - `from scripts.v457.backtest_v457 import BacktestReporter` は  
     namespace package前提で、実行環境によっては失敗しうる。

9) **WalkForwardResult 二重定義が残存**
   - `ztb/evaluation/walk_forward/types.py` と `result.py` が並存。  
   - `walk_forward_adapter.py` は types 側を参照しており、**整合性が崩れる**。

---

## 2. Doc12の主張と実装のズレ

- **「全期間評価保証」**  
  → resetはランダムのまま。`max_steps` は env_config に注入されていない。  

- **「複数seedで安定 (3ウィンドウ)」**  
  → 3ウィンドウは seed 数ではない。**4 seeds検証の証跡が不足**。  

- **「profit_factor 2.8 / expectancy 1250」**  
  → ROIが負なのにPFが高いのは、**trade損益の定義が不整合**な兆候。  

- **「tests/unit/evaluation 全通過」**  
  → 既存テストは `_evaluate_on_df()` をほぼ検証していない。  
     **実動作の検証としては不足**。

---

## 3. 未達事項（Doc11以前から継続）

### Doc11チェックリストで未完
- **固定開始位置の導入**（`reset(options={"start_index": ...})` 等）  
- **overfitting_ratio 定義の完全統一**（`walk_forward_adapter` との整合）  
- **legacy modules の削減**（`scripts/v456/phase4/modules/*`）  
- **Doc10末尾のノイズ除去**（`</content>`, `<parameter>` が残存）

### Doc09/Doc05/Doc04から未完
- **4 seeds評価の正式実施**  
- **baseline比較 (buy/hold, flat, short) の標準化**  
- **ABテスト / 探索 (`tools/ab_test_runner.py`) の活用**  
- **Entry Gate / Calibration Gate の導入**  
- **Paper trading / risk管理統合の検証**

---

## 4. 改善アドバイス（最優先順）

### P0: 評価の正確性
1) `FastIntradayEnvV456.reset()` に **start_index指定オプション**を追加  
2) `env_factory` で **max_steps / start_index** を明示注入  
3) `entry_price/exit_price` の default を **`close_prices[current_step]`** へ修正  
4) `fee/slippage` は **`fee_paid/slippage_paid`** を使用する

### P1: trade統計の再設計
1) position差分ではなく **open/closeの状態遷移**を追跡  
2) 取引確定時にのみ `record_trade()`  
3) **ROI と PF/Expectancy が矛盾しない形**で集計

### P2: 統一・重複削減
1) WalkForwardResult を **1定義に統合**  
2) walk_forward_adapter の統計と **同一式に統一**  
3) legacy modules の削除 or re-export に整理

---

## 5. コメント
Doc12は進展があるが、**「評価が正しい」前提をまだ満たしていない**。  
まずは「全期間評価」「trade統計の正当性」「指標統一」をクリアし、  
その後に性能改善の是非を判断するべき。
