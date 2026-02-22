# Phase 5.3: Walk-Forward Fix Plan (Doc 10)

**Date**: 2026-01-22  
**Scope**: Doc09の未解決問題に対する解決策検討  
**Purpose**: Walk-Forward評価の信頼性確保と取引検知の実装

---

## 0. Executive Summary

Doc09で特定されたWalk-Forwardの問題（win_rate=0、取引検知不能）を解決するための包括的計画。過去事例の分析と既存資産の活用により、確実な修正を実装する。

---

## 1. 問題の再確認

### 1.1 Walk-Forwardでの取引検知不能
- **症状**: `win_rate: 0.0`, `profit_factor: Infinity`, `expectancy: 0.0`
- **原因**: `ztb/evaluation/walk_forward/evaluator.py` が `trade_pnl` を環境から取得できていない
- **影響**: 評価指標がすべて0になり、モデル性能が正しく評価されない

### 1.2 評価期間のランダム性
- **症状**: `FastIntradayEnvV456.reset()` がランダム開始
- **原因**: `max_steps` が環境に渡されていない
- **影響**: val/test区間が全期間をカバーしない可能性

### 1.3 過去事例の確認
- **MODEL_EVALUATION_STATUS.md**: v384でHOLD 100%、取引なし（win_rate=0）
- **原因**: 環境設定orモデルの問題（reward_penalty過大）
- **教訓**: rewardパラメータ調整で解決可能

---

## 2. 既存資産の活用分析

### 2.1 BacktestReporter (v457)
- **場所**: `scripts/v457/backtest_v457.py`
- **機能**: 完全な取引統計計算（trade_pnl, win_rate, profit_factor, expectancy）
- **実装**: `record_trade()` でnet_pnlを蓄積、`finalize_stats()` で指標計算
- **活用度**: ❌ 未使用（v458 Walk-ForwardでBacktestStatsRecorderをインポートしていない）

### 2.2 v456 Walk-Forward実装
- **場所**: `scripts/v456/phase4/modules/evaluator.py`
- **機能**: BacktestStatsRecorderを使用した取引検知
- **実装**: `recorder.record_trade()` で取引記録、`recorder.stats` から指標取得
- **活用度**: ❌ 未活用（v458で独自実装）

### 2.3 環境のtrade_pnl生成
- **場所**: `ztb/trading/environment/fast_intraday_env_v456.py`
- **機能**: `info` にtrade情報を含む
- **実装**: `info.get("trade_executed")`, `info.get("pnl")` など
- **活用度**: ❌ 未活用（Walk-Forward evaluatorがinfoを使用していない）

---

## 3. 解決策の提案

### 3.1 優先度P0: 取引検知の実装
1. **BacktestReporter統合**
   - `ztb/evaluation/walk_forward/evaluator.py` にBacktestReporterをインポート
   - `_evaluate_on_df()` でrecorderを使用した取引記録
   - `WindowPerformance` にtrade_pnlsを追加

2. **環境infoの活用**
   - `step()` のinfoからtrade情報を抽出
   - `recorder.record_trade()` で記録

3. **指標計算の統一**
   - win_rate, profit_factor, expectancyをBacktestReporter準拠に

### 3.2 優先度P1: 評価期間の固定化
1. **max_stepsの環境渡し**
   - `create_fast_intraday_env_v456()` にmax_stepsパラメータ追加
   - reset() で固定開始位置を使用

2. **全期間評価保証**
   - `env_factory` で `max_steps=len(df)` を強制

### 3.3 優先度P2: rewardパラメータ調整
1. **過去事例からの学習**
   - vol_floor_penalty: 20000000.0 → 0.0（Doc09既実施）
   - reward_scale: 100000.0 → 10000000.0（Doc09既実施）

2. **追加調整**
   - min_edge_mult: 1.5 → 1.0（取引しやすく）
   - edge_penalty_rate: 0.0（維持）

---

## 4. 実装計画

### Phase 1: BacktestReporter統合（1-2日）
1. `ztb/evaluation/walk_forward/evaluator.py` 修正
   - BacktestReporterインポート
   - `_evaluate_on_df()` でrecorder使用
   - trade情報抽出ロジック追加

2. `ztb/evaluation/walk_forward/types.py` 修正
   - `WindowPerformance` にtrade_pnlsフィールド追加

3. テスト実行
   - Walk-Forwardでwin_rate > 0 確認

### Phase 2: 評価期間固定化（1日）
1. `fast_intraday_env_v456_utils.py` 修正
   - max_stepsパラメータ追加

2. `walk_forward/evaluator.py` 修正
   - env_factoryでmax_steps=len(df)渡し

3. 全期間評価確認

### Phase 3: rewardチューニング継続（2-3日）
1. rewardパラメータ探索
   - min_edge_mult調整
   - 取引発生確認

2. 複数seedでの検証
   - 3-4 seedでWalk-Forward実行
   - 安定した取引発生確認

---

## 5. リスク評価

### 高リスク
- **環境変更の影響**: BacktestReporter統合で既存Walk-Forward破綻の可能性
- **指標計算の不一致**: v457 vs v458の計算差異

### 中リスク
- **パフォーマンス低下**: recorder追加で評価速度低下
- **メモリ使用増**: trade_history蓄積

### 低リスク
- **reward調整**: 既存パラメータ変更のみ

---

## 6. 成功基準

- ✅ Walk-Forwardで `win_rate > 0`
- ✅ `profit_factor < Infinity`
- ✅ `expectancy != 0.0`
- ✅ 全期間評価実行
- ✅ 複数seedで安定した結果

---

## 7. 次のステップ

1. **Phase 1実装開始**
2. **Doc11作成**: 実装結果と最終検証
3. **Doc09完了宣言**: 問題解決後

**Doc10 completion: 計画策定完了、実装準備整い**</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\v458\10_phase5_3_walk_forward_fix_plan.md