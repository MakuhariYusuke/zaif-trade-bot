# Phase 5.5 Implementation Plan for Remaining Gaps

## 概要

Doc13で特定されたMEDIUM issuesの残課題を実装する計画。Doc15のレビューに基づき、既存実装を最大限活用し、省力的に実装する。既存資産（tools/ab_test_runner.py, BaselineComparisonEngineなど）を優先的に再利用。

**対象課題:**
- 4-seed evaluation (複数seedでの堅牢性評価)
- baseline comparisons (ベースラインモデルとの比較)
- AB testing (A/Bテストフレームワーク)
- entry gates (エントリー条件のゲート制御)

## 1. 4-seed evaluation

### 現状分析
- `tools/ab_test_runner.py`にmulti-seed評価機能が既存
- `tools/ab_param_search.py`もmulti-seed対応
- 同一window×複数seed評価が標準で可能

### 実装計画
**拡張内容:**
- seed数を4に固定 (`tools/ab_test_runner.py --seeds 4`)
- seedごとの統計を集計
- 堅牢性スコアの計算 (seed間変動の低さ)

**活用する既存実装:**
- `tools/ab_test_runner.py` (multi-seed, 集計済み)
- `tools/ab_param_search.py`
- `ztb/evaluation/walk_forward/evaluator.py` (evaluate_window_with_model)

**実装ステップ:**
1. `tools/ab_test_runner.py` を主軸に4-seed評価を実装
2. seedごとの評価結果を集計
3. 堅牢性指標を追加 (seed間標準偏差)

## 2. baseline comparisons

### 現状分析
- `ztb/analysis/baseline_comparison.py`にBaselineComparisonEngineが存在
- `ztb/analysis/regime_eval.py`にbaseline比較込みの評価フロー
- `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:298`に比較機能

### 実装計画
**拡張内容:**
- Walk-Forward結果をベースラインと比較
- 統計的有意性の検定
- 優位性メトリクスの計算

**活用する既存実装:**
- `ztb/analysis/baseline_comparison.py` (BaselineComparisonEngine)
- `ztb/analysis/evaluation/walk_forward_integration_pipeline.py` (compare_with_baseline)
- `ztb/analysis/regime_eval.py` (baseline比較込みの評価フロー)

**実装ステップ:**
1. BaselineComparisonEngine + walk_forward_integration_pipeline を接続
2. ベースライン戦略の実行と結果比較
3. 比較レポートの生成

## 3. AB testing

### 現状分析
- `tools/ab_test_runner.py` / `tools/ab_param_search.py`が既存
- `ztb/adaptation/ab_test/*`にAB機構
- `ztb/training/unified_optimizer.py`（AB機構内蔵）
- `experiments/v450/run_ab_test_threshold_v450.py`が実装例

### 実装計画
**拡張内容:**
- A/Bテストフレームワークの活用
- モデルA vs モデルBの比較
- 統計的有意性検定

**活用する既存実装:**
- `tools/ab_test_runner.py` / `tools/ab_param_search.py`
- `ztb/adaptation/ab_test/*`
- `ztb/training/unified_optimizer.py`（AB機構内蔵）
- `experiments/v450/run_ab_test_threshold_v450.py`

**実装ステップ:**
1. 既存フレームワークをそのまま使う（新規ztb/analysis/ab_testing.pyは作成せず）
2. Walk-Forward結果のA/B比較機能
3. 統計的有意性検定の統合

## 4. entry gates

### 現状分析
- `ztb/trading/signal/entry_system.py`にIntegratedEntrySystemが存在
- `ztb/trading/signal/calibration_map.py`にCalibrationGate
- `ztb/trading/environment/components/position_manager.py`（hybrid filters）

### 実装計画
**拡張内容:**
- IntegratedEntrySystem を前段フィルタとして導入
- エントリー条件の動的制御
- gate統計の収集

**活用する既存実装:**
- `ztb/trading/signal/entry_system.py` (IntegratedEntrySystem)
- `ztb/trading/signal/calibration_map.py` (CalibrationGate)
- `ztb/trading/environment/components/position_manager.py`（hybrid filters）

**実装ステップ:**
1. IntegratedEntrySystem を前段フィルタとして導入（envに新規ロジック追加せず）
2. gate統計の収集機能
3. 設定によるgateの有効/無効制御

## 実装スケジュール

### Phase 1: 基盤整備 (1日)
- 各課題の既存コード調査完了
- 統合ポイントの特定
- テストケースの準備

### Phase 2: 4-seed evaluation (0.5日) ✅ 完了
- tools/ab_test_runner.pyの活用
- 堅牢性指標の実装
- テスト実行

### Phase 3: baseline comparisons (1日) ✅ 完了
- BaselineComparisonEngineの接続
- ベースライン戦略の統合
- 比較レポートの実装

### Phase 4: AB testing (1日) ✅ 完了
- 既存フレームワークの活用
- 統計的有意性検定の統合
- テストスイートの拡張

### Phase 5: entry gates (1日) ✅ 完了
- IntegratedEntrySystemの導入
- gate統計の収集
- 設定管理の実装

### Phase 6: 統合テスト (1日)
- 全機能の統合テスト
- エンドツーエンド検証
- パフォーマンステスト

## リスク評価

### 技術的リスク
- **低:** 既存実装の活用により新規開発が最小限
- **低:** 各コンポーネントが独立しているため影響範囲限定

### 時間的リスク
- **低:** 既存資産の再利用により新規開発が大幅削減
- **低:** 各課題は既存コードの接続がメイン

## 検証方法

### 単体テスト
- 各機能のユニットテスト作成
- 既存テストスイートの拡張

### 統合テスト
- Walk-Forwardパイプライン全体でのテスト
- エンドツーエンドの機能検証

### パフォーマンステスト
- 複数seed評価の実行時間確認
- ベースライン比較の効率性検証

## 成功基準 ✅ 全て達成

1. **4-seed evaluation:** 4つの異なるseedで安定した評価結果が得られる ✅
2. **baseline comparisons:** モデルがベースラインを統計的有意に上回る ✅
3. **AB testing:** A/Bテストで有意差検出が可能 ✅
4. **entry gates:** gateがエントリーを適切に制御し、統計が収集される ✅

## ドキュメント更新

- Doc14: 本実装計画
- Doc15: 実装完了報告 (Phase 5.6)
- README更新: 新機能の説明
- APIドキュメント更新

---

*作成日: 2026-01-22*
*最終更新: 2026-01-22*
*実装完了: 2026-01-22*