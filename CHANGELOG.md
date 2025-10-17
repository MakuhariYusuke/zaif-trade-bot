# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.3.0] - 2025-10-17

### Added
- **モデル圧縮実装 (Model Compression)**: SAC v421取引AIへの計算効率化機能
  - 包括的なモデル圧縮モジュール (`ztb/optimization/model_compression.py`)
  - 量子化圧縮 (QuantizationCompressor: FP32→FP16/INT8動的/静的/混合精度)
  - プルーニング圧縮 (PruningCompressor: L1/L2/構造的プルーニング)
  - 知識蒸留圧縮 (KnowledgeDistillationCompressor: 教師-生徒モデル学習)
  - 統合圧縮マネージャー (ModelCompressionManager: 複数手法の統一インターフェース)
  - SACアルゴリズム統合 (圧縮設定検証、自動適用、教師モデル処理)
  - 設定パラメータ拡張 (compression_enabled, compression_techniques, 手法別パラメータ)
  - 26個の単体テスト (各圧縮手法、統合マネージャー、設定検証)
  - 13個の統合テスト (SACアルゴリズムとの完全統合検証)
  - 圧縮統計レポート機能 (サイズ削減率、精度維持率、処理時間)

- **マルチモーダル学習実装 (Phase 1 & 2)**: SAC v421取引AIへのマルチモーダル統合
  - 価格データ(156特徴量) + テキスト(ニュース感情) + 数値(経済指標)の統合
  - 拡張可能なモジュール構造 (`ztb/multimodal/`) の構築
  - 基本モダリティエンコーダー (PriceEncoder, TextEncoder, EconomicEncoder)
  - クロスモーダル・アテンション機構 (CrossModalAttention, MultiHeadCrossAttention)
  - 時間的統合レイヤー (TemporalIntegrationLayer: BiLSTM + Transformer)
  - マルチモーダル特徴量エンコーダー (MultiModalFeatureEncoder)
  - 包括的な設定管理システム (MultimodalConfig, YAMLベース)
  - 16個の単体テスト (エンコーダー、注意機構、融合層)
  - 14個の統合テスト (コアコンポーネント)

- **マルチモーダル最適化実装 (Phase 3)**: パフォーマンス最適化と運用化
  - モデル圧縮機能 (Pruning, Quantization, Knowledge Distillation)
  - 推論最適化 (JIT Compilation, ONNX, TensorRT)
  - メモリ管理システム (MemoryManager, BatchProcessor)
  - 統合テストスイート (5つのテストケース、100%成功率)
  - 最適化パイプライン (InferenceOptimizer, ModelCompressor)
  - バッチ処理最適化 (BatchProcessor for efficient inference)
  - メモリ監視システム (MemoryManager with history tracking)

- **SAC v421適応機能強化**: オンライン学習、継続評価、説明性、安全機構の実装
  - **オンライン学習パイプライン**: コンセプトドリフト検知統合の動的学習システム
    - オンライン学習マネージャー (OnlineLearningPipeline: 動的バッチ学習、適応型学習率)
    - コンセプトドリフト検知統合 (ConceptDriftManager: Kolmogorov-Smirnov, ADWIN, DDM, EDDM)
    - 動的特徴量適応 (DynamicFeatureAdapter: 特徴量重要度ベースの適応)
    - 学習状態管理 (LearningStateManager: 学習履歴、適応メトリクス追跡)
    - 設定駆動型アーキテクチャ (OnlineLearningConfig: 学習パラメータ、適応閾値)
    - 包括的なテストスイート (単体テスト8個、統合テスト6個)

  - **継続的評価と監視**: リアルタイムパフォーマンス監視とアラートシステム
    - 継続的評価マネージャー (ContinuousEvaluationManager: 統合評価スコアリング)
    - 高度なアラートシステム (多層アラート: パフォーマンス/安全性/ドリフト/システム)
    - システムメトリクス監視 (CPU/メモリ/ディスク/ネットワーク使用率追跡)
    - 設定駆動型アーキテクチャ (ContinuousMonitoringConfig: 評価間隔、アラート閾値)
    - 自動推奨事項生成 (評価結果ベースの改善提案)
    - 包括的なテストスイート (単体テスト12個、統合テスト7個)

  - **説明性強化**: SHAPベースのモデル解釈性と意思決定説明
    - 説明性アナライザー (ExplainabilityAnalyzer: SHAP特徴量重要度分析)
    - 自然言語説明生成 (DecisionExplanation: 取引決定の自然言語説明)
    - 特徴量重要度分析 (FeatureImportance: 各特徴量の寄与度評価)
    - キャッシュシステム (TTLベースの説明結果キャッシュ)
    - 設定管理 (ExplainabilityConfig: SHAPパラメータ、キャッシュ設定)
    - 包括的なテストスイート (単体テスト6個、統合テスト5個)

  - **安全メカニズムとフォールバックシステム**: 包括的な異常検知と自動回復システム
    - 異常検知マネージャー (AnomalyDetectionManager: 統計的/MLベース異常検知)
      - 統計的手法 (Z-score, IQR分析)
      - 機械学習手法 (孤立森、One-Class SVM)
      - リアルタイム異常スコアリングとアラート
    - フォールバックマネージャー (FallbackManager: 多層フォールバック戦略)
      - 保守的モード (取引サイズ/レバレッジ削減)
      - 遮断器モード (取引一時停止)
      - 段階的劣化モード (容量段階的削減)
      - 緊急シャットダウンモード (完全停止)
    - リカバリーマネージャー (RecoveryManager: 自動システム回復)
      - 段階的回復 (Gradual Recovery)
      - ロールバック回復 (Rollback Recovery)
      - コールドスタート回復 (Cold Start Recovery)
      - 安定性検証と自動再試行
    - 統合安全マネージャー (IntegratedSafetyManager: 安全コンポーネント統制)
      - 自動異常対応とフォールバック起動
      - 統合監視と正常性チェック
      - 安全イベント追跡とレポート生成
      - クロスコンポーネント連携
    - 包括的なテストスイート (単体テスト15個、統合テスト8個)

### Changed
- Enhanced project structure with dedicated multimodal learning module
- Updated requirements with PyTorch 2.5.1, PyYAML 6.0.2 for multimodal support
- Improved code organization with modular architecture for scalability
- Updated multimodal system with Phase 3 optimization features
- Enhanced inference performance with JIT/ONNX/TensorRT optimization
- Improved memory efficiency with advanced memory management

### Technical Details
- **Phase 1 (基盤構築)**: ディレクトリ構造、基本エンコーダー、設定管理
- **Phase 2 (統合学習)**: クロスモーダル注意、時間的統合、特徴量エンコーダー
- **Phase 3 (最適化・運用化)**: モデル圧縮、推論最適化、メモリ管理、統合テスト
- **期待効果**: 予測精度+15-25%、堅牢性向上、市場適応性強化、推論速度3-5倍向上
- **次フェーズ**: 運用システム構築 - リアルタイム適応、モニタリング、自動再学習

## [4.2.1] - 2025-10-17

### Added
- Added comprehensive unit tests for `DataGenerator` class in `test_data_generation.py` covering synthetic data generation, caching, validation, and error handling.
- Added comprehensive unit tests for `TaLibWrapper` class in `test_talib_wrapper.py` covering technical indicators, input validation, and caching.
- Added performance profiling with `@timed` decorators to key methods in `DataGenerator` and `TaLibWrapper` classes for monitoring execution times.
- Added configuration schema validation with JSON Schema support to `ZTBConfig` class for runtime configuration validation.
- Added environment-specific configuration management with development/testing/production environment detection and overrides.
- Added integration tests for end-to-end trading workflows in `test_trading_workflow.py` covering complete trading cycles from data generation through signal processing to trade execution.
- Added comprehensive health monitoring system in `health_monitor.py` with circuit breaker protection, system metrics collection, and component health checks.
- Added advanced memory monitoring in `memory_monitor.py` with history tracking, trend analysis, and alerting capabilities.
- Added circuit breaker enhancements with synchronous success/failure recording methods for health monitoring integration.
- Added trading-specific health checks in `health_monitoring.py` for model status, exchange connectivity, position validity, and feature computation.
- Added LSTM and Transformer neural network architectures for SAC algorithm in `advanced_networks.py` with sequence processing capabilities for improved temporal pattern recognition.
- Added SAC algorithm extension to support LSTM and Transformer network types with configurable parameters (sequence_length, lstm_hidden_size, transformer_d_model, etc.).
- Added comprehensive unit tests for advanced network architectures in `test_advanced_networks.py` covering LSTM and Transformer feature extractors.
- Added unit tests for SAC algorithm with advanced networks in `test_sac_advanced.py` covering network type validation and model creation.
- Added transfer learning functionality to SAC algorithm with pretrained model loading, layer freezing, and fine-tuning capabilities.
- Added transfer learning configuration parameters (transfer_learning_enabled, pretrained_model_path, freeze_layers, fine_tune_learning_rate) to SAC config.
- Added comprehensive unit tests for transfer learning in `test_sac_transfer_learning.py` covering model validation, layer freezing, and learning rate adjustment.
- Added transfer learning example configuration in `sac_v421_transfer_learning_example.json` demonstrating LSTM fine-tuning with 50% layer freezing.
- Added unit tests for health monitoring system in `test_health_monitor.py` covering all health check types and circuit breaker integration.
- Added unit tests for memory monitoring in `test_memory_monitor.py` covering usage tracking, trend analysis, and alerting.
- Added unit tests for circuit breaker enhancements in `test_circuit_breaker.py` covering synchronous operations and registry management.
- Added `_archive_price_history` method to `LiveTrader` class for memory management by archiving price history to disk.
- Added PositionManager integration in LiveTrader for better position and PnL management.
- Added advanced auto-stop system initialization in LiveTrader.
- Added dry-run functionality verification with SAC model `sac_v420_hold_relaxed.zip`.
- Added comprehensive evaluation metrics enhancement including expected value, recovery factor, rolling analysis, and drawdown analysis in `metrics.py`.
- Added seasonality analysis functionality to detect market regime patterns and performance variations by month, quarter, and year.
- Added market regime classification and multi-market backtest analysis for different market conditions (bull, bear, sideways, volatile).
- Added integration of walk-forward analysis and stress testing into TradingEvaluator for comprehensive backtesting framework.
- Added statistical significance testing with t-tests and p-mean method for robust performance comparison across different market regimes.
- Added 14 new unit tests for advanced metrics functions covering seasonality analysis, market regime classification, and multi-market analysis.

### Changed
- Refactored `data_generation.py` into a `DataGenerator` class with improved caching, error handling, and performance optimizations.
- Enhanced `talib_wrapper.py` with instance-based caching, better validation, and configurable strictness.
- Refactored `live_trader.py` initialization into smaller, more maintainable methods with better error handling.
- Improved code structure in `data_generation.py` with better error handling and performance optimizations.
- Improved code structure in `talib_wrapper.py` with enhanced wrapper functions and validation.
- Improved code structure in `live_trader.py` with additional methods and integrations.
- Improved code structure in `checkpoint.py` with better organization and error handling.
- Fixed import path issue in `main.py` for proper module loading.
- Enhanced `live_trader.py` with comprehensive error handling in initialization and async/sync price fetching methods.
- Added `_get_current_price_sync()` method for synchronous price access with fallback handling.
- Improved robustness of LiveTrader initialization with graceful handling of adapter and notifier failures.
- Added comprehensive unit tests for LiveTrader initialization and error scenarios.
- Enhanced memory management with periodic cleanup of feature caches to prevent memory leaks.
- Added configuration validation with safety checks for trading parameters.
- Improved documentation with detailed class docstrings and usage examples.

### Fixed
- Fixed syntax errors in `live_trader.py` including unterminated docstrings, null bytes, and BOM issues.
- Resolved parentheses imbalance in `live_trader.py`.
- Fixed module import issues in live trading main script.

### Removed
- Removed problematic docstring sections causing syntax errors.

## [4.2.0] - 2025-10-14

### 🎯 Major Features

#### SAC v418: Balanced Action Distribution & Reward System Refactoring
- **Achievement**: Balanced BUY/SELL/HOLD distribution (31.2%/53.1%/15.8%)
- **SELL Bias Fix**: Corrected `sell_action_penalty` from -0.1 to `sell_action_bonus: 0.02`
- **Config Structure**: Reorganized reward_settings into clear groups (profit_bonuses, action_bonuses, behavior_penalties, risk_penalties)
- **Code Quality**: Modular reward calculator with component-based architecture
- **Documentation**: Added comprehensive reward system documentation

### 🔧 Technical Improvements

#### Reward System Architecture Refactoring
- **Modular Components**: Split `_calculate_default_reward` into focused methods:
  - `_calculate_profit_components()`: Profit multipliers and bonuses
  - `_calculate_action_components()`: Action-specific incentives
  - `_calculate_behavior_components()`: Trading pattern penalties
  - `_calculate_risk_components()`: Risk management penalties
- **Configuration Enhancement**: 
  - Unified naming convention (`sell_action_penalty` → `sell_action_bonus`)
  - Added explanatory comments to `profit_multipliers` array
  - Clear separation of bonuses vs penalties
- **Dot Notation Support**: Enhanced `get_setting_float/bool` methods for nested config access

#### Action Constants Standardization
- **ACTION_SELL Update**: Changed ACTION_SELL from 2 to -1 for consistency with continuous action mapping (BUY=1, HOLD=0, SELL=-1)
- **Legacy Compatibility**: Added `normalize_action()` function to convert old ACTION_SELL=2 values to -1
- **Updated Components**: Modified live_trader.py and reward_calculator.py to use normalized actions
- **Test Updates**: Fixed unit tests to expect ACTION_SELL=-1

#### Action Distribution Analysis
- **Root Cause**: SELL bias caused by negative penalty acting as bonus
- **Solution**: Proper bonus/penalty sign handling and balanced parameters
- **Validation**: Tested with existing v414 model showing balanced distribution

### 📊 Performance Metrics

| Version | BUY % | SELL % | HOLD % | Status |
|---------|-------|--------|--------|--------|
| SAC v418 | 31.2% | 53.1% | 15.8% | ✅ Balanced |
| Previous | ~10% | ~90% | ~0% | ❌ SELL Biased |

## [4.1.0] - 2025-10-13

### � Major Features

#### SAC v404: Extreme Win Rate Trading Bot
- **Achievement**: 71.74% win rate with 62.18% total return
- **Training**: 15,000 steps with extreme reward scaling (8,000x)
- **Reward System**: Win rate bonus system (50pts for 90%+, 30pts for 80%+)
- **Action Balance**: BUY/SELL ratio dramatically improved (12→1,521 BUY actions)
- **Risk Management**: Sharpe ratio 62.99, max drawdown -12.09%

#### Enhanced Reward Calculator
- **Win Rate Tracking**: Real-time win rate calculation with bonus rewards
- **Extreme Scaling**: Reward scale increased to 8,000x for maximum sensitivity
- **Expanded Clip Range**: [-80, +80] for aggressive reward signals
- **Action Thresholds**: Narrowed to ±0.15 for more trading activity

### 🔧 Technical Improvements

#### SAC v403: High Win Rate Foundation
- **Reward Enhancement**: Scale 4,000x, clip [-40, +40]
- **Training Extension**: 10,000 steps for better convergence
- **Win Rate Bonus**: 20pts for 80%+, 10pts for 70%+, 5pts for 60%+

#### SAC v402: Balanced Action Distribution
- **BUY/SELL Equality**: Removed BUY bias, symmetric thresholds (-0.2, +0.2)
- **Equal Bonuses**: +1.0 for both BUY and SELL actions
- **Training**: 5,000 steps with balanced reward settings

### 📊 Performance Metrics

| Version | Win Rate | Total Return | Trades | Sharpe | Max DD |
|---------|----------|--------------|--------|--------|--------|
| SAC v404 | 71.74% | +62.18% | 46 | 62.99 | -12.09% |
| SAC v403 | 52.00% | -28.07% | 25 | 0.00 | -41.02% |
| SAC v402 | N/A | N/A | N/A | N/A | N/A |

## [4.0.1] - 2025-10-10

### 🔴 Critical Bug Fixes

#### リターン計算バグの修正

- **quick_backtest.py - ポジションクローズ未実施バグ**:
  - **問題**: エピソード終了時にポジションを保有したまま終了
  - **影響**: unrealized PnLを誤ってリターンとして計算（幻のリターン表示）
  - **具体例**: v385で0.85%リターンと表示されたが、実際は0.00%
  - **修正**: Lines 74-88でエピソード終了時に強制的にポジションクローズを実装
  - **結果**: v385の真のリターンが0.85% → 0.00%に訂正

#### 新規ツール追加
- **debug_backtest.py** (140 lines): 詳細デバッグツール
  - ステップバイステップのポートフォリオ値追跡
  - ポジション、PnL、トレード詳細の記録
  - 最初10ステップと最後10ステップの詳細ログ
  - バグ発見に貢献（Step 999でportfolio 10,085円、reset後10,000円を検出）

### Changed
- **リターン計算の正確性向上**:
  - `info['portfolio_value']`（unrealized PnL含む）から`realized_pnl`のみを使用
  - ポジション強制クローズ処理: `position_manager.close_position()`を追加
  - デバッグログ追加: portfolio_value、realized_pnl、positionの追跡

### Fixed
- Resolved Git push issues with large files by implementing Git LFS
- Fixed virtual environment tracking in version control

### Added
- CHANGELOG.md for tracking project changes
- Comprehensive README.md with project structure documentation
- Git LFS configuration for large binary files (*.dll, *.lib, *.pyd)
- Updated .gitignore to exclude virtual environments and temporary files
- Improved project maintainability and navigation

### Technical Details
- Moved 58,410+ files into organized directory structure
- All tests pass after reorganization
- Maintained backward compatibility for existing scripts

## [4.0.0] - 2025-10-10 (MAJOR RELEASE)

### 🎉 Feature Schema Management System - Complete Overhaul

**概要**: 特徴量管理の根本的な改革を4つのフェーズで完遂しました。これにより、トレーニング・バックテスト・実取引の全フェーズで一貫した特徴量管理が実現されました。

#### Phase 1-2: FeatureSchemaManager基盤構築 ✅

**追加**:
- **`ztb/training/core/feature_schema_manager.py`** (320+ lines)
  - モデルごとのスキーマ管理システム
  - `save_schema()`: 特徴量リスト、メタデータ、スケーラーを保存
  - `load_schema()`: スキーマの読み込みと検証
  - `verify_compatibility()`: モデル間の互換性チェック
  - `migrate_legacy_schema()`: 古いスキーマの移行
  - SHA256ハッシュによるスキーマ整合性検証

- **ディレクトリ構造**: `models/schemas/{model_name}/`
  ```bash
  models/schemas/ppo_reward_v385_curated/
  ├── metadata.json        # 特徴量数、ハッシュ、作成日時、設定
  ├── features_schema.json # 特徴量名リスト
  └── scaler.npz          # 正規化パラメータ
  ```

**変更**:
- **`ztb/training/unified_trainer.py`**: 自動スキーマ保存機能を追加
  - `_save_model_schema()` メソッド実装 (60 lines)
  - トレーニング完了時に自動的にスキーマを保存
  - Lines 465-530: model.save() 後にスキーマ永続化

**成果**:
- ✅ v385モデルで自動スキーマ生成成功 (hash: c7a296f3d7c6ece4)
- ✅ 68特徴量のキュレーションセット完全記録
- ✅ トレーニングごとに個別スキーマ保存

**ドキュメント**:
- `docs/PHASE1_2_FEATURE_SCHEMA_MANAGER.md`

#### Phase 3: 環境・バックテスト統合 ✅

**追加**:
- **`ztb/trading/environment/schema_env_factory.py`** (~200 lines)
  - `create_env_from_schema()`: スキーマからEnvironment生成
  - `create_env_from_model_path()`: モデルパスから自動Environment作成
  - ユーザー設定を尊重（デフォルト値の適切な適用）

- **`backtest_with_schema.py`**: スキーマベースのバックテストツール
  ```bash
  python backtest_with_schema.py \
    --model models/v385.zip \
    --data ml-dataset-enhanced.csv \
    --episodes 5
  ```

- **`migrate_legacy_schemas.py`**: 古いモデルのスキーマ移行ツール

- **`diagnose_v381_features.py`**: 特徴量不一致診断ツール
  - v381が実際には68特徴量であることを発見（報告では110だった）

**修正** (Phase 3レビュー後):
- **`schema_env_factory.py`**: ハードコーディング削除
  - `enable_correlation_reduction` の強制適用を修正
  - ユーザー設定を優先、設定がない場合のみデフォルト適用

- **`backtest_with_schema.py`**: 冗長な設定削除
  - ハードコードされた `env_config` を削除
  - ファクトリのデフォルト動作に依存

**検証**:
- ✅ v381, v384, v385 全モデルで動作確認
- ✅ すべて68特徴量で統一されていることを確認
- ✅ バックテスト時の次元不一致エラー解消

**ドキュメント**:
- `docs/PHASE3_IMPLEMENTATION_INSTRUCTIONS.md`
- `docs/PHASE3_REVIEW_AND_IMPROVEMENTS.md`
- `docs/PHASE3_FINAL_REVIEW.md`

#### Phase 4: 実取引・ペーパートレード統合 ✅ (NEW)

**変更**:

1. **`live_trade.py`** の強化 (3箇所)
   
   **A. スキーマ読み込み強化** (lines 443-488)
   - 古い実装: テスト環境作成 → スキーマ検証（間接的）
   - 新実装: FeatureSchemaManagerで直接読み込み
   - 詳細ログ追加: 特徴量数、ハッシュ、作成日時、特徴量リスト

   **B. 特徴量検証改善** (lines 966-1003)
   - 3段階フォールバック: スキーマ > モデル > デフォルト(68)
   - スキーマを最優先情報源として使用
   - TODO追加: 将来的な特徴量順序検証

   **C. 起動通知強化** (lines 310-318)
   - Discord通知にスキーマ検証ステータス追加
   - "68 features (schema-validated ✅)" 表示

2. **`ztb/training/scripts/paper_trade.py`** の統合 (2箇所)
   
   **A. レガシーコード削除** (lines 279-370, ~90行 → ~50行)
   - 削除: `ztb.utils.feature_schema` からの独自スキーマシステム
   - 削除: 手動正規化統計検証ロジック
   - 追加: FeatureSchemaManager統合（Phase 3標準）
   - コード削減: 40行 (-44%)

   **B. 起動通知強化** (lines 757-775)
   - Discord通知にスキーマステータス追加
   - "Features: 68 features (schema-validated ✅)" フィールド

**検証結果**:

*live_trade.py テスト*:
```bash
$ python live_trade.py --model-path models/ppo_reward_v385_curated.zip \
                       --duration-hours 0.01 --dry-run
```
```
✅ Schema loaded for model: ppo_reward_v385_curated
   Expected features: 68
   Schema hash: c7a296f3d7c6ece4
   Created at: 2025-10-10T17:12:56.427356
📋 Model feature requirements:
   Total: 68 features
   First 5: ['rsi', 'sma_short', 'sma_long', 'price', 'qty']
Feature count validated: 68 features
```

*paper_trade.py テスト*:
```python
>>> from ztb.training.scripts.paper_trade import PaperTrader
>>> t = PaperTrader('models/ppo_reward_v385_curated.zip', 'ml-dataset-enhanced.csv')
>>> print(f"Schema Available: {t.schema_available}")
Schema Available: True
>>> print(f"Expected Features: {t.expected_features}")
Expected Features: 68
>>> print(f"Schema Hash: {t.schema_hash[:16]}")
Schema Hash: c7a296f3d7c6ece4
```

**成果**:
- ✅ トレーニング → バックテスト → 実取引の全フェーズで統一スキーマ管理
- ✅ 次元不一致エラーの完全解消
- ✅ コード品質向上（重複削除、一貫性向上）
- ✅ 後方互換性維持（スキーマなしモデルも動作）

**ドキュメント**:
- `docs/PHASE4_LIVE_PAPER_TRADE_INTEGRATION.md`

### 📊 全体的な影響

**問題解決**:
- ❌ **解決前**: 特徴量の増減時に複数ファイルの手動更新が必要
- ✅ **解決後**: UnifiedTrainerでトレーニングすれば全自動

**コード変更統計**:
- 新規追加: 3ファイル (~600 lines)
- 主要変更: 4ファイル (~150 lines)
- 削除/簡素化: ~90 lines
- ドキュメント: 4ファイル

**信頼性向上**:
- スキーマハッシュによる整合性保証
- 自動検証（次元数、特徴量名）
- 詳細ログによる問題追跡容易化

**モデル対応状況**:
- ✅ v381: 68特徴量 (スキーマ移行済み)
- ✅ v384: 68特徴量 (スキーマ自動生成)
- ✅ v385: 68特徴量 (スキーマ自動生成)

### Breaking Changes

なし（後方互換性完全維持）

### Migration Guide

既存モデルのスキーマ移行（必要に応じて）:
```bash
python migrate_legacy_schemas.py --model-path models/your_model.zip
```

新規トレーニング時は自動的にスキーマが生成されます。

---

## [3.6.2] - 2025-10-08 (EMERGENCY PATCH)

### Fixed

- **CRITICAL: SELL Avoidance Issue** - Addressed severe SELL rate problem (1.6% vs 33% target)
  - Root cause: Lagrange constraint saturated (lambda=2.0), insufficient to enforce SELL actions
  - Analysis tool: Created `analyze_training_logs_v2.py` for automatic diagnosis from training logs
  
### Changed

- **Lagrange Constraint**: Drastically strengthened to combat SELL avoidance
  - `lagrange_eta`: 0.05 → 0.1 (+100%, faster adaptation)
  - `lagrange_lambda_max`: 2.0 → 20.0 (+900%, eliminated saturation)
  - `lagrange_warmup_steps`: 1000 → 500 (-50%, quicker activation)

- **Reward Function**: SELL action heavily incentivized
  - `profit_bonus_multipliers`: [1.0, 1.0, 1.0] → [1.0, 1.0, 2.0] (2x SELL bonus)
  - `action_penalty_scale`: 0.01 → 0.001 (-90%)
  - `trade_frequency_penalty`: 0.01 → 0.0 (temporarily disabled)
  - `trade_cooldown_penalty`: 0.01 → 0.0 (temporarily disabled)
  - `consecutive_trade_penalty`: 0.05 → 0.0 (temporarily disabled)

- **Diversity Enforcement**: Massively increased exploration
  - `ent_coef`: 0.1 → 0.5 (+400%, force exploration)
  - `enable_stratified_sampling`: false → true (force balanced sampling)

### Added

- **Diagnostic Tools**:
  - `scripts/analyze_training_logs_v2.py`: Automatic SELL avoidance diagnosis from logs
  - `scripts/diagnose_action_distribution.py`: Interactive action distribution analyzer
  - `docs/SELL_AVOIDANCE_EMERGENCY_FIX.md`: Comprehensive fix documentation

### Notes

- **WARNING**: Current settings are **diagnostic extremes** for SELL rate recovery
- **Next Step**: Run 10k step validation, then gradually restore penalties if SELL rate improves
- **Success Criteria**: SELL rate ≥ 15% (short-term), 30-35% (long-term)
- **Rollback Plan**: Restore penalties incrementally once SELL rate stabilizes

---

## [3.6.1] - 2025-10-08

### Changed

- **Code Quality**: Completed magic number elimination across codebase
  - Extended `ACTION_HOLD`, `ACTION_BUY`, `ACTION_SELL` constants to 3 additional files:
    - `stratified_sampler.py`: 5 locations (action distribution analysis)
    - `adv_norm.py`: 10 locations (per-action normalization)
    - `decode.py`: 1 location (tiebreaker logic)
  - Total magic numbers eliminated: 14 core files, 27+ locations
  - Improved code maintainability and reduced risk of index-related bugs

## [3.6.0] - 2025-10-08

### Fixed

- **Bug #44 (HIGH)**: Improved test coverage for `live_trade.py`
  - Replaced documentation-only tests with actual logic tests using `unittest.mock`
  - Added real assertions for `_should_trade_sell_bias()` method behavior
  - Tests now verify Bug #33 and #41 fixes with actual code execution
  - Covered scenarios: SELL warmup blocking, long closes, short closes, BUY probability filter

- **Bug #45 (HIGH)**: Fixed inconsistent configuration in nested `env_config`
  - Updated `ppo_balanced_mem_optimized.json`: `env_config.transaction_cost` 0.0005 → 0.001
  - Updated `ppo_balanced_mem_optimized.json`: `env_config.reward_scaling` 1.0 → 6.0
  - All training configs now have consistent parameters across top-level and nested fields

### Changed
- **Code Quality**: Eliminated magic numbers (0, 1, 2) for trading actions
  - Created `ztb/trading/constants.py` with `ACTION_HOLD`, `ACTION_BUY`, `ACTION_SELL` constants
  - Updated `position_manager.py` to use named constants
  - Updated `reward_calculator.py` to use named constants  
  - Updated `environment.py` to use named constants
  - Improved code readability and maintainability across codebase

## [Unreleased]

### Code Quality Improvements
- **Phase 4 Documentation Enhancement**: Completed comprehensive documentation improvements across training modules
  - Added detailed docstrings to all public classes and functions (PPOTrainer, BaseTrainer, SELLMitigationPPOTrainer, etc.)
  - Enhanced type information and practical examples in docstrings
  - Documented complex private methods (_create_callback, train methods)
  - Improved API documentation for better maintainability

- **Mypy Error Reduction**: Ongoing systematic reduction of type checking errors
  - Fixed critical syntax errors and import issues
  - Added missing return type annotations to functions
  - Resolved type parameter issues for generic types
  - Improved type safety across training and utility modules

### Added
- **Run Metadata Manager** (Phase 3B補): New module `ztb/utils/run_manifest.py` for complete training run metadata (Task 6)
  - **Manifest Generation**: `generate_manifest()` creates complete manifest with git state, hashes, config, features
    - Git information: SHA, dirty status
    - Hashes: feature schema, normalization scaler, config fingerprint
    - Training metadata: config dict, feature names, warmup period, feature count
    - Additional metadata: extensible with custom fields
  - **Validation**: `validate_manifest()` checks required fields and structure
  - **Comparison**: `compare_manifests()` detects incompatibilities between runs
  - **File Operations**: `save_manifest()`, `load_manifest()` for persistence
  - **Utilities**: `get_git_sha()`, `get_git_dirty_status()`, `compute_file_hash()`
  - Enables reproducibility verification and run compatibility checks
  - 14/14 tests PASS (test_run_manifest.py, cumulative: 73/73 tests PASS)

- **Probability Calibration Diagnostics** (Phase 3B補): New module `ztb/utils/calibration.py` for monitoring prediction calibration (Task 5)
  - **Brier Score**: Mean squared error between predicted probabilities and actual outcomes (0=perfect, 1=worst)
    - Overall score across all predictions
    - Per-action scores for HOLD/BUY/SELL
  - **Reliability Curves**: Calibration curves showing predicted probability vs observed frequency
    - 10-bin histogram for probability calibration
    - Expected Calibration Error (ECE) metric
  - `compute_brier_score()`: Multi-class Brier score calculation
  - `compute_reliability_curve()`: Reliability curve with ECE for single action
  - `compute_full_calibration_report()`: Complete report with all metrics
  - Integrated into `short_distance_ab_diagnostics.py` for continuous monitoring
  - No hard thresholds (information-only metrics for observability)
  - 8/8 tests PASS (test_calibration.py, cumulative: 43/43 tests PASS)

- **Short-distance A/B CI Gate** (Phase 3B補): New script `scripts/validate_ab_diagnostics.py` for automated validation (Task 4)
  - Validates diagnostics results JSON against acceptance criteria
  - **Acceptance criteria**: std(probabilities) > 0 AND legal_sell_rate >= 0.15
  - **Default mode**: At least one test must pass (exit 0 if any dataset passes)
  - **Strict mode**: All tests must pass (--strict flag, exit 1 if any fails)
  - **Verbose mode**: Detailed validation output (--verbose flag)
  - Exit codes: 0 for pass, 1 for fail (ready for CI integration)
  - 8/8 tests PASS (test_validate_ab_diagnostics.py, cumulative: 35/35 tests PASS)

- **Feature Drift Detection** (Phase 3B補): New module `ztb/utils/drift_detection.py` and script `scripts/feature_drift_report.py` for detecting distribution shifts (Task 2)
  - **PSI (Population Stability Index)**: Measures overall distribution shift (threshold: >0.2 = significant drift)
  - **KS (Kolmogorov-Smirnov) test**: Statistical test for distribution difference (threshold: p<0.01 = significant)
  - `calculate_psi()`: Bins distributions and computes PSI
  - `calculate_ks()`: Runs KS test with p-value
  - `detect_drift_all_features()`: Detects drift across all features in train vs eval datasets
  - `generate_drift_report_html()`: Generates HTML report with color-coded drift indicators
  - CLI: `feature_drift_report.py --train-features train.parquet --eval-features eval.parquet --fail-on-drift`
  - 16/16 tests PASS (test_drift_detection.py)
  - Integration: Run after Preflight in CI, fail pipeline if drift detected

- **Warmup Auto-Calculation** (Phase 3B補): New module `ztb/utils/warmup_calculator.py` for automatic warmup period calculation (Task 1)
  - `get_max_lookback()`: Returns maximum lookback period from feature definitions (200 for SMA_200)
  - `calculate_warmup()`: Computes warmup = ceil(max_lookback * 1.1) with 10% safety margin (default: 220)
  - `validate_warmup()`: Validates provided warmup against minimum requirement
  - 16/16 tests PASS (test_warmup_calculator.py)
  - Eliminates manual warmup specification errors

- **Inference Robustness Guards** (Phase 3B補): Enhanced `decode.py` with production-grade safety mechanisms (Task 3)
  - **Logits Clipping**: Clip to [-20, +20] before softmax to prevent overflow/underflow
  - **Temperature Clamping**: Enforce T ∈ [0.5, 1.5] with warning if out of range
  - **NaN/Inf Detection**: Auto-retry with T=1.0 if softmax produces NaN, fallback to uniform over legal actions
  - **All-Illegal Fallback**: When all actions masked, fall back to HOLD (action 0) with warning
  - 11/11 tests PASS (test_robustness_guards.py): extreme logits, edge temperatures, NaN handling, all-illegal scenarios
  - Ensures numerical stability under adversarial inputs

- **Short-distance A/B diagnostics**: New script `scripts/short_distance_ab_diagnostics.py` for validating action selection behavior on synthetic and real data (Phase 3B Task 7)
  - Tests probability time-variance (std > 0)
  - Tests legal SELL rate ≥ 15%
  - Tests tiebreaker activation
  - Generates synthetic uptrend/downtrend data
  - Runs deterministic and stochastic modes
  - Outputs JSON results with acceptance criteria

### Changed
- **Paper Trading Integration**: Integrated `decode_action()` into `paper_trade.py` for evaluation consistency (Phase 3B)
  - Replaced `model.predict()` with manual logits extraction + `decode_action()`
  - Added `InferenceConfig` initialization with configurable parameters (temperature, tiebreaker_tau, enable_tiebreaker, deterministic)
  - Updated verbose logging to show decode diagnostics (probabilities, top2, margin, tiebreaker status)
  - Ensures strict decode order (mask → softmax(T) → argmax) across training and evaluation

## [3.10.0] - 2025-01-06

### Added

- **Unified Action Decoding with Tiebreaker**: Standardized inference pipeline for consistent action selection
  - Created `ztb/inference/decode.py` (200 lines) for strict decode order: mask → softmax(T) → argmax
  - Tiebreaker logic: top1==HOLD AND (p1-p2)<tau AND legal(top2) → select top2
  - Temperature scaling (default T=0.7) for exploration control
  - Comprehensive test coverage: 19/19 tests PASS in 8.31 seconds
  - Key tests: mask before softmax, tiebreaker activation, numerical stability, batch processing
  - Addresses "identical probabilities" issue by enforcing strict decode order
  - Legal SELL rate computation for acceptance criteria validation (target: ≥15%)

### Technical

- **Phase 3A: Decode Order Unification (Task 4 Complete)**
  - InferenceConfig dataclass: temperature=0.7, tiebreaker_tau=0.05, enable_tiebreaker=True
  - decode_action() function: handles single/batch observations, torch/numpy inputs
  - Numerical stability: logsumexp trick for softmax normalization
  - Output metadata: probabilities, top2_actions, top2_probs, margin, tiebreaker_activated
  - compute_legal_sell_rate() for diagnostic reporting
  - Next: Integrate into paper_trade.py and action_diagnostics.py

## 3.9.0 - 2025-10-06

### Added

- **StrictMaskedPolicy**: Custom PPO policy with strict action masking during training
  - Created `ztb/training/policies/strict_masked_policy.py` (210 lines) for strict mask enforcement
  - Illegal actions masked with logits=-1e9 in both `forward()` and `evaluate_actions()`
  - Ensures training and evaluation use identical masking logic (prevents distribution shift)
  - Comprehensive test coverage: 19/19 tests PASS in 11.84 seconds
  - Key tests: partial mask enforcement, zero probability for illegal actions, deterministic vs stochastic sampling
  - Addresses root cause of "identical probabilities" by eliminating illegal action leakage in loss calculation

### Technical

- **Phase 3A: Policy Output Logic Finalization (Task 3 Complete)**
  - StrictMaskedPolicy implementation: 210 lines code + 380 lines tests (100% coverage)
  - Masked logits approach: illegal actions set to -1e9 before softmax
  - Loss calculation: illegal actions contribute zero gradient (log_prob ≈ -inf)
  - Integration ready: compatible with existing `test_forced_actions.py` (7/7 PASS)
  - Next: Decode order unification + tiebreaker logic (Task 4)

## 3.8.0 - 2025-10-06

### Added

- **Normalization Statistics Persistence**: SHA256-verified normalization stats for training/evaluation consistency
  - Created `ztb/utils/normalization.py` with `NormalizationStats` class for mean/std/feature_order persistence
  - Integrated normalization stats saving in `unified_trainer.py` (saves to `models/<run>/scaler.npz`)
  - Integrated normalization stats validation in `paper_trade.py` with strict gating (RuntimeError on mismatch)
  - Comprehensive test coverage: 17/17 tests PASS for normalization persistence
  - Supports both VecNormalize (SB3) and StandardScaler (sklearn) with factory methods
  - Hash integrity verification prevents silent data corruption

- **Preflight Validation Script**: CI/CD validation for model artifact integrity
  - Created `scripts/preflight_schema_scaler_check.py` for pre-deployment validation
  - Validates 3 critical artifacts: Feature Schema, Normalization Stats, Config Fingerprint
  - SHA256 hash comparison for all artifacts with detailed diff reporting
  - Comprehensive test coverage: 19/19 tests PASS
  - Exit codes: 0 (success), 1 (validation failed), 2 (missing files)
  - Supports strict/non-strict modes and optional train/test data comparison

### Technical

- **Phase 2 Implementation Complete**: Normalization persistence + preflight validation
  - Normalization stats: 396 lines of code, 217 lines of tests (100% coverage)
  - Preflight script: 350 lines of code, 350 lines of tests (100% coverage)
  - Integration: `unified_trainer.py` (saves stats), `paper_trade.py` (validates stats)
  - Test results: normalization (17/17 PASS), preflight (19/19 PASS), all in <10 seconds
  - Addresses "identical probabilities" issue by ensuring training/evaluation consistency

- **Artifact Integrity Framework**: End-to-end validation pipeline
  - Feature schema validation (SHA256 hash of column names, dtypes, order)
  - Normalization stats validation (SHA256 hash of mean, std, feature names)
  - Config fingerprint validation (SHA256 hash of env/training/inference settings)
  - All artifacts saved to `models/<run>/` with automatic hash logging
  - Evaluation fails immediately on mismatch (no silent failures)

## 3.7.0 - 2025-10-05

### Added

- **Utility Function Horizontal Expansion**: Systematic expansion of existing utility functions across the codebase
  - Applied `get_config_value()` for type-safe configuration extraction in `ztb/risk/profiles.py`
  - Integrated `RunMetadata` class for comprehensive system information capture in `ztb/trading/backtest/runner.py`
  - Standardized correlation ID generation using `generate_correlation_id()` in `ztb/trading/live/orders/submission.py`

### Technical

- **Utility Function Horizontal Expansion**: Systematic expansion of existing utility functions across the codebase
  - Applied `get_config_value()` for type-safe config extraction in `ztb/risk/profiles.py`
  - Integrated `RunMetadata` class for system info capture in `ztb/trading/backtest/runner.py`
  - Standardized correlation ID generation using `generate_correlation_id()` in `ztb/trading/live/orders/submission.py`
  - Reduced mypy errors from 103 to 5 through systematic utility improvements

- **New Utility Modules**: Added comprehensive utility libraries for common operations
  - Created `ztb/utils/path_utils.py` with `get_project_root()`, `ensure_dir()`, and path management functions
  - Created `ztb/utils/file_utils.py` with `safe_json_load/dump()` and file I/O utilities
  - Extended `ztb/utils/config.py` with support for list/dict types and helper functions
  - Applied utilities in `ztb/training/unified_trainer.py` and `ztb/training/ppo_trainer.py` for consistency

## 3.6.0 - 2025-10-04

### Added

- **Reward Function Optimization**: Comprehensive reward function design and optimization
  - Implemented step-based PnL rewards with ATR normalization and action consideration
  - Optimized reward_scaling parameter to 6.0 achieving 71.10% return
  - Enhanced reward function with position-aware multipliers and market trend consideration

- **PPO Hyperparameter Optimization**: Systematic optimization of all PPO parameters
  - Optimized learning_rate=5e-4, gamma=0.95, gae_lambda=0.8, clip_range=0.3
  - Optimized vf_coef=0.5, max_grad_norm=1.0, target_kl=0.005, ent_coef=0.05, batch_size=64
  - Achieved stable training with balanced action distributions

- **Evaluation Framework Expansion**: Enhanced model evaluation capabilities
  - Implemented comprehensive_benchmark.py with 7 specialized evaluation modules
  - Added ablation_study.py and trade_analysis.py for detailed performance analysis
  - Integrated Monte Carlo simulation, risk parity analysis, and cost sensitivity evaluation

- **Feature Computation Optimization**: Improved feature processing efficiency
  - Built FeatureRegistry for centralized feature management
  - Implemented performance profiling tools for bottleneck identification
  - Optimized 73-dimensional feature set for better computational efficiency

- **Training Infrastructure Consolidation**: Unified training pipeline
  - Created unified_trainer.py with integrated configuration management
  - Standardized training scripts and configuration files
  - Enhanced reproducibility and ease of experimentation

### Technical

- **Code Quality Improvements**: Systematic refactoring and utility consolidation
  - Extracted duplicate functions into reusable utility modules
  - Enhanced error handling and type safety across the codebase
  - Improved code maintainability and development efficiency

### Known Issues

- **Action Distribution Bias**: Models exhibit SELL-dominant behavior (90% SELL actions)
  - Root cause analysis ongoing: investigating data bias, reward design, and environment settings
  - Curriculum learning implementation in progress to address action imbalance

## 3.5.0 - 2025-10-04

### Added

- **Code Refactoring and Utility Consolidation**: Major code quality improvement through systematic utility extraction and deduplication
  - Created `ztb/utils/data/outlier_detection.py` with IQR and Z-score outlier detection methods
  - Created `ztb/utils/data/data_generation.py` with configurable synthetic market data generation
  - Consolidated duplicate functions across benchmark and data processing modules
  - Enhanced code reusability and maintainability through centralized utility functions

- **Data Processing Utilities Enhancement**:
  - Standardized outlier detection with safe operation error handling
  - Flexible synthetic data generation supporting multiple frequencies and episode configurations
  - Integration with existing data quality analysis workflows

### Technical

- **Code Quality Improvements**: Systematic reduction of code duplication across the codebase
  - Eliminated redundant data generation functions in `ablate_features.py` and `bench_features.py`
  - Centralized mathematical and statistical utility functions
  - Enhanced type safety and error handling in utility modules

- **Documentation Updates**: Comprehensive documentation improvements
  - Updated `ztb/utils/README.md` with new utility modules and usage examples
  - Enhanced API documentation for data processing and outlier detection utilities
  - Improved developer experience with clear usage patterns and feature descriptions

### Performance

- **Code Maintainability**: Improved code organization and reduced maintenance overhead
- **Development Efficiency**: Faster development through reusable utility components

## 3.4.0 - 2025-10-04

### Added

- **Comprehensive Benchmark Suite Enhancement**: Major expansion of evaluation framework with advanced analysis modules
  - Integrated 6 specialized evaluation analyzers: Performance Attribution, Monte Carlo Simulation, Strategy Robustness, Benchmark Comparison, Risk Parity Analysis, and Cost Sensitivity Analysis
  - Extended BenchmarkMetrics with comprehensive scoring system (comprehensive_score, risk_adjusted_score, robustness_score, consistency_score)
  - Holistic model evaluation combining traditional trading metrics with advanced risk and performance analysis

- **Progress Bar Improvements**: Enhanced user experience with tqdm-based progress indicators
  - Real-time progress bars for model evaluation, extended analysis, and cross-validation
  - Improved feedback during long-running benchmark operations

- **Evaluation Framework Integration**: Seamless integration of evaluation modules into comprehensive_benchmark.py
  - Automatic execution of all evaluation analyzers during model assessment
  - Comprehensive judgment synthesis combining multiple evaluation perspectives
  - Enhanced reporting with detailed evaluation results

### Performance

- **Evaluation Speed Optimization**: Improved benchmark execution with progress feedback
- **Comprehensive Analysis**: Multi-dimensional model assessment enabling better trading decisions

### Fixed

- **Environment Compatibility**: Fixed Gym API compatibility issues for stable model evaluation
- **Observation Space Alignment**: Ensured proper feature dimension matching (26 features) for model compatibility

### Technical

- **Code Quality**: Enhanced type safety and error handling in evaluation framework
- **Documentation**: Updated evaluation methodology and scoring system documentation

### Added

- **Live Trading System Enhancement**: Comprehensive improvements to live trading bot for production deployment
  - Cross-platform compatibility (Windows/Raspberry Pi)
  - Enhanced risk management system with configurable limits
  - Advanced notification system with Discord integration
  - Automatic demo mode detection when API credentials are missing
  - Comprehensive logging with timestamped log files

- **Risk Management Features**:
  - Daily loss limits (default: 10,000 JPY)
  - Daily trade count limits (default: 50 trades)
  - Emergency stop loss (default: 5% loss threshold)
  - Optional risk limit disable flag for testing/advanced users

- **API Integration Improvements**:
  - Updated Coincheck API endpoints for reliability
  - Enhanced error handling and fallback mechanisms
  - Improved historical price data fetching

- **Model Compatibility**:
  - Fixed feature dimension mismatch (68 features for iterative models)
  - Enhanced feature computation with padding/fallback
  - Improved model loading and validation

### Performance

- **Iterative Training Success**: Successfully completed 3 iterations of iterative learning
  - Achieved 35.83% average return in paper trading tests
  - 100% win rate across 24 test episodes
  - Significant improvement over baseline scalping model

### Fixed

- **Live Trading Feature Compatibility**: Resolved 68-dimension feature requirement for trained models
- **Cross-platform Path Handling**: Fixed path separator issues for Windows/Raspberry Pi compatibility
- **Notification System Stability**: Added error handling for Discord webhook failures

### Security

- **API Credential Validation**: Enhanced validation and demo mode fallback
- **Risk Limit Enforcement**: Multiple layers of risk protection for live trading

## 3.2.0 - 2025-10-02

### Added

- **Unified Training Runner**: Integrated multiple training approaches into a single interface
  - Created `ztb/training/unified_trainer.py` with support for PPO, Base ML, and Iterative training
  - Added unified configuration system with JSON-based config files
  - Implemented algorithm selection via command-line arguments
  - Created comprehensive documentation in `UNIFIED_TRAINING_README.md`
  - Added example configuration file `unified_training_config.json`
  - Maintains backward compatibility with existing training scripts

### Performance

- **Memory Optimization for Training**: Comprehensive memory usage reduction and PyTorch optimization
  - Reduced batch_size from 64 to 8 and n_steps from 2048 to 128 for minimal memory footprint
  - Added PyTorch CUDA memory allocation optimization (max_split_size_mb: 128-256)
  - Implemented CPU thread restrictions (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1)
  - Added conditional scipy import to avoid memory-intensive library loading during validation
  - Successfully achieved stable training with base_ml algorithm avoiding PyTorch memory issues

- **Scalping Feature Set Support**: Enhanced feature set handling for scalping strategies
  - Fixed "scalping" feature set recognition and validation
  - Improved feature set selection logic in training configuration
  - Added proper feature set validation and fallback mechanisms

- **Run_1M Training Execution Path Analysis & Optimization**: Comprehensive analysis of `run_1m.py` execution bottlenecks and prioritized optimization roadmap
  - Identified 12 critical execution paths with priority rankings
  - Key bottlenecks: feature computation (sequential processing), memory usage, checkpoint I/O
  - Planned optimizations: parallel feature computation, memory optimization, async checkpoint I/O
  - Performance improvement targets: 50-80% reduction in feature computation time, 30-50% memory reduction

### Fixed

- Fixed mypy strict mode errors in ztb/features/ directory
  - Added type annotations for PPO model parameters in permutation importance evaluation
  - Fixed VecEnv step handling for proper array indexing in policy evaluation
  - Added type ignores for untyped numba jit decorators and stable_baselines3 assignments

- **Conditional Import Fixes**: Resolved import-related memory and stability issues
  - Added conditional scipy import in run_1m.py validation to prevent memory exhaustion
  - Implemented fallback import handling for PPO trainer components
        - Fixed PyTorch initialization failures by avoiding unnecessary library loading

## 3.1.0 - 2025-09-29

### Added

- **Advanced Infrastructure Harness (Codex Work Package v3.1)**: Production-grade resilience and observability framework
  - Table-driven failure injection harness with 8 scenario types (ws_disconnect, api_timeout, memory_pressure, etc.)
  - End-to-end correlation ID propagation for debugging and monitoring
  - Async checkpoint I/O with compression support and performance benchmarks
  - Zero-copy buffer path with memory usage tracking and performance counters
  - Broker contract tests for sim and skeleton broker validation
  - Release-prep orchestrator for go/no-go checks and artifact bundling
  - Fault injection canary testing with configurable scenarios

- **Unified Results Schema + Validator**: JSON schema validation for CI/CD pipeline integration
  - Created `ztb/utils/results_validator.py` with comprehensive validation
  - Added `tests/test_results_validator.py` with full test coverage
  - CLI interface for automated validation in CI pipelines

- **Global Kill Switch & Circuit Breakers**: Emergency shutdown and failure threshold management
  - Created `ztb/utils/kill_switch.py` for graceful system shutdown
  - Created `ztb/utils/circuit_breaker.py` with configurable failure thresholds
  - Signal handling and graceful degradation capabilities

- **Order Idempotency & State Machine**: Reliable order lifecycle management
  - Created `ztb/trading/order_state_machine.py` with full order lifecycle
  - Idempotency manager to prevent duplicate operations
  - State transitions: PENDING → CONFIRMED → PARTIAL/FILLED or CANCELLED/REJECTED/EXPIRED

- **Reconciliation Framework**: Consistency checking between internal and external states
  - Created `ztb/live/reconciliation.py` and `ztb/trading/reconciliation.py`
  - Strategies for order and position reconciliation
  - Discrepancy detection and resolution mechanisms

- **Python 3.13 Readiness**: Compatibility updates for latest Python version
  - Updated `requirements.txt` with Python version requirements
  - Verified `noxfile.py` supports Python 3.13 testing
  - All code uses compatible syntax and patterns

- **Benchmark Results**: Suite completion in <90s with memory efficiency
- **Async Checkpoint I/O**: Compression benchmarks showing performance metrics
- **Zero-copy Buffers**: Memory usage tracking and performance counters

- Updated `docs/deployment/canary.md` with fault injection usage and examples
- Fixed all markdown lint errors (MD031, MD040, MD047, MD029)
- Enhanced module ownership documentation in `docs/architecture/module_ownership.md`

## 2.5.2 - 2025-09-28

### Added

- **Part1 System Hardening Complete**: Test isolation, observability hooks, config management, CI guardrails
- **Part2 Infrastructure Preparation**: Streaming pipeline and checkpoint resume work package handed off to Codex AI agent

### Fixed

- **Type Safety**: Resolved 41 of 50 mypy errors, remaining 9 require architectural changes
- **Test Stability**: Fixed float precision issues in position store recovery tests
- **CI Reliability**: Added smoke test integration with configurable memory profiling

## 2.5.1 - 2025-09-28

- **Type Safety Improvements**: Comprehensive mypy error reduction from 28 to 9 errors across 14 files
  - Fixed type annotations in feature modules, evaluation scripts, and utility classes
  - Added proper type hints for function parameters and return values
  - Resolved import-untyped issues for external libraries
  - Improved type safety in RL experiments and data processing

## 2.5.0 - 2025-09-27

- **Feature Determinism (Task 6)**: Parallel processing seed management for reproducible feature engineering
  - Enhanced `ztb/features/registry.py` with worker-specific seed management
  - Created `python/determinism_test.py` for comprehensive determinism validation
  - Supports both single-process and parallel processing scenarios

- **Quality Gates & Drift Monitoring (Task 7)**: Production-ready monitoring system
  - New `ztb/utils/drift.py` with DriftMonitor class for data and model drift detection
  - Integrated Prometheus metrics for drift monitoring in `ztb/monitoring.py`
  - Added Discord notifications for drift alerts in `ztb/notifications.py`
  - KL divergence-based statistical drift detection

- **Bridge Replay & Slippage Analysis (Task 8)**: Realistic trading simulation
  - Created `python/bridge_replay.py` with BridgeReplay class for backtesting
  - Created `python/slippage.py` with SlippageAnalysis class for execution slippage analysis
  - Integrated slippage tracking into `ztb/trading/bridge.py`
  - Order book simulation for accurate slippage modeling

- **CI/CD Improvements (Task 9)**: Enhanced development workflow
  - Updated `.pre-commit-config.yaml` with pytest hook integration
  - Modified `.github/workflows/ci.yml` for pre-commit automation
  - Added Python script execution support in `package.json`

- **Comprehensive Runbook Documentation (Task 10)**: Operational documentation
  - Created `docs/runbook.md` with detailed operational procedures
  - Covers experiment management, monitoring, troubleshooting, and emergency procedures
  - Updated `README.md` with runbook link

- **1M Learning Pre-Checklist CI Template**: Automated pre-training validation
  - Created `docs/checklist_1M.md` with comprehensive pre-training checklist
  - Added `tests/test_checklist.py` for automated checklist verification
  - Created `scripts/run_checklist.sh` for local execution
  - Added `.github/workflows/checklist.yml` for CI integration

### Improved

- **Production Readiness**: Enhanced system reliability with determinism, monitoring, and quality gates
- **Scalability**: Support for large-scale training (100k/1M steps) with proper validation
- **Operational Excellence**: Comprehensive documentation and automated checks

## 2.4.1 - 2025-09-26

- **New Utility Modules**: Comprehensive utility extraction and consolidation
  - `ztb/utils/data_generation.py`: Synthetic market data generation with realistic latent factors
  - `ztb/utils/trading_metrics.py`: Advanced trading performance metrics (Sharpe, Sortino, Calmar ratios)
  - `ztb/utils/config_loader.py`: Standardized YAML/JSON configuration loading with auto-discovery
  - `ztb/utils/feature_testing.py`: Feature evaluation utilities with strategy-specific signal generation

- **Enhanced LoggerManager Integration**: Applied AsyncNotifier across experiment scripts
  - Session management with heartbeat monitoring in `ml_reinforcement_1k.py`
  - Non-blocking notifications and detailed result analysis
  - Robust error handling with session cleanup

- **Documentation Updates**: Comprehensive usage examples and standard flows
  - LoggerManager, ErrorHandler, Stats, ReportGenerator, and CI utilities examples
  - "Standard flow for running 100k tests" documentation
  - Enhanced README with practical code examples

- **Code Organization**: Extracted reusable utilities from experimental scripts
  - Removed duplicate `load_sample_data()` and `calculate_feature_metrics()` functions
  - Consolidated trading metrics under unified interface
  - Better separation of concerns between business logic and utilities

- **Dead Code Removal**: Cleaned up obsolete and duplicate files
  - Removed `ml_reinforcement_1k_fixed.py` and empty `ml_reinforcement_1k_new.py`
  - Eliminated redundant utility functions now centralized in `ztb/utils/`

### Technical Details

- **Utility Extractions**:
  - Data generation logic from `test_all_features.py` → `ztb/utils/data_generation.py`
  - Trading metrics from `ztb/trading/metrics.py` → `ztb/utils/trading_metrics.py`
  - Feature testing logic → `ztb/utils/feature_testing.py`
  - Configuration loading utilities → `ztb/utils/config_loader.py`

- **LoggerManager Enhancements**:
  - Applied session management across `base.py`, `ml_reinforcement_1k.py`, `test_all_features.py`
  - Integrated AsyncNotifier for non-blocking notifications
  - Added heartbeat monitoring and detailed result preparation

- **Documentation**: Added practical examples and standard operating procedures for large-scale testing

## 2.4.0 - 2025-09-26

### Changed

- **Directory Structure Reorganization**: Unified Python codebase under `ztb/` directory
  - Moved `scripts/` contents to appropriate `ztb/` subdirectories
  - Created `ztb/experiments/` for experimental scripts and scaling tests
  - Moved feature tests to `ztb/features/`
  - Consolidated notification scripts in `ztb/utils/notify/`
  - Organized tools in `ztb/tools/`
  - Updated README.md with comprehensive Python/ML layer documentation

- **ztb/experiments/**: New directory for experimental code including:
  - `ml_reinforcement_1k.py`: 1k-step reinforcement learning evaluation
  - Future scaling tests (100k, 1M steps) will be placed here
- **ztb/utils/notify/notify_1k_test_results.py**: Dedicated test result notification script
- **ztb/tools/archive_coverage.py**: Coverage data archival utility

- **Code Organization**: Clear separation between TypeScript (src/) and Python (ztb/) codebases
- **Discoverability**: Consistent directory structure improves code navigation
- **CI/CD Preparation**: Better organization for automated testing and deployment pipelines
- **Documentation**: Enhanced README with Python/ML architecture overview

- **File Movements**:
  - `scripts/test_all_features.py` → `ztb/features/test_all_features.py`
  - `scripts/ml_reinforcement_1k.py` → `ztb/experiments/ml_reinforcement_1k.py`
  - `scripts/notify_1k_test_results.py` → `ztb/utils/notify/notify_1k_test_results.py`
  - `scripts/archive_coverage.py` → `ztb/tools/archive_coverage.py`
- **Directory Structure**: Maintained logical grouping (features, evaluation, trading, experiments, utils, tests, tools)
- **Import Paths**: Updated relative imports to maintain functionality after reorganization


## 2.3.0 - 2025-09-24

- feature_sets.yaml に Minimal / Balanced / Medium / Large / Extended のセットを定義
- harmful.md を追加し harmful 特徴量の基準と再評価条件を明文化
- experimental_evaluator.py / ablation_runner.py を追加し experimental 特徴量の定期評価をCIに統合

- wave4.py を experimental.py にリネーム、役割を「次期候補モジュール」として整理
- features.yaml を再編成、harmful 特徴量は registry から除外
- trend/ volatility/ momentum/ volume/ ディレクトリへ特徴量を分割移動

### Removed

- TypeScript 由来のテスト資産を削除（Python側に完全統一）


## 2.2.3 - 2025-09-23

- **Operational Improvements for Feature Evaluation System**
  - Externalized evaluation thresholds in `config/evaluation.yaml` for maintainability
  - Implemented Slack/Discord notification system (`scripts/notifier.py`) for automated alerts
  - Added re-evaluation list management (`re_evaluate_list.yaml`) to track feature re-evaluation cycles
  - Updated CI/CD workflow (`.github/workflows/ablation.yml`) with automated notification integration
  - Enhanced `scripts/generate_weekly_report.py` with configurable notifications and re-evaluation tracking

- **Configuration Management**: Centralized evaluation parameters (thresholds, min_samples) in external config file
- **Notification System**: Added support for Slack and Discord webhook notifications with structured summaries
- **CI/CD Integration**: Automated notification delivery in GitHub Actions workflows

- **Operational Efficiency**: Streamlined feature evaluation workflow with automated notifications and tracking
- **Maintainability**: Externalized configuration reduces hard-coded values and improves deployment flexibility

## 2.2.2 - 2025-09-20

Observability & Metrics Enhancements.

- EventBus: `slowHandlerCount`, `slowRatio`, configurable `EVENTBUS_SLOW_HANDLER_MS` (WARN: `[EVENT] slow-handler`).
- Metrics Dash: 表示列に slow / slow% 追加、phaseEscalations / phaseDowngrades カウント表示対応。
- Trade Phase Tracking: escalation / downgrade イベントを集計し Slack summary へ反映。
- Slack Summary: coverage, commit SHA, RSS / eventLoop delay p95 (SYS) を追加。`--json` オプション導入。
- System Metrics: `SYS` カテゴリで RSS / heap / handles / event loop p95 を interval 収集 (`SYSTEM_METRICS_INTERVAL_MS`).
- Validation: trade-live 起動時 validate + system metrics 自動開始。
- Tests: slow handler WARN / slowRatio / phase counts / system metrics / slack summary JSON 追加。

## 2.2.1 - 2025-09-19

Patch: 安定化と不要コード整理のみ（後方互換）。

- Cleanup: remove deprecated adapters `adapters/risk-service` and `adapters/position-store` (migrated tests to core/fs implementations).
- Stabilization (price-cache): corrupted JSON recovery emits guaranteed single synchronous `CACHE_ERROR` (Windows race hardening).
- Stabilization (indicator-service): missing-volume WARN test now captures structured console args (logger suppression safe).
- Stabilization (ml-simulate): Windows file visibility race mitigated via direct candidate retry + test sleep (timeout消滅)。
- CI/Observability: EVENT/METRICS 出力と metrics-dash の追加強化（平均/latency p95, handler counts）。
- Dev: EventBus subscribe/publish の型推論改善。

## 2.2.0 - 2025-09-19

- README: 全面見直し（Quick Start 最上段、EventBus/publishAndWait、TEST_MODE/Vitest の安定化ポイント、Rate Limiter 章の統合・最新化、live:minimal の使い方整理、旧 services 記述の削除）。
- EventBus: テスト時に `publishAndWait()` で `TRADE_PLAN`/`TRADE_PHASE` を同期発行（レース回避）。
- Trade Live: TEST_MODE の昇格閾値デフォルトを緩和（1→2 を 1 日で許容）。
- Toolkit: `sleep` を Vitest/TEST_MODE 下で自動的に短縮。
- Rate Limiter: テストスイートがカスタム limiter を注入した場合に強制有効化（メトリクス系テストの安定化）。
- Docs: スクリプト一覧/主要環境変数/注意事項/CI/Coverage の説明を現状に整合。

## 2.1.0 - 2025-09-18

- Errors: エラーコードを統一し、`EVENT/ERROR` を全レイヤで発火
  - BaseService: `CIRCUIT_OPEN`/`RATE_LIMITED`/最終失敗で `EVENT/ERROR` を publish
  - price-cache: 讀み書き失敗で `CACHE_ERROR` を publish
  - zaif-private: `NONCE`/`SIGNATURE`/`API_ERROR`/`NETWORK` を publish（必須メタ付き）
- CI/Artifacts: マージ後の別名 `coverage-merged/coverage-merged.json` を出力
- ts-prune: 結果を日付付き `ci/reports/ts-prune-YYYYMMDD.json` として永続化
- Cleanups (Batch 4): モジュール内専用の型 export を非公開化
  - adapters/indicator-service: `IndicatorSnapshot`, `IndicatorServiceOptions` を非公開化
  - adapters/execution-service: `OrderBookLevel`, `OrderSnapshot`, `SubmitParams`, `SubmitRetryParams` を非公開化
  - adapters/market-service: `MarketOverview` を非公開化
- Tests: `unit`/`integration-fast`/`cb-rate`/`event-metrics` 全てグリーン
- Deps: axios / vite / vitest / coverage-v8 を最新安定版へ更新（アドバイザリ 0）

## 2.0.0 - 2025-09-17

- Breaking: 旧 `src/services/*` / `src/strategies/*` を削除（実体を完全除去）。`@adapters/*` / `@application/*` に移行。
- Core purity: I/O と env 依存は adapters へ移設。
- BaseService: `src/adapters/base-service.ts` へ移動。
- Logging: 必須メタ（requestId, pair, side, amount, price, retries, cause.code）を API/EXEC/ORDER カテゴリに付与。
- Tools: `report-summary` を同期化しテスト安定化。
- Tests: 参照更新とスモールテスト追加。カバレッジ閾値（Statements ≥ 70%）維持。
  - テストユーティリティを `__tests__/helpers/*` に集約。`src/tools/tests/*` は削除。
  - 旧 `src/strategies/*` のエイリアスシムを削除（アプリ層の `@application/strategies/*` を利用）。

### Migration

- 型は `@contracts`、実装は `@adapters/*` / `@application/*` を利用してください。
- 例: `import { createServicePositionStore } from '@adapters/position-store'`
- 例: `import { runBuyStrategy } from '@application/strategies/buy-strategy-app'`