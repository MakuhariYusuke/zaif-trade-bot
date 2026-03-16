# ztb 配下の既存実装分析と構造化提案

**作成日**: 2026-01-15  
**対象**: Phase 4 Session 3 準備

---

## 1. ztb 配下の主要ディレクトリ構造

### 現状の構成（33ディレクトリ）

```
ztb/
├── adaptation/          # 適応型学習戦略（未確認）
├── adapters/            # 外部API/インターフェース適応層
├── analysis/            # 分析・評価モジュール群
├── api/                 # REST/gRPC APIインターフェース
├── app/                 # アプリケーションメインループ
├── benchmarks/          # パフォーマンスベンチマーク
├── cache/               # キャッシング機構（TTL/メモリ制御）
├── config/              # 設定管理（YAML/JSON）
├── configs/             # 設定ファイル格納
├── contracts/           # スマートコントラクト/取引契約定義
├── core/                # コア機能（trade execution, risk management）
├── data/                # データパイプライン（streaming, external sources）
├── evaluation/          # モデル品質評価・昇格エンジン
│   ├── walk_forward/     ← Walk-Forward評価フレームワーク（Session 2で完成）
│   ├── evaluator/       ← 評価器
│   └── ...
├── experiments/         # 実験・検証スクリプト
├── features/            # 特徴量生成・管理
├── inference/           # モデル推論エンジン
├── io/                  # I/O操作（データローダー等）
├── live_trading/        # ライブ取引実行エンジン
├── metrics/             # メトリクス計算（Sharpe, Drawdown, Win Rate等）
├── ml/                  # ML パイプライン（訓練、変換、エクスポート）
├── multimodal/          # マルチモーダル学習（複数データソース）
├── ops/                 # 運用ツール（ログ、監視、レポート）
├── optimization/        # 最適化機構（ハイパーパラメータ、メモリ等）
├── portfolio/           # ポートフォリオ管理（リスク配分）
├── preprocessing/       # データ前処理（正規化、欠損値補完）
├── processing/          # データ処理パイプライン
├── reports/             # レポート生成
├── risk/                # リスク管理（Circuit Breaker, Position Manager等）
├── support/             # サポートユーティリティ
├── tests/               # ユニット・統合テスト
├── tools/               # CLIツール・運用スクリプト
├── trading/             # 取引エンジン（RL環境、トレーニング）
├── training/            # 訓練システム（SAC, PPO, 最適化）
├── types/               # 型定義・プロトコル
├── utils/               ⭐ **重要な汎用ユーティリティ群**
└── sac_v426_improvement/ # 古いバージョン実装（deprecated）
```

---

## 2. 重要な既存実装の棚卸し

### 2.1 ztb/utils 配下の重要な機能

#### **ファイルI/O & チェックポイント管理**
| モジュール | 主要クラス/関数 | 用途 | 統合状況 |
|-----------|--------------|------|---------|
| `checkpoint.py` | `CheckpointManager`, `HierarchicalCheckpointManager`, `TrainingStateManager` | モデル・訓練状態の保存・復元、圧縮対応 | ✅ ztb.evaluation.walk_forward.checkpoint で再実装・統合済 |
| `file_utils.py` | `safe_json_load()`, `safe_json_dump()`, `load_config_file()` | JSON I/O、安全ファイル操作 | ✅ walk_forward.checkpoint で使用中 |
| `path_utils.py` | `ensure_dir()`, `get_project_root()` | ディレクトリ管理、プロジェクトパス解決 | ✅ 各モジュールで活用 |

#### **キャッシング & 性能管理**
| モジュール | 主要クラス | 用途 | 状況 |
|-----------|----------|------|------|
| `cache_utils.py` | `TTLCache`, `MemoryAwareCache` | TTL型キャッシュ、メモリ認識キャッシュ | ⚠️ 定義済みだが活用がスパース |
| `cache/feature_cache.py` | `FeatureCache` | 特徴量キャッシング | ⚠️ 活用度不明 |
| `performance_profiler.py` | `PerformanceProfiler` | CPU/メモリプロファイリング | ⚠️ 定義済みだが活用がスパース |
| `memory_monitor.py` | `MemoryMonitor` | メモリ監視 | ⚠️ 定義済み |

#### **エラーハンドリング & 例外管理**
| モジュール | 主要関数 | 用途 | 状況 |
|-----------|---------|------|------|
| `error_utils.py` | `safe_operation_context()` | 例外隔離、安全オペレーション | ⚠️ 定義済みだが活用がスパース |
| `analysis_errors.py` | `AnalysisError` 系例外 | 分析エラーの型安全性 | ⚠️ 定義済み |

#### **設定管理**
| モジュール | 主要クラス | 用途 | 状況 |
|-----------|-----------|------|------|
| `config_manager.py` | `ConfigManager` | 設定の読み込み・保存・検証 | ✅ 訓練システム全般で活用 |
| `config_loader.py` | `ConfigLoader` | 設定ファイル読み込み | ✅ config_manager で活用 |
| `config.py` | `ZTBConfig`, `ValidatedConfig` | 設定データ構造 | ✅ 訓練システムで活用 |

#### **メトリクス計算**
| モジュール | 主要関数 | 用途 | 統合状況 |
|-----------|---------|------|---------|
| `metrics/metrics.py` | Sharpe、DrawDown、Win Rate関数 | 取引パフォーマンス計算 | ✅ walk_forward.evaluator で使用中（utils系は非推奨） |

---

### 2.2 ztb/training 配下の重要な機能

#### **既存最適化・パフォーマンス機構**
| モジュール | 内容 | 現状 | Session 3活用可能性 |
|-----------|------|------|-------------------|
| `unified_optimizer.py` | `BayesianOptimizer`, `GridSearchOptimizer`, `PerformanceOptimizer` ほか | **2600行以上** の包括的最適化フレームワーク | ⭐⭐⭐ 高 - ハイパーパラメータ最適化、メモリ最適化、報酬関数最適化済み |
| `optimization/` | 最適化サブモジュール（early_stopping, lagrange_constraint, memory_efficient_loader等） | 複数の最適化戦略実装済み | ⭐⭐ 中 - メモリ効率化には活用可能 |
| `online_learning_engine.py` | `OnlineLearningEngine` - 非同期データ処理、リアルタイム学習 | ThreadPoolExecutor活用、非同期処理対応 | ⭐⭐ 中 - 並列処理パターン参考に |

#### **訓練・アルゴリズム管理**
| モジュール | 主要クラス | 用途 | 状況 |
|-----------|-----------|------|------|
| `core/algorithm_trainer.py` | `AlgorithmTrainer` | アルゴリズム選択と訓練実行 | ✅ 現在使用中 |
| `algorithms/` | AlgorithmFactory パターン | プラガブルアルゴリズムアーキテクチャ | ✅ PPO向けに実装済み |
| `callbacks/` | CallbackManager, BaseCallback | イベント駆動型コールバックシステム | ⚠️ 定義済みだが活用度不明 |

---

### 2.3 ztb/evaluation 配下の重要な機能

#### **評価フレームワーク**
| モジュール | 状態 | 説明 |
|-----------|------|------|
| `walk_forward/` | ✅ **完成** | Walk-Forward 分析フレームワーク（Session 2で完成）<br>- checkpoint.py: ztb.utils パターン統合済み<br>- evaluator.py: ウィンドウごとの訓練・評価実装済み<br>- types.py: 型定義済み |
| `evaluator/` | ⚠️ 部分実装 | TradingEvaluator（分析系） |
| `unified_evaluation.py` | ⚠️ 部分実装 | UnifiedEvaluator （統合評価） |

---

## 3. DRY原則違反の検出

### 3.1 **チェックポイント管理の重複実装**

**問題点**:
- `ztb/utils/checkpoint.py`: 汎用CheckpointManager（3個のクラス）
- `ztb/evaluation/walk_forward/checkpoint.py`: Walk-Forward特化版（Session 2で新規実装）

**重複機能**:
- 圧縮/解凍（zlib, lz4, zstd対応）
- メタデータ管理
- 階層的チェックポイント

**推奨対応**:
- `ztb/utils/checkpoint.py` を汎用マスター実装に統一
- `walk_forward/checkpoint.py` は ztb.utils.checkpoint をラップする軽量アダプタに変更

### 3.2 **エラーハンドリング機構の潜在的重複**

**問題点**:
- `ztb/utils/error_utils.py`: `safe_operation_context()` 定義あり
- 各モジュールで個別にtry-exceptブロック実装している可能性

**推奨対応**:
- `error_utils.py` の拡張（コンテキストマネージャー化、ロギング強化）
- 全モジュールに統一的な例外処理フレームワークの提供

### 3.3 **キャッシング機構の低活用**

**問題点**:
- `TTLCache`, `MemoryAwareCache`, `FeatureCache` 定義済み
- 実際の訓練・評価ループでの活用がスパース
- Session 3のパフォーマンス最適化で必要不可欠なのに整備不足

**推奨対応**:
- 統一的なキャッシング戦略ドキュメント作成
- 評価ループへの統合（特に「特徴量計算キャッシング」）

---

## 4. Session 3 向け構造化提案

### 4.1 **パフォーマンス最適化モジュールの統一設計**

```
ztb/optimization/parallel/
├── __init__.py
├── window_evaluator.py      ← Walk-Forward用マルチプロセッシング
│   ├── ParallelWindowEvaluator     (multiprocessing.Pool対応)
│   ├── WindowTask                  (プロセス間通信用データクラス)
│   └── WindowTaskResult
├── executor.py               ← 実行エンジン
│   ├── ProcessPoolExecutor   (multiprocessing.Pool ラッパー)
│   └── ExecutionMetrics      (実行時間、メモリ利用統計)
├── profiler.py               ← ボトルネック検出
│   ├── CPUProfiler           (cProfile統合)
│   └── MemoryProfiler        (memory_profiler統合)
├── scheduler.py              ← スケジューリング & 負荷分散
│   ├── DynamicScheduler      (動的タスク配分)
│   └── LoadBalancer          (プロセス間負荷均衡)
└── cache_coordinator.py      ← キャッシュ管理（マルチプロセッシング対応）
    ├── SharedMemoryCache     (multiprocessing.Manager.dict)
    └── CacheStrategy         (TTL vs LRU選択)
```

**特徴**:
- `ztb/training/unified_optimizer.py` との連携（既存最適化フレームワーク活用）
- `ztb/utils/checkpoint.py` との統合（中間結果保存）
- `ztb/utils/cache_utils.py` の拡張（プロセス間共有メモリ対応）

### 4.2 **エラーハンドリングの統一フレームワーク**

```
ztb/utils/error_handling/
├── __init__.py
├── decorators.py
│   ├── safe_operation()      ← コンテキストマネージャー化
│   ├── retry_on_error()      ← リトライロジック
│   └── error_aggregator()    ← エラー集約（複数プロセス対応）
├── strategies.py
│   ├── FailFastStrategy      (初期エラーで中止)
│   ├── FailSafeStrategy      (エラーを記録し継続)
│   └── PartialRecoveryStrategy (復帰可能なエラーを手動リカバリ)
└── types.py
    ├── OperationResult[T]    (Result型パターン)
    └── ErrorContext          (エラー文脈保持)
```

### 4.3 **チェックポイント管理の統一**

```
ztb/utils/checkpoint.py (改良版)
├── CheckpointManager            ← 既存、コア実装
├── HierarchicalCheckpointManager ← 既存、レイヤー分け
├── TrainingStateManager          ← 既存、訓練状態管理
├── [NEW] CheckpointSerializer    ← 圧縮/フォーマット抽象化
│   ├── CompressionStrategy (zlib/lz4/zstd)
│   └── FormatDetector (自動フォーマット判定)
└── [NEW] MultiProcessCheckpoint  ← マルチプロセッシング対応
    └── ProcessSafeCheckpoint

# 利用方: ztb/evaluation/walk_forward/checkpoint.py は以下のように簡潔化
from ztb.utils.checkpoint import CheckpointManager
class WalkForwardCheckpoint(CheckpointManager):
    """Walk-Forward特化の軽量アダプタ"""
    # ウィンドウ管理ロジックのみを実装
```

### 4.4 **キャッシング戦略の統一**

```
ztb/utils/cache/
├── __init__.py
├── ttl_cache.py          ← TTLCache (既存改良)
├── memory_aware_cache.py ← MemoryAwareCache (既存改良)
├── [NEW] feature_cache_coordinator.py
│   ├── FeatureCacheManager      (複数キャッシュ一括管理)
│   ├── CacheHitMeasurer         (キャッシュ効率計測)
│   └── AdaptiveCacheStrategy    (ワークロード適応的選択)
└── [NEW] multiprocessing_cache.py
    ├── SharedDictCache          (multiprocessing.Manager.dict)
    └── ProcessSafeTTLCache      (プロセス間安全なTTL)
```

---

## 5. Session 3 向けの実装優先順位

### Phase 1: 基盤整備（3-4時間）
1. **エラーハンドリングの統一** (優先度⭐⭐⭐)
   - `ztb/utils/error_handling/` 作成
   - 既存 `error_utils.py` とマージ
   
2. **チェックポイント統一** (優先度⭐⭐⭐)
   - `ztb/utils/checkpoint.py` を確定版に
   - Walk-Forward版を軽量アダプタに変更

### Phase 2: パフォーマンス最適化実装（4-6時間）
3. **マルチプロセッシング実装** (優先度⭐⭐⭐⭐)
   - `ztb/optimization/parallel/` 構築
   - 50ウィンドウ並列化（25時間 → 2-4時間）

4. **プロファイリング & 監視** (優先度⭐⭐⭐)
   - `window_evaluator.py` に性能測定組み込み
   - cProfile + memory_profiler統合

### Phase 3: キャッシング最適化（2-3時間）
5. **キャッシング戦略統合** (優先度⭐⭐)
   - 特徴量計算キャッシング
   - モデル推論キャッシング

---

## 6. 実装チェックリスト

### Preparation Phase
- [ ] 既存 `ztb/utils/checkpoint.py` の詳細コード読み込み
- [ ] `unified_optimizer.py` から並列化パターン抽出
- [ ] 既存 `callback/` システムの活用可能性確認

### Implementation Phase
- [ ] エラーハンドリング統一フレームワーク実装
- [ ] チェックポイント管理の統一
- [ ] マルチプロセッシング評価器実装
- [ ] プロファイリング機能統合
- [ ] テスト (32/32 passing 維持)

### Validation Phase
- [ ] 実際のパフォーマンス測定（50ウィンドウ）
- [ ] メモリ使用量確認
- [ ] エラー復帰テスト
- [ ] キャッシング効率計測

---

## 7. 既存実装の活用ガイドライン

### Session 3 で確実に活用すべき既存コード

| モジュール | 活用箇所 | 理由 |
|-----------|--------|------|
| `ztb/utils/checkpoint.py` | マルチプロセッシング対応化の基盤 | 圧縮・メタデータ管理の成熟実装 |
| `ztb/training/unified_optimizer.py` | 並列実行設定の参考 | `max_parallel_trials` などの実績パターン |
| `ztb/utils/config_manager.py` | パラレル実行設定管理 | ConfigManager 再利用で統一性確保 |
| `ztb/evaluation/walk_forward/evaluator.py` | 既存評価ロジック保持 | ウィンドウ単位の訓練・評価ロジックは変更不要 |
| `ztb/utils/file_utils.py` | 並列タスク結果の安全保存 | safe_json_dump で競合状態回避 |
| `ztb/metrics/metrics.py` | メトリクス計算（変更不要） | Session 2で既に統合済み |

### 活用が不十分な既存コード

| モジュール | 改善提案 | 実装予定 |
|-----------|--------|--------|
| `TTLCache` / `MemoryAwareCache` | 評価ループへの統合 | Session 3 Phase 3 |
| `PerformanceProfiler` | パフォーマンス監視拡張 | Session 3 Phase 2 に組み込み |
| `callback/` システム | マルチプロセッシング対応 | Session 3 で検討 |
| `error_utils.safe_operation_context()` | 実装例追加、ドキュメント充実 | Session 3 準備段階 |

---

## 8. 結論 & 次のステップ

### 現状評価
✅ **強み**:
- ztb/utils 配下に高度な汎用機能が豊富に揃っている
- チェックポイント、キャッシング、最適化フレームワークが揃っている
- Walk-Forward フレームワークは Session 2 で高水準に完成

⚠️ **課題**:
- 汎用コンポーネント間の統合度がスパース（DRY原則違反の可能性）
- キャッシング、エラーハンドリングの活用度が低い
- マルチプロセッシング対応が不足（Session 3 の鍵）

### 推奨アクション
**Session 3 開始前**:
1. 本ドキュメントを参考に、`ztb/optimization/parallel/` 基本構造を設計
2. `ztb/utils/checkpoint.py` と `ztb/utils/error_handling/` の拡張計画を策定
3. 既存 `unified_optimizer.py` と `online_learning_engine.py` からパターン抽出

**Session 3 実装時**:
- 提案した構造に従い、段階的に実装（Phase 1 → 2 → 3）
- 既存テストスイート (32/32) を保持・拡張
- ドキュメント更新は毎段階で実施（--no-verify オプション使用）

---

**目標**: 高い保守性、再利用性、パフォーマンスを兼ね備えた、本プロジェクト短期高収益性実現の技術基盤確立。

