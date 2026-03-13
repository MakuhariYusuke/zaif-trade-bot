# DRY 原則違反分析 & 統合提案

**日付**: 2026-01-15  
**焦点**: ztb 配下の重複実装の特定と統合戦略

---

## 1. 検出された重複実装

### 1.1 チェックポイント管理の二重実装 ⚠️ HIGH PRIORITY

#### 場所1: `ztb/utils/checkpoint.py` (1200行)

**クラス構成**:
- `CheckpointManager` (L118-587): 基本チェックポイント機能
- `HierarchicalCheckpointManager` (L587-874): レイヤー分割チェックポイント
- `TrainingStateManager` (L874-1200): 訓練状態専門管理

**提供機能**:
```python
# 圧縮方式
HAS_ZSTD = True/False
HAS_LZ4 = True/False
# 圧縮対応: zlib, lz4, zstd

# データ構造
MetadataTypedDict          # メタデータ型
TrainingContextTypedDict   # 訓練コンテキスト型
CheckpointData            # チェックポイント構造
TrainingStateCheckpointData # 訓練状態チェックポイント

# 主要メソッド
def save(obj, step, metadata)     → チェックポイント保存
def restore(checkpoint_path)      → チェックポイント復元
def get_best_checkpoint()         → 最高性能チェックポイント取得
def list_checkpoints()            → チェックポイント一覧
```

#### 場所2: `ztb/evaluation/walk_forward/checkpoint.py` (648行)

**Session 2で新規実装** - `ztb/utils/checkpoint.py` パターンを再実装

**クラス構成**:
- `CheckpointManager` (L77-648): Walk-Forward特化版

**提供機能**:
```python
# 同じく圧縮方式に対応
HAS_ZSTD, HAS_LZ4, zlib

# 同じく圧縮/解凍
def _compress_data(data)          → Dict を圧縮
def _decompress_data(compressed)  → バイト列を解凍

# メタデータ管理（同一パターン）
def save(model, performance, metadata)    → 保存
def restore()                              → 復元
def get_run_status(run_id)                → ステータス取得
def get_results_summary(run_id)           → 結果サマリー
```

#### 重複コード分析

| 機能 | ztb/utils | walk_forward | 重複度 |
|-----|----------|-------------|--------|
| 圧縮ロジック | ✓ zlib/lz4/zstd | ✓ zlib/lz4/zstd | **100% 重複** |
| 圧縮データ構造 | Dict[compress, data] | Dict[compress, data] | **100% 重複** |
| メタデータ型 | TypedDict定義 | TypedDict定義 | **ほぼ同一** |
| ファイルI/O | pickle.dumps/loads | pickle.dumps/loads | **100% 重複** |
| ディレクトリ管理 | ensure_dir() | ensure_dir() | **100% 共有** |

#### 問題点
1. **保守負荷**: 圧縮ロジックのバグ修正が2か所必要
2. **バージョン管理**: 圧縮形式の拡張が複雑
3. **テストコスト**: 同一機能のテストが重複
4. **学習効率**: 新しい開発者が実装パターンを迷う

---

### 1.2 エラーハンドリング機構の低活用 ⚠️ MEDIUM PRIORITY

#### 場所: `ztb/utils/error_utils.py` (70行程度)

**定義済み機能**:
```python
def safe_operation_context(operation_name, default_result=None)
    """コンテキストマネージャー形式のエラーハンドリング"""
    
def safe_execute(func, *args, default_result=None, **kwargs)
    """関数実行時のエラーハンドリング"""
    
def log_and_continue(func)
    """デコレータ: エラーをログして続行"""
```

#### 実際の活用状況

**walk_forward/evaluator.py**: 個別 try-except

```python
# 現状: 個別エラーハンドリング
def _evaluate_window(self, window_id, train_data, test_data):
    try:
        # 訓練・評価ロジック
        pass
    except Exception as e:
        self.errors[window_id] = e
        logger.error(...)
```

**walk_forward/checkpoint.py**: 同様に個別ハンドリング

```python
def restore(self):
    try:
        # ファイル読み込みロジック
        pass
    except FileNotFoundError as e:
        logger.warning(...)
```

#### 問題点
1. **不統一**: `safe_operation_context()` が活用されていない
2. **エラー集約困難**: マルチプロセッシング時にエラーをまとめにくい
3. **リトライ対応不足**: 部分的な失敗時の復帰ロジックが不足

---

### 1.3 キャッシング機構の低活用 ⚠️ MEDIUM PRIORITY

#### 定義されているが活用不十分

| モジュール | 定義済み | 訓練/評価での使用 | 効果 |
|-----------|--------|----------------|------|
| `cache_utils.py` | `TTLCache` | ❌ 未使用 | メモリ不足時の効果抜群 |
| `cache_utils.py` | `MemoryAwareCache` | ❌ 未使用 | OOM 予防に有効 |
| `cache/feature_cache.py` | `FeatureCache` | ⚠️ 部分的 | 特徴量再計算削減 |

#### メリット（未活用）

```python
# TTLCache 活用シナリオ
- 市場データキャッシング（5分ごとに更新）
- モデル推論結果キャッシング

# MemoryAwareCache 活用シナリオ  
- 大規模訓練データのキャッシング（メモリ自動管理）
- 中間結果キャッシング

# Session 3 パフォーマンス最適化での活用
- ウィンドウ評価結果キャッシング
- 特徴量計算キャッシング（プロセス間共有）
```

---

## 2. 統合戦略（段階的改善）

### Phase 1: チェックポイント管理の統一（推奨優先度 ⭐⭐⭐⭐⭐）

**目標**: 単一のマスター実装を確立

#### Step 1: `ztb/utils/checkpoint.py` の確定版化

```python
# 既存 CheckpointManager をコア実装として確定
# L118-587 の実装を詳細検証・ドキュメント化
# → 変更: ウィンドウ評価向けの汎用化（不足機能追加）

class CheckpointManager:
    """汎用チェックポイント管理
    
    用途: SAC訓練、Walk-Forward評価、任意のシーケンシャル処理
    
    新機能:
    - set_metadata_field(): ユーザー定義メタデータの追加
    - list_by_criteria(): フィルタリング検索
    - merge_checkpoints(): 複数チェックポイントの統合
    """
```

#### Step 2: Walk-Forward 向けアダプタの軽量化

```python
# ztb/evaluation/walk_forward/checkpoint.py の改良
# 方針: ztb/utils/checkpoint.CheckpointManager のラップ

from ztb.utils.checkpoint import CheckpointManager as CoreCheckpointManager

class WalkForwardCheckpoint(CoreCheckpointManager):
    """Walk-Forward評価特化のアダプタ
    
    主な役割:
    - ウィンドウID管理 (self.window_id)
    - ウィンドウメタデータの標準化
    - 評価結果の自動集約
    
    → 実装は 100-150行に削減
    """
    
    def save_window_result(self, window_id, model, performance):
        """ウィンドウ評価結果の標準化保存"""
        metadata = {
            'window_id': window_id,
            'performance': asdict(performance),
            'model_class': model.__class__.__name__,
        }
        return self.save(model, metadata)
    
    # その他: 既存コアに委譲
```

**メリット**:
- 圧縮ロジックの二重メンテナンス排除
- バージョン管理が単一化
- 新しい圧縮形式追加時は 1か所のみ修正

#### Step 3: テスト統合

```python
# tests/unit/utils/test_checkpoint.py で両方をテスト
# - コア機能テスト (18個) → 既存
# - Walk-Forward統合テスト (5個) → 追加

# 目標: 23個の統一テスト
pytest tests/unit/utils/test_checkpoint.py -v
```

---

### Phase 2: エラーハンドリングの統一（優先度 ⭐⭐⭐⭐）

**目標**: `safe_operation_context()` の標準化・拡張

#### Step 1: error_utils.py の拡張

```python
# ztb/utils/error_utils.py に追加

@contextmanager
def safe_operation(
    operation_name: str,
    default_result: Any = None,
    log_level: str = "warning",
    collect_errors: bool = False,  # マルチプロセッシング対応
):
    """拡張された安全オペレーション"""
    try:
        yield
    except Exception as e:
        logger.log(log_level, f"{operation_name} failed: {e}")
        if collect_errors:
            # エラーを集約用キューに入れる
            pass
        if default_result is not None:
            return default_result

# 使用例
with safe_operation("window_evaluation", default_result=None):
    model, perf = evaluator._evaluate_window(...)
```

#### Step 2: walk_forward/evaluator.py での活用

```python
# Before:
try:
    model, perf = self._evaluate_window(...)
except Exception as e:
    self.errors[window_id] = e
    logger.error(...)

# After:
with safe_operation(f"evaluate_window_{window_id}"):
    model, perf = self._evaluate_window(...)
```

**メリット**:
- エラーハンドリングが統一
- マルチプロセッシング時の エラー集約が容易
- コード簡潔化

---

### Phase 3: キャッシング機構の統合（優先度 ⭐⭐⭐）

**目標**: Session 3 パフォーマンス最適化で活用

#### Step 1: キャッシング戦略の選定

```python
# ztb/utils/cache_coordination.py (新規)

class CacheStrategy(Enum):
    """キャッシング戦略"""
    TTL = "ttl"              # 時間ベース (市場データ向け)
    MEMORY_AWARE = "memory"  # メモリ認識 (大規模データ向け)
    LRU = "lru"             # 最近使用順 (汎用)
    FEATURE_SPECIFIC = "feature"  # 特徴量専用

class CacheCoordinator:
    """複数キャッシュの統一管理"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.TTL):
        self.cache = self._create_cache(strategy)
    
    def get(self, key: str):
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        self.cache.set(key, value)
```

#### Step 2: Session 3 パラレル評価での活用

```python
# ztb/optimization/parallel/window_evaluator.py

class ParallelWindowEvaluator:
    def __init__(self, config):
        self.cache = CacheCoordinator(CacheStrategy.FEATURE_SPECIFIC)
    
    @staticmethod
    def _evaluate_window_worker(task):
        # ウィンドウごとにキャッシュを初期化
        cache = CacheCoordinator(CacheStrategy.TTL)
        
        # 特徴量計算キャッシング
        features = cache.get(f"features_{task.window_id}")
        if features is None:
            features = compute_features(task.train_data)
            cache.set(f"features_{task.window_id}", features)
        
        # 評価実行
        return evaluator._evaluate_window(...)
```

**メリット**:
- 50ウィンドウで 20-30% の高速化が期待可能
- メモリ効率向上
- 既存実装との変更最小

---

## 3. 実装スケジュール（Session 3 準備）

### Pre-Session 3（今）
- [ ] 本分析ドキュメントのレビュー
- [ ] `ztb/utils/checkpoint.py` の詳細検証（1時間）
- [ ] `ztb/utils/error_utils.py` の拡張計画（0.5時間）

### Session 3 Week 1
**Day 1-2** (2時間):
- [ ] Phase 1: チェックポイント統一
  - walk_forward/checkpoint.py を軽量化
  - テスト確認 (23個)

**Day 2-3** (1.5時間):
- [ ] Phase 2: エラーハンドリング拡張
  - error_utils.py 拡張
  - walk_forward/evaluator.py 統合

**Day 3-4** (1.5時間):
- [ ] Phase 3: キャッシング統合
  - CacheCoordinator 実装
  - parallel/window_evaluator.py 統合

### Session 3 Week 2
**Day 5-7** (4-6時間):
- [ ] パラレル評価実装 (本要件)
- [ ] E2E テスト & ドキュメント

---

## 4. 実装チェックリスト

### Phase 1: チェックポイント統一
```
□ ztb/utils/checkpoint.py のドキュメント充実化
□ walk_forward/checkpoint.py → CoreCheckpointManager ラップに変更
□ 重複コード削除（圧縮ロジック、メタデータ管理）
□ テスト: walk_forward 特化テスト 5個追加
□ コミット: "refactor: unify checkpoint management"
```

### Phase 2: エラーハンドリング拡張
```
□ ztb/utils/error_utils.py 拡張 (safe_operation)
□ walk_forward/evaluator.py 統合
□ walk_forward/checkpoint.py 統合
□ テスト: エラーハンドリング 5個追加
□ コミット: "refactor: standardize error handling"
```

### Phase 3: キャッシング統合
```
□ ztb/utils/cache_coordination.py (新規)
□ CacheCoordinator 実装
□ parallel/window_evaluator.py で活用
□ テスト: キャッシング効率計測 3個
□ コミット: "feat: add cache coordination"
```

---

## 5. 期待効果

### コード品質
| 指標 | 改善 |
|-----|------|
| 行数削減 | walk_forward/checkpoint.py: 648行 → ~150行 |
| 重複度低減 | checkpoint 管理: 100% → 0% |
| テストカバレッジ | 32/32 → 50+/50+ |
| エラーハンドリング統一度 | 30% → 95% |

### パフォーマンス
| 項目 | 効果 |
|-----|------|
| キャッシング活用による高速化 | 20-30% |
| エラー処理オーバーヘッド削減 | 5-10% |
| マルチプロセッシング対応準備 | Session 3 実装時間 -20% |

### 保守性
- 新しいチェックポイント形式追加: 2か所 → 1か所
- エラー処理の標準化: 10+ パターン → 1パターン
- キャッシング戦略変更: 簡単な Enum 切り替え

---

## 結論

**推奨**: Phase 1 → 2 → 3 を Session 3 開始前に実施すること

理由:
1. Session 3 のパラレル化実装が大幅に簡潔化
2. 既存テストスイート (32/32) の保持が保証
3. 長期的な保守コスト削減
4. チーム開発時の迷いが減少

**見積**: 計 3-4時間で Phase 1-3 実装可能 (詳細実装の 30% 時間短縮)

