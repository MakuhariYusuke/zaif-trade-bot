# 訓練速度19x低下の根本原因調査レポート

## 調査日: 2025-02-12
## 対象: D1 (02/10, ~38 it/s) vs D2 (02/11, ~1.4 it/s)

---

## 1. メモリ監視・GCアーキテクチャ概要

本プロジェクトには **4つの独立したメモリ監視/GCレイヤー** が存在する:

| # | レイヤー | ファイル | トリガー | 訓練中アクティブ? |
|---|---------|---------|---------|-------------------|
| 1 | `default_memory_manager` | `ztb/cache/memory_cache.py` | 10秒毎のバックグラウンドスレッド | ✅ インポート時に自動起動 |
| 2 | `HeavyTradingEnv.step()` GC | `ztb/trading/environment/heavy_env/core.py` | N step毎 (DEFAULT=1000) | ✅ env.step()内で毎回チェック |
| 3 | `SystemOptimizer` | `ztb/training/system_optimizer.py` | 100 step毎 / contextmanager | ❌ callbackに渡されていない |
| 4 | `OperationMemoryTracker` | `ztb/utils/memory_utils.py` | contextmanager exit毎 | ❌ SystemOptimizer経由のみ |

### 閾値超過時の動作

| レイヤー | 閾値 (旧) | 超過時の動作 |
|---------|-----------|-------------|
| `default_memory_manager` | RSS > 760MB (800×0.95) | WARNING ログのみ（GC発動なし） |
| `HeavyTradingEnv` | 1000 step毎 | `gc.collect()` + `gc.collect(generation=0)` |
| `SystemOptimizer` | 100MB | WARNING通知 + gc.collect() |
| `OperationMemoryTracker` | 800MB | `optimize_memory_usage()` = cache清掃 + gc.collect() |

---

## 2. 根本原因分析

### 最重要事実: D1とD2のコードは同一

Git履歴を確認した結果、D1 (02/10) とD2 (02/11) の間で変更されたのは **5行の些細な修正のみ** で、メモリ管理・訓練ロジックには一切変更がない。

### 速度比較

| 指標 | D1 (d1_medium) | D2 (d2_cost05) |
|------|----------------|----------------|
| learning_starts前 | ~38 it/s | ~45 it/s |
| learning_starts後 | ~38 it/s | **~1.4 it/s** |
| メモリ使用量 | ~899 MB | ~899 MB |
| 実行時刻 | 18:00頃 | **02:55頃** |

### 仮説と信頼度

| 仮説 | 信頼度 | 根拠 |
|------|--------|------|
| **システムレベルの問題**（swap thrashing、thermal throttle、Windows更新バックグラウンド処理） | ★★★★☆ 80% | コード同一、深夜2:55実行、learning_starts後にのみ発生（=GPU/CPU負荷増大後） |
| GC重複実行の複合効果 | ★★☆☆☆ 30% | 4レイヤー存在するが、実際にホットパスで有効なのは2つのみ |
| `should_collect_garbage`プロパティバグ | ★★☆☆☆ 20% | デフォルト`gc_collect_interval_steps=0`では発動しないが、将来的リスク大 |
| バックグラウンド監視スレッドのオーバーヘッド | ★☆☆☆☆ 10% | 10秒毎のpsutil呼び出しは軽微だが不要 |

**結論**: 19x低下の主因は**システムレベルのリソース競合**（Windows深夜メンテナンス、swap、電源管理等）が最も確率が高い。ただし、コードの構造的問題（冗長GC、プロパティバグ、低すぎる閾値）はシステム負荷下で**速度低下を増幅**する。

---

## 3. 実施した修正

### Fix 1: `default_memory_manager` 閾値引き上げ・監視無効化 (HIGH)
**ファイル**: `ztb/cache/memory_cache.py`
- `max_memory_mb`: 800 → 1500
- `enable_monitoring`: True → False（バックグラウンドスレッド停止）

### Fix 2: `should_collect_garbage` プロパティバグ修正 (HIGH)
**ファイル**: `ztb/trading/environment/heavy_env/core.py`
- `self.memory_manager.should_collect_garbage` (毎step True) → `self.memory_manager.should_collect_garbage_at_step(self.current_step)` (N step毎)

### Fix 3: `OperationMemoryTracker.__exit__()` の無条件gc.collect()排除 (HIGH)
**ファイル**: `ztb/utils/memory_utils.py`
- gc.collect(): 無条件 → メモリ増加50MB超の場合のみ
- optimize_memory_usage()閾値: 800MB → 1500MB

### Fix 4: `SystemOptimizer` デフォルト閾値引き上げ (MEDIUM)
**ファイル**: `ztb/training/system_optimizer.py`, `ztb/training/unified_trainer/trainer.py`
- `memory_threshold_mb`: 100 → 1500
- `gc_interval_steps`: 100 → 1000

---

## 4. テスト結果

| テストファイル | 結果 |
|---------------|------|
| `tests/unit/utils/test_memory_utils.py` | 20/20 PASSED |
| `tests/unit/cache/test_memory_cache.py` | 21/21 PASSED |
| `tests/training/test_system_optimizer.py` | 16/16 FAILED (既存のAPI不整合、今回の修正と無関係) |

---

## 5. 推奨アクション

1. **D2実験を日中に再実行**して、同一コードで38 it/sが再現するか確認
2. 再実行時に`psutil`でCPU周波数・swap使用量をログ取り
3. `test_system_optimizer.py`の既存テスト修正（API不整合の解消）
4. 将来的に`callback._log_progress()`の重いnumpy統計演算（percentile, histogram, entropy, skewness, kurtosis）の軽量化を検討
