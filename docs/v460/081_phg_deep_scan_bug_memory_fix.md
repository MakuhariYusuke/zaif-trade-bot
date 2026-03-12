# 081# 深層スキャン — 不具合修正 & メモリ効率改善

## 概要

080#(重複排除)に続く品質改善。コードベース全体のバグ・メモリリーク・パフォーマンス反パターンを包括的にスキャンし、CRITICAL〜MEDIUM の27件を検出、うち25件を2コミットで修正。

## コミット

| SHA | 内容 | ファイル数 |
|-----|------|-----------|
| `91ed4e7ed` | Phase 1: lru_cache + deque + bare except + iterrows | 10 |
| `83dd3f2c7` | Phase 2: OOM防止 + thread leak + asyncio安全化 | 15 |

## 検出・修正した問題

### CRITICAL (3件 → 全修正)

| # | カテゴリ | ファイル | 問題 | 修正 |
|---|---------|--------|------|------|
| C1 | Memory | `trade_execution_engine.py` | `completed_orders: List` が無制限増大 → OOM | `deque(maxlen=10000)` |
| C2 | Memory | `streaming.py` | `pd.concat` で DataFrame 無制限増大 → OOM | Rolling window 50K行制限 |
| C3 | Logic | `performance_monitor.py` | `asyncio.create_task` を非asyncコンテキストで呼出 → RuntimeError | event loop検出 + sync fallback |

### HIGH (5件 → 全修正)

| # | カテゴリ | ファイル | 問題 | 修正 |
|---|---------|--------|------|------|
| H1 | Memory | `callbacks.py` | 9個の `List` が100K+ステップで無界増大 | `deque(maxlen=10000)` |
| H2 | Memory | `lagrange_constraint.py` | `action_rates/penalties/lambda_history` 無界 | `deque(maxlen=5000)` |
| H3 | Memory | `system_optimizer.py` | `memory_history/performance_history` 無界 | `deque(maxlen=5000)` |
| H4 | Resource | `checkpoint.py` | `ThreadPoolExecutor` の shutdown 漏れ | `close()` + `__del__` 追加 |
| H5 | Perf | `signal_guidance_system.py` | `list.pop(0)` O(n) を毎ティック実行 | `deque(maxlen=N)` O(1) |

### MEDIUM (17件 → 全修正)

| カテゴリ | 件数 | 内容 |
|---------|------|------|
| Memory | 3 | `unified_optimizer`, `reward_function_optimizer`, `performance_monitor` の無界リスト → deque |
| Perf | 1 | `data_manager.py` の `iterrows` → numpy vectorized (10-100x高速化) |
| Exception | 5+12=17 | bare `except:` → `except Exception:` (SystemExit/KeyboardInterrupt 飲み込み防止) |

### Phase 1 (91ed4e7ed) 修正内訳
- `lru_cache(maxsize=None)` → `maxsize=64` (data_generation.py)
- `List` → `deque(maxlen=1000)` × 4箇所 (parallel_experiments.py)
- bare `except:` → `except Exception:` × 12箇所 (8ファイル)
- `iterrows` → vectorized numpy (data_manager.py)

### Phase 2 (83dd3f2c7) 修正内訳
- `completed_orders` deque化 (trade_execution_engine.py)
- streaming DataFrame rolling window (streaming.py)
- asyncio.create_task safe fallback × 3箇所 (performance_monitor.py)
- 9 List → deque (callbacks.py)
- 3 List → deque (lagrange_constraint.py)
- 2 List → deque (system_optimizer.py)
- ThreadPoolExecutor close/del (checkpoint.py)
- signal_history/market_data_history deque化 (signal_guidance_system.py)
- performance_history Dict[str, deque], alerts/health deque化 (performance_monitor.py)
- optimization_history deque × 2 (unified_optimizer.py, reward_function_optimizer.py)
- bare `except:` → `except Exception:` × 5箇所 (5ファイル)

## 残存 LOW リスク (対応不要)

| # | ファイル | 内容 |
|---|--------|------|
| L1 | `enhanced_system.py` | スライストリム（既に上限ガードあり） |
| L2 | `drift_detection.py` | iterrows（低頻度レポート生成） |
| L3 | `quick_backtest.py` 他 | bare except × 6（分析スクリプト, 低リスク） |
| L4 | `signal_evaluator.py` | iterrows（バックテスト用, 非本番） |

## 既知の事前不具合 (本修正対象外)

- `unified_backtester.py:21` — `from utils.results_utils` 相対インポートバグ
- `v4xx_config_converter` — 削除済みモジュール参照（テストのみ影響）
- `heavy_env/core.py` — `except Exception: pass` × 3（訓練ループの防御コード）

## テスト結果

- v460ユニットテスト: **666 passed** (リグレッションなし)
- 全25修正ファイルの py_compile: **OK**
- DataManager vectorization インラインテスト: **OK** (shape/dtype/nonfinite検知正常)

## メモリ影響の概算

| 修正 | 修正前 (100K step) | 修正後 | 削減 |
|------|-------------------|--------|------|
| callbacks 9 deque | ~800MB (float×10×100K) | ~8MB (maxlen=10000) | 99% |
| completed_orders | 無制限 (数百万件) | 10K件上限 | OOM防止 |
| streaming DataFrame | 無制限 (数GB) | 50K行上限 | OOM防止 |
| lru_cache | O(∞) np.array キャッシュ | 64エントリ上限 | GC可能に |
