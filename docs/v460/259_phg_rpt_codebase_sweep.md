# 259# Codebase Sweep Report (v2 — 2026-03-04 更新)

## 概要

258# (AS Reservation Price / VPIN Continuous / RegimeDetectorLike Protocol) 実装後の
フルスイープ。5 カテゴリ × 定量分析。対象: `scripts/v460/lib/` 配下 51 ファイル

---

## 1. 型安全ギャップ (Type Safety Gaps)

### 1-A. `getattr(` — 29 箇所

| ファイル | 行 | 種別 | 説明 |
|---|---|---|---|
| config_hot_reload.py | L353-354,372-373 | dataclass field 比較 | `getattr(self._config, f.name)` — fields ループ。構造上必要 |
| config_hot_reload.py | L389 | callback dispatch | `getattr(runner, callback_name, None)` — Protocol 化で解消可 |
| skip_gate_evaluator.py | L234,236 | trade field 取得 | **`trade: object` が根因** — TradeProtocol で解消可 |
| skip_gate_evaluator.py | L484,519 | config 動的 side 参照 | `getattr(config, f"skip_gate_model_path_{side}")` — dict 化で代替可 |
| skip_gate_evaluator.py | L830-852 | self attr 動的参照 | reload 用。side→attr map dict で代替可 |
| skip_gate_evaluator.py | L991-1021 | adapter 動的参照 | **`adapter: object` が根因** |
| fill_config.py | L691-692 | validation | dataclass field 動的検証 |
| fill_cycle_executor.py | L1226 | order type check | `getattr(order, "order_id")` — OrderLike Protocol で解消済みのはず |
| ob_utils.py | L61-73 | OB level 抽象化 | **261# P2-1 計画済み** |
| resilience.py | L58 | CB state | `getattr(cb, name)` |
| micro_circuit_breaker.py | L370 | 内部 deque | `getattr(self, attr)` |
| fill_test_cli.py | L158 | log level | `getattr(logging, level_str)` — 標準パターン、許容 |
| tasks/sac_train.py | L204,318 | gym env | gym 型が不安定、許容 |

### 1-B. `hasattr(` — 6 箇所

| ファイル | 行 | 説明 |
|---|---|---|
| skip_gate_evaluator.py | L680,717,751 | `hasattr(primary_decision, "features_used")` — **SkipDecision に必須フィールド化で解消** |
| evaluator.py | L261 | `hasattr(model, "predict_proba")` — sklearn duck typing、許容 |
| ob_utils.py | L59 | `hasattr(level, "quantity")` — **261# P2-1 計画済み** |
| tasks/sac_train.py | L203 | gym 型、許容 |

### 1-C. `: object` 型注釈 — ~20 箇所 (コメント除く)

| ファイル | 行 | 深刻度 | 説明 |
|---|---|---|---|
| skip_gate_evaluator.py | L221,225,242 | **高** | `trade: object`, `trades: object` — Trade Protocol 必要 |
| skip_gate_evaluator.py | L900,910 | **高** | `adapter: object`, `maker_price_vpin_setter: object` |
| adaptation_engine.py | L170,416 | **高** | `fast_fill_defense: object`, `adapter: object` → `type: ignore` 連鎖の根因 |
| fill_loop_orchestrator.py | L89-91 | 中 | `_mcb/_sad/_cycle_strategy: object` — 228# 計画済み |
| fill_cycle_executor.py | L58 | 低 | `_current_regime_value: object` |
| ob_recorder.py | L36-131 | 中 | `value/levels/bids/asks: object` — 261# P2-1 計画済み |
| ob_utils.py | L65,84 | 中 | `ob: object`, `levels: object` — 261# P2-1 計画済み |
| config_access.py | L20-42 | 低 | `value: object` — coercion 関数、設計上意図的 |
| order_monitor.py | L96 | 中 | Protocol 戻り値 `-> object` → SkipGateResult に変更可 |
| config_loader.py | L25 | 低 | `raw: object` — 意図的 |

### 1-D. `type: ignore` — 12 箇所

| ファイル | 行 | 原因 |
|---|---|---|
| skip_gate_evaluator.py | L1023-1024 | `ob.bids/asks  # type: ignore[attr-defined]` — adapter: object 由来 |
| adaptation_engine.py | L268,319 | `type: ignore[union-attr]` — `fast_fill_defense: object|None` 由来 |
| adaptation_engine.py | L426,434 | `type: ignore[attr-defined]` — `adapter: object` 由来 |
| lock_manager.py | L14 | `psutil  # type: ignore[import-untyped]` — サードパーティ、許容 |
| event_logger.py | L117 | `TeeWriter  # type: ignore[assignment]` — IO redirect、意図的 |
| fill_cycle_executor.py | L58,66 | class-level default 型不一致 |
| fill_loop_orchestrator.py | L97 | `deque[FillRecord]  # type: ignore[assignment]` |
| fill_test_cli.py | L369 | `SIGBREAK  # type: ignore[attr-defined]` — Windows 固有、許容 |

---

## 2. メソッド複雑度 (100行超)

| ファイル | メソッド | 行数 | 深刻度 | 備考 |
|---|---|---|---|---|
| fill_loop_orchestrator.py | `run_continuous()` | **~1694** (L635→L2329) | **🔴 致命的** | MAX LINES: 1200 超過 |
| fill_cycle_executor.py | `run_single_cycle()` | **~705** (L675→L1380) | 🟠 高 | MAX LINES: 750 目前 |
| skip_gate_evaluator.py | `evaluate()` | **~346** (L893→L1239) | 🟡 中 | velocity rule + ML + ev_weighted 一括 |
| maker_price.py | `compute()` | **~180** (L1053→L1233) | 🟡 中 | 260# stage 分割後、パイプライン呼出しが長い |
| skip_gate_evaluator.py | `_try_ev_weighted_decision()` | **~141** (L546→L687) | 🟡 中 | EV加重統合。分割候補 |
| fill_cycle_executor.py | `_build_fill_record()` | **~108** (L534→L642) | 🟡 中 | field 数由来 |
| maker_price.py | `_apply_volatility_guard()` | **~106** (L832→L938) | 🟡 中 | VG + VPIN continuous + damping |

---

## 3. マーケット理論

### 実装済み

| 理論 | ファイル | 行 | 状態 |
|---|---|---|---|
| **Avellaneda-Stoikov (2008)** reservation price | maker_price.py | L527-600 | ✅ 258# 実装 |
| **AS 指数減衰** loss boost | maker_price.py | L980-1023 | ✅ 226# T1 |
| **Roll (1984)** σ推定 proxy | maker_price.py | L564-566 | ✅ spread/(2·mid) |
| **vol_ratio hybrid σ** | maker_price.py | L574-580 | ✅ 258# MT-4 |
| **Inventory skewing** (time-decay) | maker_price.py | L264-283 | ✅ 228# C2 |
| **AS offset** (EV-weighted) | skip_gate_evaluator.py | L687-758 | ✅ |

### 未実装 — 拡張候補

| 理論 | 用途 | 優先度 |
|---|---|---|
| **Kelly Criterion** | lot sizing 最適化 ($f^* = \frac{pb-q}{b}$) | 🟡 中 — 資金効率直結 |
| **Gueant-Lehalle-Fernandez-Tapia (2013)** | AS 有限期間拡張 — τ 動的化 | 🟡 中 |
| **AS 最適スプレッド幅** δ* | fill rate k による理論的 spread | 低 — k 推定が困難 |
| **Kyle (1985) λ** | 情報非対称性 | 低 — VPIN + imbalance で概ね代替 |
| **Amihud illiquidity** | spread_adaptive 閾値動的化 | 低 — BTC/JPY は流動性十分 |

**Active TODO/FIXME: 0 件** — 技術的負債としてクリーン。

---

## 4. エラーハンドリング — `except Exception` 78 箇所

| ファイル | 件数 | 深刻度 | 備考 |
|---|---|---|---|
| order_monitor.py | **12** | 🟠 高 | API障害 vs ロジック不具合の区別が必要 |
| skip_gate_evaluator.py | **12** | 🟡 中 | ML 評価の resilience — 一部 narrow 化可 |
| fill_loop_orchestrator.py | **10** | 🟡 中 | ループ継続用 — 意図的だがログ分類不十分 |
| fill_cycle_executor.py | **6** | 🟡 中 | PnL/注文監視フォールバック |
| config_hot_reload.py | **5** | 低 | config reload resilience |
| pnl_measurer.py | **5** | 🟡 中 | PnL 計測 resilience |
| event_logger.py | **4** | 低 | ログ出力 resilience |
| fill_test_cli.py | **4** | 低 | CLI トップレベル |
| phantom_position_guard.py | **3** | 低 | balance check |
| resilience.py | **3** | 低 | CB 自体 |
| adaptation_engine.py | **3** | 低 | |
| ob_utils.py | **3** | 低 | |
| batch_persistence.py | **2** | 低 | |
| lock_manager.py | **2** | 低 | |
| tasks/sac_train.py | **2** | 低 | |
| ab_judgment.py | **1** | 低 | |
| balance_checker.py | **1** | 低 | |

**問題パターン**: `except Exception:` (変数なし = ログ不能) が 3 箇所:
lock_manager.py L155, fill_loop_orchestrator.py L1235, fill_cycle_executor.py L1194

---

## 5. 未使用インポート / Dead Code

| パターン | ファイル | 説明 |
|---|---|---|
| `Optional` + `X|None` 混在 | adaptation_engine.py, fill_cycle_executor.py, abstract_cycle_runner.py | `from __future__ import annotations` 下では `Optional` import 不要 |
| `cast` 未使用の疑い | fill_cycle_executor.py L17 | `from typing import ..., cast` — 使用箇所要確認 |
| 重複 Protocol 定義 | order_monitor.py L96 | 簡易 Protocol vs SkipGateEvaluator 実 Protocol が乖離 |

---

## 6. 優先度付き改善提案

### 🔴 P1: 収益インパクト大 + 保守性改善

| # | 項目 | 工数 | インパクト | 対象 |
|---|---|---|---|---|
| **P1-1** | `run_continuous()` 1694行分割 | 4-6h | **高** — バグ混入リスク根本原因 | fill_loop_orchestrator.py |
| **P1-2** | `adapter: object` → Protocol 化 | 2-3h | **高** — type: ignore 6箇所 + getattr 8箇所 一括解消 | adaptation_engine.py, skip_gate_evaluator.py |
| **P1-3** | order_monitor.py except narrow化 | 2h | 中 — API障害時の誤判定防止 | order_monitor.py |

### 🟡 P2: 中期品質改善

| # | 項目 | 工数 | 対象 |
|---|---|---|---|
| **P2-1** | 261# OB Protocol 完遂 (計画済み) | 1-2h | ob_utils.py, ob_recorder.py |
| **P2-2** | `SkipDecision.features_used` 必須化 → hasattr 3箇所除去 | 30m | skip_gate_evaluator.py |
| **P2-3** | Kelly Criterion lot sizing PoC | 3-4h | lot_sizer.py |
| **P2-4** | `run_single_cycle()` 705行 — PnL計測分離 | 2h | fill_cycle_executor.py |
| **P2-5** | `evaluate()` 346行 → velocity/ML/ev 3段階分離 | 2h | skip_gate_evaluator.py |

### 🟢 P3: 低リスク清掃

| # | 項目 | 工数 |
|---|---|---|
| **P3-1** | `Optional` → `X|None` 統一 + 未使用 import 削除 | 30m |
| **P3-2** | `except Exception:` 変数なし 3 箇所にログ追加 | 15m |
| **P3-3** | `config_access.py` `value: object` ドキュメント追記 | 15m |

---

## 統計サマリ

| カテゴリ | 件数 | 前回比 |
|---|---|---|
| `getattr(` | 29 | -6 (261# 実施分) |
| `hasattr(` | 6 | -2 (259# adaptation_engine 修正分) |
| `: object` 型注釈 | ~20 (実質) | -7 (261# Protocol 化) |
| `type: ignore` | 12 | ±0 |
| `except Exception` | 78 | — (初回計測) |
| `Any` 型 | 0 | ✅ 完全排除 |
| bare `except` | 0 | ✅ 完全排除 |
| TODO/FIXME (active) | 0 | ✅ |
| 100行超メソッド | 7 | -2 (260# 分割分) |
| 1000行超メソッド | 1 | ±0 (`run_continuous`) |

**最重要ネクストアクション**: P1-1 + P1-2。
前者はバグ混入リスクの根本原因、後者は `type: ignore`/`getattr` チェーンの根因。
両者解消で型安全ギャップの **~40%** が一括消滅する。
