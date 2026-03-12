# 256# Self-Review (254#/255#) & Remaining-Issues Sweep

**日付**: 2026-03-03  
**対象**: `scripts/v460/lib/` + `scripts/v460/ml/`  
**前提**: 254# (ada932c76) + 255# (08b4c4c67) 修正済み

---

## SELF-REVIEW FINDINGS

### 254# (ada932c76) — clean ✅

| 変更 | 検証結果 |
|---|---|
| `FillTestState.one_sided_frozen_side: str \| None = None` (resilience.py L267) | ✅ 正しい型・デフォルト |
| snapshot: `one_sided_frozen_side=self._one_sided_frozen_side` (orchestrator L345) | ✅ 正しくスナップショット |
| restore: L459-462 `_fs = saved_state.one_sided_frozen_side; if _fs is not None` | ✅ None ガード付き復元 |
| getattr→直接参照 8件 (`_restore_common_state` L398/427/445/459) | ✅ FillTestState に全フィールド存在確認済み |
| `_recent_records` class-level default (L95) | ✅ `list[FillRecord] = []` + `# type: ignore[assignment]` |
| `_heartbeat_task` class-level default (L97) | ✅ `asyncio.Task[None] \| None = None` |
| `_check_stop_conditions` 直接参照 (L549) | ✅ `records = self._recent_records` |
| `cleanup_heartbeat` 直接参照 (L2332) | ✅ `task = self._heartbeat_task` |
| heartbeat psutil bare except (L1234) | ✅ `logger.debug("psutil memory check unavailable", exc_info=True)` |

**ラウンドトリップ検証**: `_build_state_snapshot()` で `one_sided_frozen_side=self._one_sided_frozen_side` → `_restore_common_state()` で `saved_state.one_sided_frozen_side` → 条件付き復元。None/str 両パスで正しい。

### 255# (08b4c4c67) — clean ✅

| 変更 | 検証結果 |
|---|---|
| skip_gate_evaluator `hot_reload_check_interval_sec` 直接参照 (L771) | ✅ `self._config.hot_reload_check_interval_sec` |
| skip_gate_evaluator `_select_gate_for_side` 直接参照 (L887-890) | ✅ `self._gate_buy` / `self._gate_sell` |
| skip_gate_evaluator `_gate_buy`/`_gate_sell` __init__ 宣言 (L115-116) | ✅ None 初期化済み |
| order_monitor `stale_reprice_skip_gate_offset` 直接参照 (L157) | ✅ `self._config.stale_reprice_skip_gate_offset` |
| resilience.py bare except (L175) | ✅ `logger.debug("disk_usage check failed", exc_info=True)` |
| pnl_measurer.py bare except (L112) | ✅ `logger.debug("interim PnL calc failed at tick %d", tick, exc_info=True)` |
| lock_manager.py bare except (L155) | ✅ `logger.debug("lockfile heartbeat update failed", exc_info=True)` |
| ob_utils.py bare except (L124, L135) | ✅ `logger.debug("bid/ask_depth_volume fetch failed", exc_info=True)` |
| fill_cycle_executor.py bare except (L1194) | ✅ `logger.debug("OB fetch failed during retry...", exc_info=True)` |

**一貫性**: 全 6 件が同一パターン `logger.debug(msg, exc_info=True)` を使用 ✅

---

## REMAINING ISSUES SWEEP

### A. getattr 残存 (39件)

**LEGITIMATE (34件)** — 動的ディスパッチ・duck-typing のため排除不可:

| ファイル | 行 | 用途 | 理由 |
|---|---|---|---|
| config_hot_reload.py | L353-354, 372-373 | `getattr(config, f.name)` | dataclass フィールド動的イテレーション |
| config_hot_reload.py | L389 | `getattr(runner, callback_name, None)` | 動的コールバックディスパッチ |
| config_hot_reload.py | L400 | `getattr(runner, "_fast_fill_defense", None)` | runner duck-typing |
| skip_gate_evaluator.py | L234, 236 | `getattr(trade, key, default)` | dict/object dual API |
| skip_gate_evaluator.py | L484, 519, 833 | `getattr(config, f"skip_gate_model_path_{side}")` | f-string 動的キー |
| skip_gate_evaluator.py | L830, 852 | `getattr(self, attr_path/attr_hash)` | _SIDE_MODEL_SLOTS ループ |
| skip_gate_evaluator.py | L991, 1015 | `getattr(adapter, "get_recent_trades/get_orderbook")` | adapter duck-typing |
| skip_gate_evaluator.py | L1021 (×2) | `getattr(ob, "bids/asks", None)` | OB duck-typing (存在チェック) |
| ob_utils.py | L30, 39, 50, 51, 122, 133 | `getattr(level/ob, ...)` | level/OB duck-typing |
| ob_recorder.py | L67, 69 (×2) | `getattr(level, "price/quantity/size")` | level duck-typing |
| micro_circuit_breaker.py | L370 | `getattr(self, attr)` | deque 動的アクセス |
| resilience.py | L58 | `getattr(cb, name)` | モジュール `__getattr__` lazy import |
| tasks/sac_train.py | L204, 318 | `getattr(env.action_space/env, ...)` | gym API 互換 |
| fill_test_cli.py | L158 | `getattr(logging, config.file_log_level)` | 動的ログレベル |
| fill_cycle_executor.py | L1226 | `getattr(order, "order_id", None)` | OrderLike ガード |

**FIXABLE (5件)**:

| # | ファイル | 行 | コード | 修正案 |
|---|---|---|---|---|
| F-1 | order_monitor.py | L127 | `getattr(regime_detector, "current_regime", None)` | Protocol 型化 `RegimeDetectorLike` |
| F-2 | order_monitor.py | L130 | `getattr(current_regime, "value", None)` | Enum `.value` 直接参照 |
| F-3 | skip_gate_evaluator.py | L1022 | `getattr(ob, "bids")` | L1021 で None チェック済み → `ob.bids` |
| F-4 | skip_gate_evaluator.py | L1023 | `getattr(ob, "asks")` | 同上 → `ob.asks` |
| F-5 | fill_config.py | L680-681 | `getattr(self, _timing_name)` | ボーダーライン: 固定3フィールドのループ |

### B. bare except 残存 — 0件 ✅

全 `except Exception:` (without `as e`) を確認:

| ファイル | 行 | 状態 |
|---|---|---|
| fill_loop_orchestrator.py | L1234 | ✅ 254# 修正済 (logger.debug) |
| event_logger.py | L97, L106 | ✅ 253# 修正済 (logger.debug) |
| resilience.py | L175 | ✅ 255# 修正済 (logger.debug) |
| pnl_measurer.py | L112 | ✅ 255# 修正済 (logger.debug) |
| lock_manager.py | L155 | ✅ 255# 修正済 (logger.debug) |
| ob_utils.py | L124, L135 | ✅ 255# 修正済 (logger.debug) |
| fill_cycle_executor.py | L1194 | ✅ 255# 修正済 (logger.debug) |
| fill_test_cli.py | L351, L424 | ✅ 既に logger.debug 付き |

### C. `# type: ignore` without specific code — 0件 ✅

全 13件が具体的エラーコード付き (`[assignment]` ×3, `[attr-defined]` ×3, `[union-attr]` ×5, `[import-untyped]` ×1, `[attr-defined]` ×1)。

### D. TODO/FIXME/HACK/XXX

| ファイル | 行 | 内容 | 影響 |
|---|---|---|---|
| scripts/v460/ml/skip_gate.py | L6 | `TODO(123#): ztb/models/ への移動検討` | ml/ 配下、lib/ 外。運用影響なし |
| fill_config.py | L440 | コメント内参照 (`253# 削除完了`) | TODO ではなく履歴コメント |

**lib/ 内の TODO: 0件** ✅

### E. `Any` 型注釈 — 0件 ✅

lib/ 配下に `: Any` 宣言なし。全て `object` / 具体型 / Protocol に置換済み。

### F. Dead/unreachable code

目視検査で明確な dead code なし。`_SIDE_MODEL_SLOTS` / `_ALT_MODEL_SLOTS` の setattr ディスパッチは動的だが全パスが到達可能。

### G. 一貫性チェック

- **bare except パターン**: 全件 `logger.debug(descriptive_msg, exc_info=True)` ✅ 統一
- **getattr 排除パターン**: 全件 `self._attr` 直接参照 ✅ 統一
- **FillTestState フィールド追加パターン**: `dataclass field` + `snapshot` + `restore` 三位一体 ✅

---

## P1 items (fix in 256#)

### P1-1: skip_gate_evaluator L1022-1023 — 冗長な getattr
```python
# L1021 で getattr(ob, "bids", None) → None チェック通過後
bids = getattr(ob, "bids")   # → ob.bids で十分
asks = getattr(ob, "asks")   # → ob.asks で十分
```
**修正**: `ob.bids` / `ob.asks` に直接変更。L1021 の None チェック後なので安全。

### P1-2: order_monitor L127/L130 — regime_detector getattr
```python
current_regime = getattr(regime_detector, "current_regime", None)
return getattr(current_regime, "value", None)
```
**修正**: `_resolve_regime_name` の `regime_detector: object | None` に対して、L126 で `hasattr` チェック済み。`regime_detector.current_regime` → `.value` の直接アクセスに変更。ただし `regime_detector` 型が `object` のため、Protocol 定義が必要。

### P1-3: fill_config.py L680-681 — 自己バリデーションの getattr
```python
for _timing_name in ("order_timeout_sec", "poll_interval_sec", "cycle_interval_sec"):
    if getattr(self, _timing_name) <= 0:
        raise ValueError(f"{_timing_name} must be > 0, got {getattr(self, _timing_name)}")
```
**修正**: 各フィールドを個別にバリデーション (3フィールドなので展開しても簡潔)。

---

## P2 items (defer)

### P2-1: God Object — fill_loop_orchestrator 2453行
MAX 1200 を 2倍超過。分割候補:
- state_persistence.py (~200行): `_build_state_snapshot` + `_restore_common_state`
- stop_conditions.py (~150行): `_check_stop_conditions` + `_check_kill_conditions`

### P2-2: skip_gate_evaluator 動的 slot dispatch (setattr)
`_load_side_models()` / `_check_and_reload_side_models()` の setattr ディスパッチ。
TypedDict + dict ルックアップへのリファクタで型安全化可能だが、動作上は問題なし。

### P2-3: skip_gate.py TODO(123#)
ml/ 配下。モデル配置の設計検討 (ztb/models/ 移動)。運用影響なし。

---

## STATISTICS

| 指標 | 件数 |
|---|---|
| Remaining getattr (legitimate) | 34 |
| Remaining getattr (fixable) | 5 |
| Remaining bare except | 0 ✅ |
| Remaining `type: ignore` without code | 0 ✅ |
| Remaining TODO in lib/ | 0 ✅ |
| Remaining `Any` type annotations | 0 ✅ |

## テスト検証

254# / 255# テスト 20件: **全件 passed** ✅
