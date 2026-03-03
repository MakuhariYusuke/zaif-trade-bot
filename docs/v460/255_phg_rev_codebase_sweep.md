# 255# Codebase Sweep Report

**日付**: 2026-03-03  
**対象**: `scripts/v460/lib/` (v460 BTC/JPY maker bot)  
**前提**: 252#–254# の修正済み項目を踏まえた残存課題

---

## P1 items (fix in 255#)

### P1-1: skip_gate_evaluator — 不要な `getattr(self, ...)` 残存 (5箇所)

`_gate_buy` / `_gate_sell` は `__init__` で必ず初期化されるため `getattr` 不要。

| 行 | コード | 修正 |
|---|---|---|
| L886 | `getattr(self, "_gate_buy", None)` | `self._gate_buy` |
| L888 | `getattr(self, "_gate_sell", None)` | `self._gate_sell` |
| L920 (×2) | `getattr(self, "_gate_buy", None) is None and getattr(self, "_gate_sell", None) is None` | `self._gate_buy is None and self._gate_sell is None` |

**理由**: 254# で orchestrator の getattr を排除したが、skip_gate_evaluator にまだ残存。型安全性と静的解析の阻害要因。

### P1-2: skip_gate_evaluator — `getattr(self._config, ...)` 不要 (1箇所)

| 行 | コード | 修正 |
|---|---|---|
| L770 | `getattr(self._config, "hot_reload_check_interval_sec", None)` | `self._config.hot_reload_check_interval_sec` |

`FillTestConfig.hot_reload_check_interval_sec` は fill_config.py L531 に `float = 120.0` でデフォルト付き宣言済み。getattr→isinstance ガードは冗長。

### P1-3: order_monitor — `getattr` + `hasattr` 混在 (3箇所)

| 行 | コード | 問題 |
|---|---|---|
| L125 | `hasattr(regime_detector, "current_regime")` | duck typing — Protocol 型化すべき |
| L127 | `getattr(regime_detector, "current_regime", None)` | 同上 |
| L130 | `getattr(current_regime, "value", None)` | FillTestRegime は `.value` 保証 |
| L156-158 | `getattr(self._config, "stale_reprice_skip_gate_offset", 0.0)` | fill_config.py L349 に宣言済み |

**修正**: `regime_detector: object | None` の型を `RegimeDetectorLike` Protocol に変更し、直接アクセスに。`stale_reprice_skip_gate_offset` は直接参照。

### P1-4: resilience.py — bare `except Exception: pass` (L175)

```python
try:
    disk = self._psutil.disk_usage(".")
    status["disk_free_gb"] = round(disk.free / (1024**3), 2)
except Exception:
    pass
```

254# で orchestrator の bare except を改善したが、resilience.py にまだ残存。`logger.debug("disk_usage check failed", exc_info=True)` に変更すべき。

### P1-5: pnl_measurer.py — bare `except Exception: continue` (L112)

```python
except Exception:
    continue
```

早期退出監視ループ内で例外を黙殺。何が起きても `continue` するため、mid price 取得エラーが観測不能。`logger.debug` 追加が必要。

### P1-6: lock_manager.py — bare `except Exception: pass` (L155)

```python
except Exception:
    pass  # heartbeat 更新失敗は致命的ではない
```

コメントで妥当性を説明済みだが、`logger.debug("lock heartbeat update failed", exc_info=True)` で可観測化すべき。254# のポリシーと一貫性。

### P1-7: ob_utils.py — bare `except Exception: return 0.0` (L124, L133)

`bid_depth_volume()` / `ask_depth_volume()` がエラーを黙殺。板取得失敗時にゼロを返すと、imbalance 計算が歪む可能性。最低限 `logger.debug` 追加。

### P1-8: fill_cycle_executor.py — bare `except Exception: pass` (L1194)

```python
except Exception:
    pass  # 板取得失敗時は前回価格でリトライ
```

リトライロジック内の OB 取得失敗。コメント付きだが `logger.debug` で観測可能にすべき。

---

## P2 items (defer)

### P2-1: Amihud illiquidity ratio 未実装

`grep "amihud"` 結果: **0 件** (lib/ 配下)。

市場理論根拠: Amihud (2002) illiquidity ratio = |return| / volume は、薄い板での adverse selection risk を補足する低コスト指標。現行の VPIN + OBI ベースの Volatility Guard を補完し、出来高急減時の spread 自動拡大に利用可能。

**現状**: 未実装。VPIN (Volume-Synchronized Probability of Informed Trading) が代替として機能中。  
**評価**: VPIN + OBI でカバー範囲は広いが、低出来高環境の追加防御として中期的に検討価値あり。

### P2-2: Avellaneda-Stoikov closed-form spread 未使用

AS 最適スプレッド公式: $\delta^* = \gamma \sigma^2 T + \frac{2}{\gamma} \ln(1 + \gamma/k)$ は直接使用されていない。

**現状**: AS **原理** は複数箇所で参照・適用済み:
- maker_price.py L177, L330, L984: 損失後指数減衰 (AS §3.2)
- phantom_position_guard.py L118: 在庫リスク低減 (AS §3.2)
- fill_loop_orchestrator.py L1865: 在庫リスク∝ボラティリティ

closed-form 公式の直接代入は coincheck BTC/JPY の薄い板・離散 tick 環境では精度が出にくく、代わりに AS 原理に基づくステージパイプライン (regime boost, VG, InvSkew, loss decay) で段階的に equivalentなスプレッド調整を実現している。

**評価**: 現行パイプラインが AS 原理を包含。closed-form 直接使用は学術的には望ましいが、実用上は低優先。

### P2-3: Gate 9 advisory-only (変更不要)

cycle_gate_aggregator.py L707–L748: Gate 9 は意図的に `blocked=False` (advisory-only)。

```python
# blocked=True にすると Gate→compute() 未実行→キャッシュ更新なし
# →永久デッドロック のフィードバックループが発生するため、
# advisory-only (blocked=False)。
```

**評価**: これは設計上正しい。blocking にするとキャッシュ更新不能でデッドロックする。advisory のまま維持。

### P2-4: God Object — fill_loop_orchestrator 2452 行 (MAX 1200超過)

docstring に `MAX LINES: 1200` と明記されているが、**現在 2452 行** (2倍超過)。

**理由**: 250#–254# の機能追加 (P/L 3分離, adverse selection tracking, frozen_side 永続化) で膨張。
次の分割候補:
- `_build_state_snapshot` / `_restore_common_state` → `state_persistence.py` (~200行)
- `_check_stop_conditions` / `_check_kill_conditions` → `stop_conditions.py` (~150行)
- adaptation ロジック → `adaptation_orchestrator.py` (~200行)

### P2-5: skip_gate_evaluator 動的 slot dispatch (setattr 残存)

`_load_side_models()` / `_check_and_reload_side_models()` で `setattr(self, attr_gate, ...)` を使用 (L497, 498, 499, 843, 855)。これは `_SIDE_MODEL_SLOTS` ループによる動的ディスパッチのため getattr/setattr は正当。ただし、`TypedDict` + 辞書ルックアップへのリファクタリングで型安全化できる。

### P2-6: Inventory Target Band 未実装

Guéant-Lehalle-Fernandez-Tapia (2013) の最適在庫帯 $q^* \in [-Q, Q]$ は直接実装されていない。

**現状**: `inventory_skewing_enabled` + `inventory_skewing_neutral_band` + `inv_decay_tau_sec` (228# C2) が等価機能を提供:
- 中立帯 (neutral_band) 内は補正なし → AS 理論の在庫中立ゾーン
- 境界超過時に線形補正 → 最適帯の簡易近似
- 228# time-decay → GLFT の情報減衰を反映

**評価**: 理論的等価物は実装済み。厳密な $q^*$ 計算は在庫量の正確な把握が前提で、残高チェックの粒度と合わない。

### P2-7: `# type: ignore` without specific error code (残存)

| ファイル | 行 | コード |
|---|---|---|
| fill_cycle_executor.py | L58 | `# type: ignore[assignment]` ✅ |
| fill_cycle_executor.py | L66 | `# type: ignore[assignment]` ✅ |
| fill_loop_orchestrator.py | L95 | `# type: ignore[assignment]` ✅ |
| event_logger.py | L117 | `# type: ignore[assignment]` ✅ |
| fill_test_cli.py | L369 | `# type: ignore[attr-defined]` ✅ |
| lock_manager.py | L14 | `# type: ignore[import-untyped]` ✅ |
| adaptation_engine.py | L263, 314 | `# type: ignore[union-attr]` ✅ |
| adaptation_engine.py | L418, 426 | `# type: ignore[attr-defined]` ✅ |
| balance_checker.py | L116, 171, 177 | `# type: ignore[union-attr]` ✅ |

**評価**: 全件が具体的エラーコード付き (`[assignment]`, `[attr-defined]`, `[union-attr]`, `[import-untyped]`)。問題なし。

### P2-8: `Any` 型の使用状況

`grep ": Any"` 結果: **0 件** (コメント内の言及のみ)。`Any` 型注釈は排除済み。

---

## Status of known P2 items from previous sweeps

| 項目 | 現状 |
|---|---|
| skip_gate.py TODO(123#) | **該当なし** — `skip_gate.py` は `scripts/v460/ml/skip_gate.py` (lib/ 外)。lib/ 内に TODO(123#) は **0件**。 |
| vg_and_trend.py TODO(123#) | `scripts/v460/analysis/vg_and_trend.py` L134: regex パース脆弱性の TODO 残存。**analysis/ は lib/ 外** → 本番影響なし。 |
| Amihud illiquidity 未実装 | **未実装** (P2-1 参照)。VPIN が代替。 |
| AS closed-form 未使用 | **原理は活用済み**、closed-form は未使用 (P2-2 参照)。 |
| Gate 9 advisory-only | **設計上正しい** — blocking は deadlock を引き起こす (P2-3 参照)。 |
| skip_gate_evaluator getattr | **4箇所残存** → P1-1, P1-2 で修正対象。 |
| DD cooldown re-arm | **249# で実装済み** — `daily_drawdown_guard.py` L55-57, L83, L98-99 に `cooldown_rearm_budget_bps`, `cooldown_rearmed` 実装。 |
| God Object regression | **悪化: 2452行** (MAX 1200) → P2-4 で分割計画。 |

---

## Summary

| 区分 | 件数 | 種別 |
|---|---|---|
| **P1 (255# fix)** | 8 | getattr排除 ×3, bare except改善 ×5 |
| **P2 (defer)** | 8 | 市場理論 ×3, God Object ×1, 型安全 ×2, type:ignore ×1 (問題なし), Any ×1 (問題なし) |

**P1 の修正方針**: 254# と同一パターン (getattr→直接参照, bare except→logger.debug+exc_info)。テスト影響は軽微。

**市場理論評価**: AS 原理・在庫中立理論は **パイプライン全体で実質的に包含済み**。Amihud は VPIN の補完として中期検討。Gate 9 advisory は構造的に正しい設計判断。
